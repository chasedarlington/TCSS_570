import asyncio
import json
import numpy as np
import h5py
import pandas as pd
from aiohttp import web
from power_metrics import MacHardwareMonitor

# TUM-VIE resolution is 1280x720 (Prophesee Gen4)
W, H = 1280, 720
TIME_WINDOW_US = 20000  # 20ms frame chunks

class StreamEngine:
    def __init__(self, left_h5_path, right_h5_path, gt_csv_path):
        self.h5_l = h5py.File(left_h5_path, 'r')
        self.h5_r = h5py.File(right_h5_path, 'r')
        # vi-gt-data format: timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z

        # actualy: time(us) px py pz qx qy qz qw
        self.gt = pd.read_csv(gt_csv_path).values
        self.cursor_l = 0
        self.cursor_r = 0
        self.monitor = MacHardwareMonitor()
        self.monitor.start()

    def render_events_to_image(self, h5_obj, cursor, t_start, t_end):
        times = h5_obj['events/t']
        xs = h5_obj['events/x']
        ys = h5_obj['events/y']
        ps = h5_obj['events/p']

        # Slice contiguous segment
        end_idx = cursor
        while end_idx < len(times) and times[end_idx] < t_end:
            end_idx += 5000
        end_idx = min(end_idx, len(times))

        t_slice = times[cursor:end_idx]
        valid = (t_slice >= t_start) & (t_slice < t_end)
        idx_range = np.where(valid)[0] + cursor

        if len(idx_range) == 0:
            return np.zeros((H // 2, W // 2), dtype=np.uint8), cursor

        # Downsample 2x for low-latency WebSocket transfer
        x_sub = (xs[idx_range] // 2).astype(np.int32)
        y_sub = (ys[idx_range] // 2).astype(np.int32)
        p_sub = ps[idx_range]

        frame = np.full((H // 2, W // 2), 128, dtype=np.uint8)
        # Red/Blue or positive/negative contrast accumulation
        frame[y_sub, x_sub] = np.where(p_sub > 0, 255, 0)
        
        new_cursor = idx_range[-1] + 1 if len(idx_range) > 0 else cursor
        return frame, new_cursor

    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        t_current = self.h5_l['events/t'][0]

        try:
            while not ws.closed:
                t_next = t_current + TIME_WINDOW_US

                img_l, self.cursor_l = self.render_events_to_image(self.h5_l, self.cursor_l, t_current, t_next)
                img_r, self.cursor_r = self.render_events_to_image(self.h5_r, self.cursor_r, t_current, t_next)

                # Match nearest ground-truth pose
                gt_row = self.gt[np.abs(self.gt[:, 0] - (t_current * 1000)).argmin()]
                pose = {
                    "pos": gt_row[1:4].tolist(),
                    "quat": gt_row[4:8].tolist()
                }

                payload = {
                    "metrics": self.monitor.get_metrics(),
                    "pose": pose,
                    "img_l": img_l.tolist(),
                    "img_r": img_r.tolist(),
                    "timestamp": int(t_current)
                }

                await ws.send_str(json.dumps(payload))
                t_current = t_next
                await asyncio.sleep(0.015)  # Throttle to ~60 FPS
        finally:
            await ws.close()

        return ws

app = web.Application()
engine = StreamEngine("./mocap-6dof-events_left.h5", "./mocap-6dof-events_right.h5", "./vi_gt_data.csv")
app.router.add_get('/ws', engine.websocket_handler)
app.router.add_static('/', path='.', show_index=True)

if __name__ == '__main__':
    web.run_app(app, host='127.0.0.1', port=8080)
