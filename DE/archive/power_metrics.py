import subprocess
import plistlib
import threading
import time

class MacHardwareMonitor:
    def __init__(self, sample_interval_ms=500):
        self.interval = sample_interval_ms
        self.latest_stats = {
            "cpu_w": 0.0,
            "gpu_w": 0.0,
            "ane_w": 0.0,
            "gpu_util": 0.0
        }
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_sampler, daemon=True)
        self.thread.start()

    def _run_sampler(self):
        # powermetrics requires sudo execution
        cmd = [
            "sudo", "powermetrics",
            "-f", "plist",
            "-i", str(self.interval),
            "-s", "cpu_power,gpu_power"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        buf = b""
        
        while self.running:
            line = proc.stdout.readline()
            buf += line
            if b"</plist>" in line:
                try:
                    data = plistlib.loads(buf)
                    processor = data.get("processor", {})
                    gpu = data.get("gpu", {})
                    
                    self.latest_stats = {
                        "cpu_w": processor.get("cpu_power", 0.0) / 1000.0,
                        "gpu_w": processor.get("gpu_power", 0.0) / 1000.0,
                        "ane_w": processor.get("ane_power", 0.0) / 1000.0,
                        "gpu_util": gpu.get("gpu_busy", 0.0) * 100.0
                    }
                except Exception:
                    pass
                buf = b""

    def get_metrics(self):
        return self.latest_stats
