# TCSS 570 Async Parallel Odometry

This project explores asynchronous, parallel processing for stereo event cameras. The reconstruction design uses calibration and time-aligned rig poses to match events, triangulate 3D points by closest approach, and accumulate a time-stamped point cloud over a six-degree-of-freedom capture.

## Setup

From the project root, set up the Python environment and run the live tracker.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python file.py
```

The tracker requires a graphical desktop and the HDF5 file configured at `h5_path` in `file.py`. In an IDE, select `.venv/bin/python` as the interpreter. For stereo reconstruction, use the left and right event HDF5 files and calibration from the same TUM-VIE sequence.

## TUM-VIE dataset

| Input | Format and contents |
| --- | --- |
| L/R Event Cameras (\.HDF5) | Asynchronous Prophesee Gen4 CD events at 1280 × 720. The `events` group contains `x`, `y`, `t` (µs), and `p` (stored polarity). `ms_to_idx` indexes the stream by millisecond. |
| L/R Visual Cameras (.JPG) | Two 1024 × 1024 global-shutter grayscale cameras at 20 Hz. Use the sequence’s exposure metadata. |
| `imu_data.txt` | Nominally 200 Hz. Whitespace-separated `t gx gy gz ax ay az temp`. Time is in µs, angular rate in rad/s, acceleration in m/s², and temperature in °C. The released `mocap-6dof` file contains calibrated IMU measurements. |
| `mocap_data.txt` | OptiTrack Flex13 ground-truth poses at 120 Hz. The whitespace-separated columns are `t px py pz qx qy qz qw`. Time is in µs, translation in metres, and quaternion is normalized unit vector, scalar-last. |

The supplied `mocap_data.txt` matches the 2,338 pose rows of the released `mocap-6dof` sequence. It spans 1.079727 s to 20.554727 s on the published time axis, with steps of 8,333 or 8,334 µs. Check pose coverage before interpolation because some sequences provide ground truth only near the beginning and end. [Dataset and file format](https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset) · [TUM-VIE paper, Table I and §III](https://cvg.cit.tum.de/_media/spezial/bib/klenk2021tumvie.pdf) · [`mocap-6dof` VI and ground-truth archive](https://tumevent-vi.vision.in.tum.de/mocap-6dof/mocap-6dof-vi_gt_data.tar.gz)

## Calibration B

Use Calibration B for `mocap-6dof`. TUM lists Calibration A for `loop-floor0` through `loop-floor3`, `mocap-desk`, `mocap-desk2`, and `skate-easy`, and Calibration B for the other sequences in its calibration listing. [TUM-VIE calibration listing](https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset#calibration)

### Camera and IMU calibration

Read `camera-calibrationB.json` under its `value0` key. The entries of `intrinsics`, `resolution`, and `T_imu_cam` share this order.

| Index | Camera | Resolution (width × height) |
| --- | --- | --- |
| 0 | Left visual (`cam0`) | 1024 × 1024 |
| 1 | Right visual (`cam1`) | 1024 × 1024 |
| 2 | Left event (`cam2`) | 1280 × 720 |
| 3 | Right event (`cam3`) | 1280 × 720 |

Each `intrinsics[i]` declares `camera_type: "kb4"`. Its nested `intrinsics` object stores `fx`, `fy`, `cx`, and `cy` in pixels and all four Kannala–Brandt coefficients, `k1` through `k4`. Each `T_imu_cam[i]` stores the camera-to-IMU transform as a translation `px, py, pz` in metres and a quaternion `qx, qy, qz, qw`.

The file also supplies `imu_update_rate` (200.843139 Hz), the 9- and 12-element IMU calibration vectors `calib_accel_bias` and `calib_gyro_bias`, and three-axis noise and bias standard deviations in `accel_noise_std`, `gyro_noise_std`, `accel_bias_std`, and `gyro_bias_std`. Use the released calibrated IMU measurements directly. In Calibration B, `vignette` is empty. [Official camera calibration B](https://tumevent-vi.vision.in.tum.de/camera-calibrationB.json)

### Mocap transforms and time alignment

Use the convention $T_{A\leftarrow B}$ for a transform from frame $B$ to frame $A$. In `mocap-imu-calibrationB.json`, `value0.T_imu_marker` maps marker coordinates into the IMU frame, and `value0.T_mocap_world` maps calibration-world coordinates into the mocap frame. Both use the same translation and quaternion layout as the camera extrinsics.

Let $M$, $B$, $W$, $I$, and $C_i$ denote the mocap, marker body, calibration world, IMU, and camera frames. Each mocap row gives the marker pose $T_{M\leftarrow B}(t)$. To express camera $i$ in the calibration world, compose

$$
T_{W\leftarrow C_i}(t) =
T_{M\leftarrow W}^{-1}\,T_{M\leftarrow B}(t)\,T_{I\leftarrow B}^{-1}\,T_{I\leftarrow C_i}.
$$

The product without the leading $T_{M\leftarrow W}^{-1}$ gives the camera pose in mocap coordinates. Interpolate translation linearly and orientation with shortest-path quaternion SLERP between bracketing samples. Skip world-frame placement for intervals without pose coverage unless an independent pose estimate is available. [Official mocap–IMU calibration B](https://tumevent-vi.vision.in.tum.de/mocap-imu-calibrationB.json) · [Basalt calibration implementation](https://github.com/VladyslavUsenko/basalt/blob/master/include/basalt/optimization/spline_linearize.h#L514-L522)

The published event, IMU, and ground-truth timestamps already share the left event camera’s clock. Preserve these timestamps when joining released files and use one unit throughout. A 5 ms window is 5,000 µs. The calibration files record `cam_time_offset_ns = 5114000`, `mocap_to_imu_offset_ns = -217102426153`, and `mocap_time_offset_ns = -9879790`. These values are for calibration and custom alignment workflows with known raw clock origins. Applying them to the published timestamps would shift already aligned streams. [TUM-VIE paper, §IV-C](https://cvg.cit.tum.de/_media/spezial/bib/klenk2021tumvie.pdf)

## Event generation

**An individual pixel photodiode operates autonomously and asynchronously.**

In the ideal contrast-threshold model, a pixel emits an event $e_k=(x_k,y_k,t_k,p_k)$ when the difference $\Delta L=L(\mathbf{u}_k,t_k)-L(\mathbf{u}_k,t_{\mathrm{prev}})$ reaches a threshold in either direction; $\mathbf{u} = (x,y)^T$. Here $L(\mathbf{u},t)=\ln I(\mathbf{u},t)$, where $I(\mathbf{u},t)$ is the instantaneous scene irradiance arriving at pixel coordinate ${u} = (x,y)^T$ with $\ln$ creating a log-intensity manifold, and $t_{\mathrm{prev}}$ is the previous event time at that pixel. The mathematical polarity is $+1$ for an ON event ($\Delta L\ge C$) and $-1$ for an OFF event ($\Delta L\le -C$), with $C>0$. On Prophesee Gen4 sensors, $C$ is governed by analog comparator bias voltages (bias_diff_on, bias_diff_off) which vary slightly across pixels. $p_k$ stores `0` as `pos` and `1` as `neg`. [TUM HDF5 example](https://tumevent-vi.vision.in.tum.de/access_visualize_h5.py)

For a moving contour under brightness constancy, $\partial_t L=-\nabla L\cdot\mathbf{v}$. Over a short interval $\Delta t$, $\Delta L\approx-(\nabla L\cdot\mathbf{v})\Delta t$. This explains why moving edges generate events and why a single edge constrains only the flow component normal to the edge.

## Stereo reconstruction design

### Undistort and rectify

1. Use `cam2` and `cam3` for event stereo. At a common timestamp, the transform from the left event camera $L$ to the right event camera $R$ is $T_{R\leftarrow L}=T_{I\leftarrow R}^{-1}T_{I\leftarrow L}$. The norm of its translation gives a Calibration B event baseline of about 0.11882 m.
2. For a raw pixel $(u,v)$, form $x_d=(u-c_x)/f_x$, $y_d=(v-c_y)/f_y$, and $r_d=\sqrt{x_d^2+y_d^2}$. Invert the KB4 model for the bearing angle $\theta$ using Newton–Raphson on $r_d=\theta+k_1\theta^3+k_2\theta^5+k_3\theta^7+k_4\theta^9$. For $r_d>0$, the unit ray is $(x_d\sin\theta/r_d,\ y_d\sin\theta/r_d,\ \cos\theta)$. At $r_d=0$, use $(0,0,1)$. Mark failed inversions invalid.
3. Rotate the rays into virtual cameras with a shared pinhole projection and parallel optical axes. In a common frame, set the rectified x-axis from the left centre to the right centre. Set the z-axis along the mean optical direction projected perpendicular to x, then set y by $\hat{\mathbf{y}}=\hat{\mathbf{z}}\times\hat{\mathbf{x}}$. Precompute a lookup table for each 1280 × 720 raw event grid, storing each bearing, its rectified coordinates, and its validity. Reject rays behind the virtual image plane and projections outside the rectified image.

### Associate events in time

Group each sensor’s events into 5 ms windows for processing. For near-synchronous events, search along rectified scanlines with a y tolerance, a timestamp tolerance, and a disparity range consistent with the chosen depth bounds. Score local event support and reject ambiguous matches. Include neighboring-window events when needed to search the full time tolerance at a boundary.

For larger timestamp differences or appreciable rig motion, use the pose at each event time to compute $T_{R(t_R)\leftarrow L(t_L)}=T_{W\leftarrow R}(t_R)^{-1}T_{W\leftarrow L}(t_L)$. For a static scene point, the corresponding bearings in their camera frames satisfy $\mathbf{d}_R^\top E\mathbf{d}_L=0$. Compute $E=[\mathbf{t}_{RL}]_\times R_{RL}$ from the relative translation and rotation. For pinhole pixel coordinates in those same frames, use $F=K_R^{-T}EK_L^{-1}$. Use the time-dependent epipolar constraint for these matches and fixed rectified scanlines for the near-synchronous case.

### Triangulate by closest approach

Express both camera centres $\mathbf{o}_L,\mathbf{o}_R$ and unit rays $\mathbf{d}_L,\mathbf{d}_R$ in one frame. Use the left-camera frame for near-synchronous stereo or the world frame with camera poses evaluated at their own timestamps. Find the depths along the rays by solving the following least-squares system with SVD.

$$
\begin{bmatrix}\mathbf{d}_L & -\mathbf{d}_R\end{bmatrix}
\begin{bmatrix}\lambda_L\\ \lambda_R\end{bmatrix}
\simeq \mathbf{o}_R-\mathbf{o}_L.
$$

Form the two closest points $\mathbf{P}_i=\mathbf{o}_i+\lambda_i\mathbf{d}_i$ and take their midpoint as the reconstruction. Use the singular values to reject nearly parallel rays. Reject nonpositive depths, excessive closest-point separation in metres, or excessive reprojection error in pixels. Timestamp the point at the midpoint of the two event times. For near-synchronous matches solved in the left-camera frame, use the left-camera pose at this representative time to place the point in the common world frame before accumulating $(X,Y,Z,t)$.

After matching, each ray pair can be solved independently. On Apple silicon, batch the $3\times2$ systems for SVD with NumPy, then merge accepted points and render them in timestamp order.

### Accumulate event voxels

For an image-space event volume $V\in\mathbb{R}^{B\times H\times W}$ with $B\ge2$, use signed polarity and interpolate between temporal bins. For a window of duration $\Delta T$ starting at $t_0$, compute $s=(B-1)(t-t_0)/\Delta T$. At an integer pixel $(x,y)$, add $p(1-\alpha)$ to $V[b,y,x]$ and $p\alpha$ to $V[b+1,y,x]$, where $b=\lfloor s\rfloor$ and $\alpha=s-b$. Process only in-window events and guard the upper bin index. Use bilinear spatial weights as well if retaining subpixel rectified coordinates. To accumulate in parallel, reduce worker-local grids or use atomic additions where supported.