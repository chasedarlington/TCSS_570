# TCSS_570 Async Parallel Odometry 

Set up the Python environment and run the live tracker:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python live_video_dsu_tracker.py
```

If using an IDE, select `.venv/bin/python` as the Python interpreter.
The tracker requires a graphical desktop and the HDF5 dataset at `h5_path`
in `live_video_dsu_tracker.py`.


## OBJECTIVE OF THIS PROJECT
Illustrate the capabilities and drawbacks with asynchronous, parallel event camera processing. Triangulate depth via closest approach with calibration, IMU pose, and asynchronous event camera data (coordinates constrained to edges of distinct intensity differentials) and rendering XYZt voxel point cloud over the duration of 6dof capture. 

## ABOUT THE TUM VIE DATASET
https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset  

(2) Prophesee Gen4-CD Event Cameras: async, ~1MP 

(2) Global Shutter Visual Cameras: 20 Hz, 1MP grayscale images, 3.2ms exposure

(1) Inertial Measurement Unit: 200 Hz, 6-axis IMU, 3-axis linear acceleration and angular velocity

(1) External Motion Capture System (Vicon/Qualisys): up to 120 Hz, 6-DoF rigid-body poses

**hdf5, compressed bit-stream** for left and right event cameras: *x y t p* where polarity (p) subject to the formula below  

**imu_data.txt** 200 Hz linear/angular rate + temp: *time(us); gx gy gz (rad/s); ax ay az (m/s^2); temp (C)*  

**mocap_data.txt** 120 Hz position: *time(us); px py pz; qx qy qz qw*

**camera-calibrationB.json** intrinsics of all four cameras as well as the extrinsics T_imu_cam between camera N and imu, where N = {left visual camera, right visual camera, left event camera, right event camera}: *6-DoF px py pz qx qy qz qw transformation from cam to IMU, fx fy focal length, cx cy principal point, k1..k4 barrel distortion, accel_noise_std gyro_noise_std accel_bias_std gyro_bias_std calib_accel_bias calib_gyro_bias, and HW clock latency*

**mocap-imu-calibrationB.json** extrinsics T_mocap_world T_imu_marker between the mocap poses and imu poses *translation: qx qy qz qw and mocap_to_imu_offset_ns mocap_time_offset_ns*


### Continuous Log-Intensity Manifold
Analog logarithmic circuit continuously compresses dynamic range into a log-intensity signal:

$L(\mathbf{u}, t) = \ln I(\mathbf{u}, t)$

### Event Generation Condition
An individual pixel photodiode operates autonomously and asynchronously. An event $e_k = (x_k, y_k, t_k, p_k)$ is emitted at microsecond hardware timestamp $t_k$ if and only if the temporal change in log-intensity since the reference level at the previous event firing $t_{k - \Delta t}$ crosses a defined analog contrast threshold $C > 0$:

$\Delta L(\mathbf{u}_k, t_k) = L(\mathbf{u}_k, t_k) - L(\mathbf{u}_k, t_k - \Delta t)$

$\big\vert{}\Delta L(\mathbf{u}_k, t_k)\big\vert{} \ge C$

### Polarity Formulation
$p_k = \begin{cases} +1 & \text{if } \Delta L(\mathbf{u}_k, t_k) \ge +C \quad (\text{ON event: Positive contrast / Brightness increase}) \\ -1 & \text{if } \Delta L(\mathbf{u}_k, t_k) \le -C \quad (\text{OFF event: Negative contrast / Brightness decrease}) \end{cases}$

**OR**

$p_k = \operatorname{sgn}\big(\Delta L(\mathbf{u}_k, t_k)\big) \in \{-1, +1\}$

### Edge Optical Flow
Under the brightness constancy assumption for moving contours with apparent image plane velocity $\mathbf{v} = (v_x, v_y)^T$:

$\frac{d L}{d t} = \frac{\partial L}{\partial t} + \nabla L \cdot \mathbf{v} = 0 \implies \frac{\partial L}{\partial t} = -\nabla L \cdot \mathbf{v}$

Integrating this relation across the temporal interval yields:

$\Delta L(\mathbf{u}, t) \approx -\int_{t - \Delta t}^{t} (\nabla L \cdot \mathbf{v})\, d\tau \approx -(\nabla L \cdot \mathbf{v})\Delta t$

Consequently, the event emission polarity directly mirrors the spatial image gradient aligned with sensor motion:

$p_k = -\operatorname{sgn}\big(\nabla L(\mathbf{u}_k) \cdot \mathbf{v}(t_k)\big)$

## ALGORITHM DESIGN OVERVIEW

### Calibration & Rectification

Calibrate left and right event cameras from physical sensor parameters: intrinsics (barrel distortion, k2 and k3), extrinsics (6-DoF baseline translation and relative rotation between left and right event cameras, cam2 and cam3), and temporal offsets (cam_time_offset_ns, mocap_time_offset_ns).

$x_d = \frac{x - c_x}{f_x}, \quad y_d = \frac{y - c_y}{f_y}, \quad r_d = \sqrt{x_d^2 + y_d^2}$

Rectify using calibrated parameters to re-project left/right camera planes onto common virtual coplanar image plane (parallel optical axes and aligned horizontal epipolar scanlines).

(1) Parse Calibration and Compute Relative Stereo Pose Offline, (2) Undistort Kannala-Brandt Fisheye, (3) Compute rectifying rotation matrices, (5) shift event camera timestamps by cam_time_offset_ns and MoCap timestamps by mocap_to_imu_offset_ns + mocap_time_offset_ns, (6) 

- Parse calibration.json (cam2 & cam3) and mocap_calibration.json
- Build 1280 x 720 undistortion + rectification LUTs using Newton-Raphson on kb4 (k1..k4)
- Replace raw pixel coordinates with LUT-rectified rays and align timestamps using ns offsets

### Triangulate
Filter matches along horizontal rectified scanlines and triangulate via Closest Approach

WHATS MISSING

{USING SVD, CODE IN PARALLEL FOR M6 APPLE COMPUTE}

queue by time into 5 ms windows -> associate different camera frames within time window -> compute relative pose (Rotation + Translation) -> Compute Essential Matrix -> Rectify/Match -> Closest Approach Z calculation

Calibrate -> Epipolar Rectify -> Essential Fundamental Matrix -> Optimization (Closest Approach)

### Spatial Voxel Accumulation From Asynchronous Event Camera Stream > Voxel Grid V(x,y,t) 
Discretize a rolling time window $\Delta T$ into $B$ discrete temporal bins. Each event $(x_k, y_k, t_k, p_k)$    contributes to the grid $V \in \mathbb{R}^{B \times H \times W}$ via trilinear interpolation across time:

$t_k^* = \frac{B - 1}{\Delta T} (t_k - t_0)$

For the two nearest temporal slices $b = \lfloor t_k^* \rfloor$ and $b+1$:

$V(b, y_k, x_k) {+}= p_k \cdot (1 - (t_k^* - b))$

$V(b+1, y_k, x_k) {+}= p_k \cdot (t_k^* - b)$

**note:** accumulate this in parallel using atomic float adds into a linear buffer sized $B \times H \times W$.
