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
Illustrate the capabilities and drawbacks with asynchronous, parallel event camera processing  

## ABOUT THE TUM VIE DATASET
https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset  
**hdf5, compressed bit-stream** for left and right event cameras: *x y t p* where polarity (p) subject to the formula below  
**imu_data.txt** 200 Hz linear/angular rate + temp: *time(us); gx gy gz (rad/s); ax ay az (m/s^2); temp (C)*  
**mocap_data.txt** 120 Hz position: *time(us); px py pz; qx qy qz qw*

### Continuous Log-Intensity Manifold
Analog logarithmic circuit continuously compresses dynamic range into a log-intensity signal:
$L(\mathbf{u}, t) = \ln I(\mathbf{u}, t)$

### Event Generation Condition
An individual pixel photodiode operates autonomously and asynchronously. An event $e_k = (x_k, y_k, t_k, p_k)$ is emitted at microsecond hardware timestamp $t_k$ if and only if the temporal change in log-intensity since the reference level at the previous event firing $t_{k - \Delta t}$ crosses a defined analog contrast threshold $C > 0$:\
$\Delta L(\mathbf{u}_k, t_k) = L(\mathbf{u}_k, t_k) - L(\mathbf{u}_k, t_k - \Delta t)$\
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
### Spatial Voxel (XYdT) Extraction From Asynchronous Event Camera Stream
Purpose: discretize space-time into bounded volumetric sub-domains - supporting parallel **Map pattern**. Geometric searches are constrained strictly to the points contained within a single voxel (XYdT) or its immediate neighboring cells.
*FLAG: 
###  Cache Blocking 
Purpose: tiling over 2D sensor focal plane enables parallel **Map pattern**, decoupling camera reference frame and world reference frame coordinate systems.
*FLAG: without tiling, program shifts away from arithmetic intensity / ridge point 
