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

### Spatial Voxel Accumulation From Asynchronous Event Camera Stream > Voxel Grid V(x,y,t) 
Discretize a rolling time window $\Delta T$ into $B$ discrete temporal bins. Each event $(x_k, y_k, t_k, p_k)$    contributes to the grid $V \in \mathbb{R}^{B \times H \times W}$ via trilinear interpolation across time:

$t_k^* = \frac{B - 1}{\Delta T} (t_k - t_0)$

For the two nearest temporal slices $b = \lfloor t_k^* \rfloor$ and $b+1$:

$V(b, y_k, x_k) {+}= p_k \cdot (1 - (t_k^* - b))$

$V(b+1, y_k, x_k) {+}= p_k \cdot (t_k^* - b)$

**note:** accumulate this in parallel using atomic float adds into a linear buffer sized $B \times H \times W$.

### Spatiotemporal Gradient Tensor > 3D Structure Tensor
To detect corners, compute the spatial and temporal gradients within the voxel volume. For each temporal slice $b$, compute the spatial derivatives using $3 \times 3$ Sobel or central-difference kernels:
$I_x = \frac{\partial V}{\partial x}, \quad I_y = \frac{\partial V}{\partial y}, \quad I_t = \frac{\partial V}{\partial t}$\
Assemble the 2D spatial structure tensor (or 3D spatio-temporal structure tensor) over a local Gaussian neighborhood $W$ of size $w \times w$:\
$M = \sum_{(u, v) \in W} w(u, v) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$\

### Corner/Arc Evaluation
Evaluate the corner response $R$ per pixel per slice using the Harris criterion:
$$R(x, y, b) = \det(M) - \kappa \cdot (\text{tr}(M))^2$$  
Where:
$$\det(M) = M_{00} M_{11} - M_{01}^2$$
$$\text{tr}(M) = M_{00} + M_{11}$$
$$\kappa \in [0.04, 0.06]$$ 
A cell is a corner candidate if $R(x, y, b) > \tau$  , where $\tau$ is an empirical threshold tuned to filter out straight edge segments.

### Suppression
Raw responses produce dense clusters around features. Apply a $3 \times 3$ (or $5 \times 5$) window suppression kernel:
A candidate at $(x, y, b)$   survives if:

$$R(x, y, b) = \max_{(dx, dy) \in [-1, 1]} R(x + dx, y + dy, b)$$    
Compact the surviving indices into a contiguous list of corner coordinates:

$$\mathcal{C} = \{(x_i, y_i, b_i, R_i)\}$$   

Parallel Implementation: Use an atomic counter or prefix sum (stream compaction) to write the surviving points into a compact output buffer.

### Rasterization
To visualize the detected corners, render them over a background image (either an integrated time-surface or an accumulated grayscale frame):
Background Canvas: Render the active slice $V(b, :, :)$   normalized to $[0, 1]$  as a grayscale texture.
Instanced Splatting:
Treat each corner in $\mathcal{C}$ as an instance.
Draw a small circle, crosshair, or quad at screen-space position $(x_i, y_i)$ .
Encode the polarity, timestamp, or corner response score $R_i$ into the vertex color:
Green/Cyan: High confidence corner.
Red/Blue: Positive vs. negative polarity dominance.
Alpha Blending: Use additive or standard alpha blending (src_alpha, 1 - src_alpha) to composite the markers over the background canvas.

###  Cache Blocking 
Purpose: tiling over 2D sensor focal plane enables parallel **Map pattern**, decoupling camera reference frame and world reference frame coordinate systems.
*FLAG: without tiling, program shifts away from arithmetic intensity / ridge point 





### WORK IN PROGRESS

bit stream -> spatiotemporal voxels



establish starting view plane, collecting points (async) on that coordinate frame
identify candidate 2D lines
establish fitment probabilities
identify corners
wrap-merge
convert to vertex buffer object
