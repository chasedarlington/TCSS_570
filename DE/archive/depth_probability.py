# Define discrete depth hypotheses (e.g., 32 depth planes from 0.5m to 5.0m)
depth_bins = np.linspace(0.5, 5.0, 32)

# For each event, initialize a uniform or softmax probability distribution across depth bins
# Shape: (Num_Events, Num_Depth_Bins)
depth_logits = np.random.randn(len(X_cam), len(depth_bins))

# Apply Softmax along the depth axis to form valid probabilities (sum to 1.0 per event)
exp_logits = np.exp(depth_logits - np.max(depth_logits, axis=1, keepdims=True))
depth_probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Expected Depth (Bayesian Mean) per event
Z_cam = np.sum(depth_probabilities * depth_bins, axis=1)

