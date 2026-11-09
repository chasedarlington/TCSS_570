# Instead of a fixed constant array:
# Z_cam = np.full_like(X_cam, 1.5)

# Create a probability distribution of depths for each event:
mean_depth = 1.5  # meters
std_dev = 0.2     # uncertainty standard deviation (20 cm)

# Sample a depth for each event from a Gaussian distribution
Z_cam = np.random.normal(loc=mean_depth, scale=std_dev, size=X_cam.shape)

# Ensure no negative or zero depths occur due to extreme tails
Z_cam = np.clip(Z_cam, 0.1, 10.0)
