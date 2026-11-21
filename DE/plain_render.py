import h5py
import hdf5plugin  # Required to handle compressed datasets in TUM-VIE files
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

file_path = '/home/chase-darlington/Documents/git/TCSS_570/mocap-6dof-events_left.h5'

with h5py.File(file_path, 'r') as f:
    print("=== Top-level keys (f.keys()) ===")
    print(list(f.keys()))
    
    print("\n=== Full structural tree (f.visititems) ===")
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: /{name} | Shape: {obj.shape} | Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group:   /{name}")
            
    f.visititems(print_structure)
    
    # Slice a small portion of the x and y coordinate datasets
    x = f['events/x'][397651000:397651000+100000] # 0:568074201  note: jpg 0:389
    y = f['events/y'][397651000:397651000+100000] # 0:568074201

    #plt.figure(figsize=(10, 6))
    #plt.hist2d(x, y, bins=(1280, 720), cmap='viridis_r') # norm=LogNorm(vmin=1, vmax=100), cmin=10
    #plt.gca().invert_yaxis()
    #plt.title('TUM-VIE Binned Event Density Map')
    #plt.colorbar(label='Event Count')
    #plt.show()

    # Render a simple scatter plot of the events
    plt.figure(figsize=(10, 6))
    #plt.scatter(x[::100], y[::100], s=0.1, c='black', alpha=0.3)
    plt.scatter(x, y, s=0.5, c='black', alpha=0.3)
    plt.gca().invert_yaxis()  # Match standard image coordinate space
    plt.title('TUM-VIE Event Stream Render')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()

    


 
#  If you want to jump straight to the events that occurred at, say, $t = 1500\text{ ms}$ into the recording, you inspect the value stored at f['ms_to_idx'][1500] to find the exact starting index in /events/t.
