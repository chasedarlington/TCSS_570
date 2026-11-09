# TCSS_570

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
