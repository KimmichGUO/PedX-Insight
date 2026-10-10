"""Benchmark-only shim: make cv2.destroyAllWindows() a no-op when OpenCV has no GUI backend.

Loaded automatically (Python's `sitecustomize` hook) ONLY by the PedX subprocesses that
validation/jaad/run_pedx.py launches, via PYTHONPATH=<this directory>. It is never on the
production path.

Why: the environment has opencv-python, opencv-contrib-python AND opencv-python-headless
5.0.0.93 installed into one cv2/ package (all three dated 2026-08-11). Once torch or
ultralytics is imported, cv2.destroyAllWindows() raises cv2.error ("The function is not
implemented ... GTK+ 2.x or Cocoa support"). crosswalk.py:115, traffic_sign.py:80 and
sidewalk_detect.py:136 call it unconditionally after the detection loop, so those modes
crash -- crosswalk and traffic_sign BEFORE writing their CSV. destroyAllWindows only closes
HighGUI windows (none are open: show=False), so swallowing that specific error changes no
computed output. Every other cv2 error still propagates. The bug is reported in
validation/results/JAAD_REPORT.md; the modules are deliberately left unmodified.
"""

try:
    import cv2 as _cv2

    _orig_destroy_all_windows = _cv2.destroyAllWindows

    def _destroy_all_windows_safe():
        try:
            _orig_destroy_all_windows()
        except _cv2.error:
            pass

    _cv2.destroyAllWindows = _destroy_all_windows_safe
except Exception:  # never let the shim itself break a run
    pass
