# Tennis Ball Velocity Tracker

A beginner-friendly Streamlit application that tracks a green-yellow tennis ball in an uploaded video and generates a time-velocity graph.

## Features

- Upload MP4, MOV, or AVI videos.
- Display FPS, frame count, duration, and resolution.
- Detect green-yellow tennis balls with HSV threshold sliders and contour filtering.
- Track frame-by-frame centre coordinates with confidence scores.
- Interpolate short missing detections and optionally smooth positions.
- Calculate horizontal, vertical, and resultant velocity in pixels per second.
- Optionally calibrate velocity to metres per second, for example `1 metre = 250 pixels`.
- Download a velocity graph as PNG.
- Download tracking data as CSV.
- Download an annotated video with the detected ball, trajectory, and speed overlay.

## Setup

1. Create and activate a Python virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open the Streamlit URL shown in your terminal, upload a tennis-ball video, adjust thresholds if needed, and click **Process video**.

## Notes on velocity

Velocity is calculated from frame-by-frame displacement divided by the frame time interval:

```text
v = Δposition / Δtime
Δtime = 1 / FPS
```

Image coordinates increase downward, so the app inverts the y-axis when calculating vertical velocity. Upward motion is therefore shown as positive vertical velocity.
