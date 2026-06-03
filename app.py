"""Streamlit tennis-ball tracker.

Run with:
    streamlit run app.py

The app detects a green-yellow tennis ball in an uploaded video, tracks its
position frame-by-frame, estimates velocity, plots time against velocity, and
exports both tracking data and an annotated preview video.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class VideoInfo:
    """Basic metadata read from a video file."""

    fps: float
    total_frames: int
    duration: float
    width: int
    height: int


@dataclass
class DetectionSettings:
    """Parameters used by colour/contour based ball detection."""

    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    min_area: int
    max_area: int
    max_tracking_jump: float


@dataclass
class DetectionResult:
    """Single-frame tennis-ball detection result."""

    center: tuple[float, float] | None
    radius: float | None
    confidence: float
    contour_area: float


@st.cache_data(show_spinner=False)
def read_video_bytes(uploaded_file_name: str, uploaded_file_bytes: bytes) -> bytes:
    """Cache uploaded bytes so Streamlit reruns do not reread the upload."""

    del uploaded_file_name  # The name is present only to vary the cache key.
    return uploaded_file_bytes


def save_upload_to_tempfile(video_bytes: bytes, suffix: str) -> str:
    """Save uploaded video bytes to a temporary file and return the path."""

    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(video_bytes)
        return temp_file.name


def load_video(video_path: str | Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    """Open a video and return both the capture object and metadata."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open the uploaded video.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or math.isnan(fps):
        raise ValueError("The uploaded video does not report a valid FPS value.")

    duration = total_frames / fps if total_frames > 0 else 0.0
    return capture, VideoInfo(fps=fps, total_frames=total_frames, duration=duration, width=width, height=height)


def detect_ball(
    frame: np.ndarray,
    settings: DetectionSettings,
    previous_center: tuple[float, float] | None = None,
) -> DetectionResult:
    """Detect the most likely tennis ball in one frame using HSV and contours."""

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(settings.hsv_lower, dtype=np.uint8)
    upper = np.array(settings.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv_frame, lower, upper)

    # Reduce noise and close small holes inside the ball mask.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_result = DetectionResult(center=None, radius=None, confidence=0.0, contour_area=0.0)
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < settings.min_area or area > settings.max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        (x, y), radius = cv2.minEnclosingCircle(contour)
        enclosing_area = np.pi * radius * radius if radius > 0 else 1.0
        fill_ratio = float(min(area / enclosing_area, 1.0))

        # Area is normalized around the centre of the accepted range so tiny
        # specks and huge blobs do not dominate the selection.
        area_midpoint = (settings.min_area + settings.max_area) / 2.0
        area_score = 1.0 - min(abs(area - area_midpoint) / max(area_midpoint, 1.0), 1.0)

        continuity_score = 0.5
        if previous_center is not None:
            distance = float(np.hypot(x - previous_center[0], y - previous_center[1]))
            continuity_score = 1.0 - min(distance / max(settings.max_tracking_jump, 1.0), 1.0)

        score = (
            0.40 * min(max(circularity, 0.0), 1.0)
            + 0.25 * fill_ratio
            + 0.20 * continuity_score
            + 0.15 * area_score
        )

        if score > best_score:
            best_score = score
            confidence = float(np.clip(score, 0.0, 1.0))
            best_result = DetectionResult(center=(float(x), float(y)), radius=float(radius), confidence=confidence, contour_area=float(area))

    return best_result


def interpolate_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Interpolate missing values only when the consecutive gap is short."""

    if max_gap <= 0:
        return series
    return series.interpolate(method="linear", limit=max_gap, limit_area="inside")


def smooth_positions(data: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """Apply centred moving-average smoothing to tracked x/y positions."""

    smoothed = data.copy()
    window_size = max(3, int(window_size))
    if window_size % 2 == 0:
        window_size += 1

    for column in ("x_pixels", "y_pixels"):
        smoothed[column] = smoothed[column].rolling(window=window_size, min_periods=1, center=True).mean()
    return smoothed


def track_ball(
    video_path: str | Path,
    settings: DetectionSettings,
    interpolate_gaps: bool = True,
    max_interpolation_gap: int = 5,
    apply_smoothing: bool = False,
    smoothing_window: int = 5,
) -> tuple[pd.DataFrame, VideoInfo]:
    """Track the tennis ball through every frame of the uploaded video."""

    capture, video_info = load_video(video_path)
    rows: list[dict[str, float | int | None]] = []
    previous_center: tuple[float, float] | None = None

    frame_number = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        detection = detect_ball(frame, settings, previous_center)
        if detection.center is not None:
            previous_center = detection.center
            x_value, y_value = detection.center
        else:
            x_value, y_value = None, None

        rows.append(
            {
                "frame": frame_number,
                "time_seconds": frame_number / video_info.fps,
                "x_pixels": x_value,
                "y_pixels": y_value,
                "detection_confidence": detection.confidence,
            }
        )
        frame_number += 1

    capture.release()

    if not rows:
        raise ValueError("No frames could be read from the uploaded video.")

    data = pd.DataFrame(rows)
    if interpolate_gaps:
        data["x_pixels"] = interpolate_short_gaps(data["x_pixels"], max_interpolation_gap)
        data["y_pixels"] = interpolate_short_gaps(data["y_pixels"], max_interpolation_gap)

    if apply_smoothing:
        data = smooth_positions(data, smoothing_window)

    return data, video_info


def calculate_velocity(data: pd.DataFrame, fps: float, pixels_per_meter: float | None = None) -> pd.DataFrame:
    """Calculate horizontal, vertical, and resultant velocity from positions."""

    velocity_data = data.copy()
    dt = 1.0 / fps

    velocity_data["vx_pixels_per_second"] = velocity_data["x_pixels"].diff() / dt
    # Image y coordinates increase downward. Multiplying by -1 makes upward
    # motion positive in the physical interpretation of vertical velocity.
    velocity_data["vy_pixels_per_second"] = -(velocity_data["y_pixels"].diff() / dt)
    velocity_data["speed_pixels_per_second"] = np.hypot(
        velocity_data["vx_pixels_per_second"],
        velocity_data["vy_pixels_per_second"],
    )

    if pixels_per_meter and pixels_per_meter > 0:
        velocity_data["vx_meters_per_second"] = velocity_data["vx_pixels_per_second"] / pixels_per_meter
        velocity_data["vy_meters_per_second"] = velocity_data["vy_pixels_per_second"] / pixels_per_meter
        velocity_data["speed_meters_per_second"] = velocity_data["speed_pixels_per_second"] / pixels_per_meter

    return velocity_data


def generate_graph(data: pd.DataFrame, calibrated: bool = False) -> tuple[plt.Figure, bytes]:
    """Generate a time-velocity graph and return the figure plus PNG bytes."""

    if calibrated:
        vx_column = "vx_meters_per_second"
        vy_column = "vy_meters_per_second"
        speed_column = "speed_meters_per_second"
        y_label = "Velocity (m/s)"
    else:
        vx_column = "vx_pixels_per_second"
        vy_column = "vy_pixels_per_second"
        speed_column = "speed_pixels_per_second"
        y_label = "Velocity (pixels/s)"

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(data["time_seconds"], data[vx_column], label="Horizontal velocity", linewidth=1.8)
    axis.plot(data["time_seconds"], data[vy_column], label="Vertical velocity (upward +)", linewidth=1.8)
    axis.plot(data["time_seconds"], data[speed_column], label="Resultant speed", linewidth=2.2)
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel(y_label)
    axis.set_title("Tennis Ball Velocity vs Time")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
        figure.savefig(temp_image.name, dpi=160, bbox_inches="tight")
        temp_image.seek(0)
        png_bytes = Path(temp_image.name).read_bytes()

    return figure, png_bytes


def iter_valid_points(data: pd.DataFrame) -> Iterable[tuple[int, int]]:
    """Yield valid integer trajectory points from a tracking DataFrame."""

    valid_points = data[["x_pixels", "y_pixels"]].dropna()
    for x_value, y_value in valid_points.itertuples(index=False):
        yield int(round(x_value)), int(round(y_value))


def create_annotated_video(
    input_video_path: str | Path,
    tracking_data: pd.DataFrame,
    output_video_path: str | Path,
) -> str:
    """Create an annotated video with ball marker, trajectory, and velocity text."""

    capture, video_info = load_video(input_video_path)
    output_path = str(output_video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, video_info.fps, (video_info.width, video_info.height))

    if not writer.isOpened():
        capture.release()
        raise ValueError("Could not create the annotated video file.")

    trajectory: list[tuple[int, int]] = []
    frame_number = 0

    while True:
        success, frame = capture.read()
        if not success or frame_number >= len(tracking_data):
            break

        row = tracking_data.iloc[frame_number]
        x_value = row.get("x_pixels")
        y_value = row.get("y_pixels")

        if pd.notna(x_value) and pd.notna(y_value):
            center = (int(round(x_value)), int(round(y_value)))
            trajectory.append(center)
            cv2.circle(frame, center, 14, (0, 255, 255), 3)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)

        if len(trajectory) > 1:
            cv2.polylines(frame, [np.array(trajectory, dtype=np.int32)], False, (255, 0, 0), 2)

        speed = row.get("speed_pixels_per_second")
        velocity_text = "Speed: -- px/s" if pd.isna(speed) else f"Speed: {speed:.1f} px/s"
        cv2.putText(frame, velocity_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(frame, velocity_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Time: {row['time_seconds']:.2f}s",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(frame, f"Time: {row['time_seconds']:.2f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_number += 1

    capture.release()
    writer.release()
    return output_path


def export_csv(data: pd.DataFrame) -> bytes:
    """Export tracking data to CSV bytes."""

    return data.to_csv(index=False).encode("utf-8")


def tracking_quality_message(data: pd.DataFrame) -> tuple[str, str] | None:
    """Return a Streamlit message level and text for tracking quality warnings."""

    detection_rate = float((data["detection_confidence"] > 0).mean()) if len(data) else 0.0
    average_confidence = float(data["detection_confidence"].mean()) if len(data) else 0.0

    if detection_rate < 0.35:
        return "error", (
            f"Tracking quality is poor: the ball was detected in only {detection_rate:.0%} of frames. "
            "Try widening the HSV range, improving lighting, or changing the contour area limits."
        )
    if detection_rate < 0.65 or average_confidence < 0.45:
        return "warning", (
            f"Tracking quality is moderate: detection rate {detection_rate:.0%}, average confidence {average_confidence:.2f}. "
            "Review the annotated video and adjust thresholds if needed."
        )
    return "success", f"Tracking quality looks good: detection rate {detection_rate:.0%}, average confidence {average_confidence:.2f}."


def render_video_info(video_info: VideoInfo) -> None:
    """Display basic uploaded-video metadata in Streamlit."""

    cols = st.columns(4)
    cols[0].metric("FPS", f"{video_info.fps:.2f}")
    cols[1].metric("Frames", f"{video_info.total_frames:,}")
    cols[2].metric("Duration", f"{video_info.duration:.2f}s")
    cols[3].metric("Resolution", f"{video_info.width} × {video_info.height}")


def main() -> None:
    """Streamlit entry point."""

    st.set_page_config(page_title="Tennis Ball Velocity Tracker", page_icon="🎾", layout="wide")
    st.title("🎾 Tennis Ball Velocity Tracker")
    st.write(
        "Upload a bouncing tennis-ball video, tune the green-yellow colour detection if needed, "
        "and generate position data, velocity curves, and an annotated preview video."
    )

    uploaded_file = st.file_uploader("Upload an MP4, MOV, or AVI video", type=["mp4", "mov", "avi"])

    with st.sidebar:
        st.header("Detection settings")
        st.caption("Defaults target the green-yellow colour of a tennis ball in HSV space.")
        h_min, h_max = st.slider("Hue range", 0, 179, (25, 75))
        s_min, s_max = st.slider("Saturation range", 0, 255, (80, 255))
        v_min, v_max = st.slider("Value / brightness range", 0, 255, (80, 255))
        min_area = st.slider("Minimum contour area", 5, 5000, 50, step=5)
        max_area = st.slider("Maximum contour area", 100, 50000, 5000, step=100)
        max_jump = st.slider("Maximum tracking jump (pixels)", 10, 500, 120, step=10)

        st.header("Interpolation and smoothing")
        interpolate_gaps = st.checkbox("Interpolate short missing gaps", value=True)
        max_gap = st.slider("Maximum interpolation gap (frames)", 1, 30, 5)
        apply_smoothing = st.checkbox("Smooth positions with moving average", value=False)
        smoothing_window = st.slider("Smoothing window (frames)", 3, 21, 5, step=2)

        st.header("Calibration")
        use_calibration = st.checkbox("Convert pixels/s to metres/s", value=False)
        pixels_per_meter = None
        if use_calibration:
            pixels_per_meter = st.number_input("Pixels per metre", min_value=0.1, value=250.0, step=10.0)

    if uploaded_file is None:
        st.info("Upload a tennis-ball video to begin.")
        return

    video_bytes = read_video_bytes(uploaded_file.name, uploaded_file.getvalue())
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    input_video_path = save_upload_to_tempfile(video_bytes, suffix)

    try:
        preview_capture, preview_info = load_video(input_video_path)
        preview_capture.release()
        st.subheader("Video information")
        render_video_info(preview_info)
        st.video(video_bytes)
    except Exception as exc:  # noqa: BLE001 - show friendly Streamlit message for upload errors.
        st.error(f"Could not read video information: {exc}")
        return

    process = st.button("Process video", type="primary")
    if not process:
        st.info("Adjust detection settings if needed, then click **Process video**.")
        return

    settings = DetectionSettings(
        hsv_lower=(h_min, s_min, v_min),
        hsv_upper=(h_max, s_max, v_max),
        min_area=min_area,
        max_area=max_area,
        max_tracking_jump=max_jump,
    )

    try:
        with st.spinner("Tracking tennis ball frame by frame..."):
            tracking_data, video_info = track_ball(
                input_video_path,
                settings,
                interpolate_gaps=interpolate_gaps,
                max_interpolation_gap=max_gap,
                apply_smoothing=apply_smoothing,
                smoothing_window=smoothing_window,
            )
            tracking_data = calculate_velocity(tracking_data, video_info.fps, pixels_per_meter if use_calibration else None)

        quality = tracking_quality_message(tracking_data)
        if quality:
            level, message = quality
            getattr(st, level)(message)

        st.subheader("Time–velocity graph")
        figure, graph_png = generate_graph(tracking_data, calibrated=bool(use_calibration and pixels_per_meter))
        st.pyplot(figure)
        plt.close(figure)

        st.download_button(
            "Download graph as PNG",
            data=graph_png,
            file_name="tennis_ball_velocity_graph.png",
            mime="image/png",
        )

        st.subheader("Tracking data")
        st.dataframe(tracking_data, use_container_width=True)
        st.download_button(
            "Download tracking data as CSV",
            data=export_csv(tracking_data),
            file_name="tennis_ball_tracking_data.csv",
            mime="text/csv",
        )

        st.subheader("Annotated video preview")
        with st.spinner("Creating annotated video..."):
            output_path = str(Path(tempfile.gettempdir()) / f"annotated_{Path(uploaded_file.name).stem}.mp4")
            create_annotated_video(input_video_path, tracking_data, output_path)
            annotated_bytes = Path(output_path).read_bytes()

        st.video(annotated_bytes)
        st.download_button(
            "Download annotated video",
            data=annotated_bytes,
            file_name="annotated_tennis_ball_video.mp4",
            mime="video/mp4",
        )

    except Exception as exc:  # noqa: BLE001 - keep the beginner-facing app graceful.
        st.error(f"Processing failed: {exc}")
        st.stop()


if __name__ == "__main__":
    main()
