# Real-Time Cover Drive Analysis

## What this does (Base)
- Processes a full cricket video frame-by-frame using MediaPipe pose.
- Annotates each frame with skeleton and live biomechanical metrics:
  - Front elbow angle (shoulder-elbow-wrist)
  - Spine lean (mid-shoulder vs vertical)
  - Head-over-front-knee vertical projection (normalized)
  - Front foot direction (heel -> foot_index angle)
- Shows short feedback cues on frame when thresholds are breached.
- Outputs:
  - `output/annotated_video.mp4`
  - `output/evaluation.json` (final 1–10 scores + comments per category)

## Requirements
- Python 3.8+
- See `requirements.txt`. Install with `pip install -r requirements.txt`.
- If analyzing a YouTube video provide the URL and install `pytube`.

## Run
Local file:
python cover_drive_analysis_realtime.py --input path/to/video.mp4

YouTube:


python cover_drive_analysis_realtime.py --input "https://youtube.com/shorts/..."

Quick test (first 500 frames):


python cover_drive_analysis_realtime.py -i input.mp4 --max-frames 500


Optional flags:
- `--show` : display annotated frames live (press `q` to quit)
- `--fps 15` : set target output FPS

## Output
- `output/annotated_video.mp4` — annotated with skeleton and metrics
- `output/evaluation.json` — JSON containing scores and comments for:
  - Footwork, Head Position, Swing Control, Balance, Follow-through

## Notes & Assumptions
- Uses MediaPipe Pose landmarks. Missing/occluded landmarks are handled gracefully.
- Heuristics are simple and rule-based (not ML). Thresholds can be tuned in code.
- Front side detection is heuristic-based (head vs knee vertical distance). May be wrong in some frames.
- No bat tracking or contact detection in base. See BONUS ideas for enhancements.
- Performance: real-time depends on hardware and video size. Lower resolution -> faster processing.

## Limitations
- Scoring is heuristic and coarse; for production replace with domain-calibrated thresholds or labeled data.
- Foot direction uses heel->foot_index — accuracy depends on visible foot landmarks.
- MediaPipe may fail for extreme occlusion / motion blur frames.

## Extending (suggestions)
- Implement phase segmentation using velocities (bonus).
- Add wrist-velocity detection for contact moment auto-detection.
- Export plots (angle vs time) as PNGs and embed in `output/`.