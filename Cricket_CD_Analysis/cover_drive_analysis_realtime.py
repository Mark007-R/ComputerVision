#!/usr/bin/env python3
"""
cover_drive_analysis_realtime.py
AthleteRise — Real-Time Cover Drive Analysis from Full Video
Base features:
 - Download (optional) input video (YouTube) or read local path
 - Frame-by-frame MediaPipe Pose detection
 - Compute elbow angle, spine lean, head-over-front-knee distance, front foot direction
 - Overlay skeleton, metrics and short feedback per frame
 - Save annotated video to /output/annotated_video.mp4 and evaluation.json
"""

import os
import cv2
import time
import json
import math
import argparse
import numpy as np
from collections import deque, defaultdict

# Try to import optional pytube for YouTube downloading
try:
    from pytube import YouTube
    PYTUBE_AVAILABLE = True
except Exception:
    PYTUBE_AVAILABLE = False

# MediaPipe imports
try:
    import mediapipe as mp
except Exception as e:
    raise ImportError("mediapipe is required. Install via pip install mediapipe") from e

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# OUTPUT paths
OUTPUT_DIR = "output"
ANNOTATED_VIDEO = os.path.join(OUTPUT_DIR, "annotated_video.mp4")
EVALUATION_FILE = os.path.join(OUTPUT_DIR, "evaluation.json")

# Helpers
def ensure_out():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def download_youtube(url, out_path="downloaded_input.mp4"):
    if not PYTUBE_AVAILABLE:
        raise RuntimeError("pytube not installed. Install 'pytube' to enable YouTube downloading.")
    yt = YouTube(url)
    # choose progressive stream (video+audio) with reasonable resolution
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not stream:
        raise RuntimeError("No suitable stream found.")
    stream.download(filename=out_path)
    return out_path

def vec(a, b):
    return np.array([b[0]-a[0], b[1]-a[1]])

def angle_between(a, b, c):
    """
    Angle at point b formed by points a-b-c (in degrees).
    Handles None by returning None.
    """
    if None in (a, b, c):
        return None
    v1 = np.array(a) - np.array(b)
    v2 = np.array(c) - np.array(b)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return None
    cosang = np.dot(v1, v2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def vertical_distance_projection(ptA, ptB):
    """Returns absolute vertical (y-axis) pixel distance between two normalized points (x,y)."""
    if ptA is None or ptB is None:
        return None
    return abs(ptA[1] - ptB[1])

def normalize_landmark(landmark, frame_w, frame_h):
    if landmark is None:
        return None
    return (landmark.x * frame_w, landmark.y * frame_h)

def landmark_safe(landmarks, idx):
    try:
        lm = landmarks[idx]
        # mediapipe includes visibility; accept if present
        return (lm.x, lm.y, lm.z, getattr(lm, 'visibility', None))
    except Exception:
        return None

# Mediapipe Pose landmark indices (use mp_pose.PoseLandmark enum for clarity)
LM = mp_pose.PoseLandmark

def extract_keypoints(landmarks, frame_w, frame_h):
    """
    Return a dict with coordinates (x,y) or None if unavailable for important keypoints
    keys: head, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
          left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_heel, right_heel, left_foot_index, right_foot_index
    """
    # landmarks could be None
    if landmarks is None:
        return {}
    def get(idx):
        try:
            lm = landmarks[idx]
            return (lm.x * frame_w, lm.y * frame_h)
        except Exception:
            return None

    keys = {}
    keys['nose'] = get(LM.NOSE.value)
    keys['left_eye'] = get(LM.LEFT_EYE.value)
    keys['right_eye'] = get(LM.RIGHT_EYE.value)
    keys['left_shoulder'] = get(LM.LEFT_SHOULDER.value)
    keys['right_shoulder'] = get(LM.RIGHT_SHOULDER.value)
    keys['left_elbow'] = get(LM.LEFT_ELBOW.value)
    keys['right_elbow'] = get(LM.RIGHT_ELBOW.value)
    keys['left_wrist'] = get(LM.LEFT_WRIST.value)
    keys['right_wrist'] = get(LM.RIGHT_WRIST.value)
    keys['left_hip'] = get(LM.LEFT_HIP.value)
    keys['right_hip'] = get(LM.RIGHT_HIP.value)
    keys['left_knee'] = get(LM.LEFT_KNEE.value)
    keys['right_knee'] = get(LM.RIGHT_KNEE.value)
    keys['left_ankle'] = get(LM.LEFT_ANKLE.value)
    keys['right_ankle'] = get(LM.RIGHT_ANKLE.value)
    keys['left_heel'] = get(LM.LEFT_HEEL.value)
    keys['right_heel'] = get(LM.RIGHT_HEEL.value)
    keys['left_foot_index'] = get(LM.LEFT_FOOT_INDEX.value)
    keys['right_foot_index'] = get(LM.RIGHT_FOOT_INDEX.value)
    return keys

# Metric computations
def compute_front_elbow_angle(kps, side='right'):
    """
    Compute shoulder-elbow-wrist angle for the 'front' side.
    side: 'left' or 'right' - typically the batting front side (we'll compute both and pick the more flexed)
    """
    s = kps.get(f"{side}_shoulder")
    e = kps.get(f"{side}_elbow")
    w = kps.get(f"{side}_wrist")
    return angle_between(s, e, w)

def compute_spine_lean(kps):
    """
    Compute spine lean: angle between vertical and shoulder-hip line.
    We'll compute the angle between vector (mid_hip -> mid_shoulder) and the vertical direction.
    Return degrees; 0 means perfectly vertical, >0 means leaning forward/backward.
    """
    ls = kps.get('left_shoulder')
    rs = kps.get('right_shoulder')
    lh = kps.get('left_hip')
    rh = kps.get('right_hip')
    if None in (ls, rs, lh, rh):
        return None
    mid_shoulder = ((ls[0]+rs[0])/2.0, (ls[1]+rs[1])/2.0)
    mid_hip = ((lh[0]+rh[0])/2.0, (lh[1]+rh[1])/2.0)
    vec_sh_hip = np.array(mid_shoulder) - np.array(mid_hip)
    # vertical vector (0, -1) in image pixel space (upwards), but angle independent of sign
    vertical = np.array([0.0, -1.0])
    denom = (np.linalg.norm(vec_sh_hip) * np.linalg.norm(vertical))
    if denom == 0:
        return None
    cosang = np.dot(vec_sh_hip, vertical) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def compute_head_over_knee_distance(kps, front_side='right', frame_h=1.0):
    """
    Project vertical distance between head (use nose) and front knee in normalized pixels.
    Returns distance in pixels (frame coordinate units). We will normalize by frame_h if provided.
    """
    nose = kps.get('nose')
    knee = kps.get(f"{front_side}_knee")
    if nose is None or knee is None:
        return None
    # vertical distance in pixels (y axis)
    return abs(nose[1] - knee[1])

def compute_front_foot_direction(kps, front_side='right'):
    """
    Approximate foot direction as angle between vector (heel -> foot_index) and video x-axis.
    Returns angle in degrees (0 = horizontal along +x).
    """
    heel = kps.get(f"{front_side}_heel")
    fi = kps.get(f"{front_side}_foot_index")
    if heel is None or fi is None:
        return None
    v = np.array(fi) - np.array(heel)
    if np.linalg.norm(v) == 0:
        return None
    # angle with x-axis
    ang = math.degrees(math.atan2(v[1], v[0]))
    return ang

# Scoring heuristics (simple rule-based)
def score_metric(metric_name, value):
    """
    Map a metric to 1-10 based on heuristics. Missing values -> neutral 5.
    metric_name: 'elbow', 'spine', 'head_over_knee', 'foot_dir', 'balance_proxy'
    """
    if value is None:
        return 5  # neutral
    if metric_name == 'elbow':
        # ideal front elbow angle might be ~90-120 degrees depending on technique
        # map 60 -> 1, 90 -> 8, 120 -> 10, >140 -> 3 (arm too straight)
        if value < 60:
            return 2
        if value < 80:
            return 5
        if value < 100:
            return 8
        if value <= 125:
            return 10
        if value <= 140:
            return 6
        return 3
    if metric_name == 'spine':
        # smaller angle (closer to vertical) generally good; but slight lean forward okay.
        # angle 0 (vertical) -> 9, 5->8, 10->6, 20->3, >30->1
        if value < 4:
            return 9
        if value < 8:
            return 8
        if value < 12:
            return 6
        if value < 20:
            return 4
        return 2
    if metric_name == 'head_over_knee':
        # smaller vertical distance (i.e., head over knee) better. We'll interpret in pixels normalized later.
        # We'll expect distances normalized by frame height. We'll treat value as normalized fraction.
        d = value
        if d < 0.02:
            return 10
        if d < 0.05:
            return 8
        if d < 0.08:
            return 6
        if d < 0.12:
            return 4
        return 2
    if metric_name == 'foot_dir':
        # Ideally foot roughly points toward crease (parallel to x-axis) -> angle near 0 or 180
        # We'll take absolute of angle normalized to [-90,90]
        ang = abs(((value + 180) % 180) - 90)  # map to 0..90 with 0 meaning pointing perpendicular - but we want near 0 for pointing along x?
        # Simpler: prefer small absolute angle relative to x-axis (|ang| small)
        ang = abs(value)
        if ang < 10:
            return 9
        if ang < 25:
            return 7
        if ang < 45:
            return 5
        if ang < 70:
            return 3
        return 2
    if metric_name == 'balance_proxy':
        # balance proxy could be distance between mid-hip x and mid-ankles x normalized
        v = value
        if v is None:
            return 5
        if v < 0.05:
            return 9
        if v < 0.1:
            return 7
        if v < 0.2:
            return 5
        return 3
    return 5

def aggregate_scores(score_dict):
    """
    score_dict contains arrays of scores per frame for each metric.
    We'll average across frames and map to 1-10 integers.
    Also produce small feedback lines per category.
    """
    out = {}
    # categories: Footwork, Head Position, Swing Control, Balance, Follow-through
    # Basic mapping:
    # Footwork -> foot_dir
    # Head Position -> head_over_knee
    # Swing Control -> elbow
    # Balance -> balance_proxy
    # Follow-through -> spine / distribution (use spine)
    # For each, average score
    def avg(arr):
        if not arr:
            return 5.0
        return float(sum(arr) / len(arr))
    foot = avg(score_dict.get('foot_dir', []))
    head = avg(score_dict.get('head_over_knee', []))
    swing = avg(score_dict.get('elbow', []))
    balance = avg(score_dict.get('balance_proxy', []))
    follow = avg(score_dict.get('spine', []))

    def make_comment(cat, val):
        if cat == 'Footwork':
            if val >= 8: return "Front foot direction is strong and oriented towards the crease."
            if val >= 6: return "Foot direction generally good; a bit more pointing through the line will help."
            return "Work on pointing the front foot more towards the target; improves transfer and timing."
        if cat == 'Head Position':
            if val >= 8: return "Head stays well over front knee — good balance and vision."
            if val >= 6: return "Head usually over knee but occasionally drifts — tighten balance during stride."
            return "Keep head over front knee (lower center of gravity) to improve control."
        if cat == 'Swing Control':
            if val >= 8: return "Elbow angles are in the desirable range — controlled swing."
            if val >= 6: return "Swing control acceptable; small adjustments in elbow elevation recommended."
            return "Work on maintaining front elbow elevation for stroke control (drills: shadow front-elbow holds)."
        if cat == 'Balance':
            if val >= 8: return "Good balance through the shot."
            if val >= 6: return "Moderate balance; improve stable base and weight transfer."
            return "Focus on a stable base and lower-center-of-gravity drills to improve balance."
        if cat == 'Follow-through':
            if val >= 8: return "Follow-through and spine alignment are effective."
            if val >= 6: return "Follow-through is decent; ensure full extension after contact."
            return "Extend follow-through and avoid early upright posture; this helps power & control."
        return ""
    aggregated = {
        "Footwork": {"score": int(round(foot)), "comment": make_comment('Footwork', foot)},
        "Head Position": {"score": int(round(head)), "comment": make_comment('Head Position', head)},
        "Swing Control": {"score": int(round(swing)), "comment": make_comment('Swing Control', swing)},
        "Balance": {"score": int(round(balance)), "comment": make_comment('Balance', balance)},
        "Follow-through": {"score": int(round(follow)), "comment": make_comment('Follow-through', follow)},
    }
    return aggregated

# Main processing function
def analyze_video(input_path, output_path=ANNOTATED_VIDEO, target_fps=None, show=False, max_frames=None):
    ensure_out()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = target_fps or src_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (src_w, src_h))

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_idx = 0
    start_time = time.time()
    frame_times = []
    # rolling metrics store
    scores_per_frame = defaultdict(list)
    elbow_history = deque(maxlen=30)
    spine_history = deque(maxlen=30)

    # optional: pick front side by checking which knee moves forward (heuristic). We'll compute both and choose per-frame.
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        t0 = time.time()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        # draw skeleton
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0,200,200), thickness=1, circle_radius=1))
        # extract kps in pixel space
        kps = extract_keypoints(results.pose_landmarks.landmark if results.pose_landmarks else None, src_w, src_h)

        # compute side metrics for both left and right
        elbow_r = compute_front_elbow_angle(kps, 'right')
        elbow_l = compute_front_elbow_angle(kps, 'left')
        # choose front side as side with smaller head-over-knee distance (heuristic: front knee closer to head)
        head_r_dist = compute_head_over_knee_distance(kps, 'right', src_h)
        head_l_dist = compute_head_over_knee_distance(kps, 'left', src_h)
        # pick side that has smaller vertical distance -> considered front
        front_side = 'right'
        if head_l_dist is not None and head_r_dist is not None:
            front_side = 'left' if head_l_dist < head_r_dist else 'right'
        elif head_l_dist is not None:
            front_side = 'left'
        elif head_r_dist is not None:
            front_side = 'right'

        elbow = compute_front_elbow_angle(kps, front_side) or (elbow_r or elbow_l)
        spine = compute_spine_lean(kps)
        head_over_knee_px = compute_head_over_knee_distance(kps, front_side, src_h)
        head_over_knee_norm = (head_over_knee_px / src_h) if head_over_knee_px is not None else None
        foot_dir = compute_front_foot_direction(kps, front_side)
        # simple balance proxy: normalized horizontal offset between hips midpoint and ankles midpoint
        try:
            if kps.get('left_hip') and kps.get('right_hip') and kps.get('left_ankle') and kps.get('right_ankle'):
                mid_hip_x = (kps['left_hip'][0] + kps['right_hip'][0]) / 2.0
                mid_ankle_x = (kps['left_ankle'][0] + kps['right_ankle'][0]) / 2.0
                balance_proxy = abs(mid_hip_x - mid_ankle_x) / src_w
            else:
                balance_proxy = None
        except Exception:
            balance_proxy = None

        # compute per-frame scores
        s_elbow = score_metric('elbow', elbow)
        s_spine = score_metric('spine', spine)
        s_head = score_metric('head_over_knee', head_over_knee_norm)
        s_foot = score_metric('foot_dir', foot_dir)
        s_balance = score_metric('balance_proxy', balance_proxy)

        # store for aggregation
        scores_per_frame['elbow'].append(s_elbow)
        scores_per_frame['spine'].append(s_spine)
        scores_per_frame['head_over_knee'].append(s_head)
        scores_per_frame['foot_dir'].append(s_foot)
        scores_per_frame['balance_proxy'].append(s_balance)

        # overlays: metrics text
        y0 = 30
        def put(txt, offset=0, color=(255,255,255)):
            cv2.putText(frame, txt, (10, y0 + offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        put(f"Frame: {frame_idx}/{total_frames}")
        put(f"Front side (heuristic): {front_side}", 25)
        put(f"Elbow: {f'{elbow:.1f}°' if elbow else 'N/A'}  Score:{s_elbow}", 55)
        put(f"Spine lean: {f'{spine:.1f}°' if spine else 'N/A'}  Score:{s_spine}", 85)
        put(f"Head-over-knee (norm): {f'{head_over_knee_norm:.3f}' if head_over_knee_norm else 'N/A'}  Score:{s_head}", 115)
        put(f"Front foot ang: {f'{foot_dir:.1f}°' if foot_dir else 'N/A'}  Score:{s_foot}", 145)
        put(f"Balance proxy: {f'{balance_proxy:.3f}' if balance_proxy else 'N/A'}  Score:{s_balance}", 175)

        # short feedback cues
        feedback = []
        if elbow is not None:
            if elbow >= 90 and elbow <= 125:
                feedback.append(("✅ Good elbow elevation", (0,200,0)))
            elif elbow < 80:
                feedback.append(("❌ Elbow too close (lower), lift front elbow", (0,0,255)))
            else:
                feedback.append(("⚠️ Elbow outside ideal range", (0,140,255)))
        if head_over_knee_norm is not None:
            if head_over_knee_norm < 0.05:
                feedback.append(("✅ Head over front knee", (0,200,0)))
            else:
                feedback.append(("❌ Head not over front knee", (0,0,255)))
        if spine is not None:
            if spine < 12:
                feedback.append(("✅ Good spine lean", (0,200,0)))
            else:
                feedback.append(("❌ Excessive spine lean", (0,0,255)))

        # render 2 top feedback cues
        for i, (txt, col) in enumerate(feedback[:2]):
            cv2.putText(frame, txt, (src_w - 520, 40 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

        # write frame
        out.write(frame)
        if show:
            cv2.imshow("Annotated", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        t1 = time.time()
        frame_times.append(t1 - t0)

    # cleanup
    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    avg_fps = (frame_idx / elapsed) if elapsed > 0 else 0.0

    print(f"Processed {frame_idx} frames in {elapsed:.1f}s — avg FPS: {avg_fps:.2f}")

    # aggregate to evaluation
    aggregated = aggregate_scores(scores_per_frame)
    summary = {
        "video_processed": os.path.abspath(input_path),
        "frames_processed": frame_idx,
        "avg_fps": avg_fps,
        "scores": aggregated,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    # save eval
    with open(EVALUATION_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Annotated video saved to {os.path.abspath(output_path)}")
    print(f"Evaluation saved to {os.path.abspath(EVALUATION_FILE)}")
    return summary

# CLI
def main():
    parser = argparse.ArgumentParser(description="Real-time cover drive analysis")
    parser.add_argument("--input", "-i", required=True, help="Path to local video or YouTube URL")
    parser.add_argument("--output", "-o", default=ANNOTATED_VIDEO, help="Output annotated video path")
    parser.add_argument("--fps", type=float, default=None, help="Target output FPS (keeps resolution)")
    parser.add_argument("--show", action="store_true", help="Show annotated frames live (press q to quit)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional: limit number of frames for quick tests")
    args = parser.parse_args()

    src = args.input
    # If looks like a youtube url, try to download
    local_path = src
    if src.startswith("http://") or src.startswith("https://"):
        if not PYTUBE_AVAILABLE:
            print("You passed a URL but pytube is not installed. Please install pytube or provide a local file.")
            return
        print("Downloading YouTube video (this may take a moment)...")
        local_path = "downloaded_input.mp4"
        try:
            download_youtube(src, out_path=local_path)
        except Exception as e:
            print("Failed to download video:", e)
            return

    print("Starting analysis...")
    analyze_video(local_path, output_path=args.output, target_fps=args.fps, show=args.show, max_frames=args.max_frames)

if __name__ == "__main__":
    main()
