import cv2
import os
import json
import imagehash
from PIL import Image
import numpy as np
import subprocess
import time

# ---------------- CONFIG ----------------
VIDEO_PATH = "video_sample_1.mp4"
OUTPUT_VIDEO = "compressed_output.mp4"
FRAMES_DIR = "frames"
JSON_OUTPUT = "segments_kept.json"
HTML_REPORT = "compression_report.html"

FRAME_SKIP = 6                      # balanced (not aggressive)
PROCESS_W, PROCESS_H = 96, 72       # good speed-quality balance
OUTPUT_W, OUTPUT_H = 640, 360

# ---------------------------------------

# Clean frames folder
if os.path.exists(FRAMES_DIR):
    for f in os.listdir(FRAMES_DIR):
        os.remove(os.path.join(FRAMES_DIR, f))
else:
    os.makedirs(FRAMES_DIR)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- UTILS ----------------
def compute_phash(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(img)

def compute_motion(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 2, 7, 2, 5, 1.1, 0   # optimized params
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)

def generate_report():
    orig = os.path.getsize(VIDEO_PATH) / (1024 * 1024)
    comp = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
    reduction = (1 - comp / orig) * 100

    html = f"""
    <html>
    <body>
        <h1>Compression Report</h1>
        <p>Original Size: {orig:.2f} MB</p>
        <p>Compressed Size: {comp:.2f} MB</p>
        <p>Reduction: {reduction:.2f}%</p>
        <h2>Storyboard</h2>
    """

    files = sorted(os.listdir(FRAMES_DIR))[:10]
    for f in files:
        html += f'<img src="frames/{f}" width="200">'

    html += "</body></html>"

    with open(HTML_REPORT, "w") as f:
        f.write(html)

# ---------------- MAIN ----------------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Video not found")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    prev_gray = None
    last_hash = None
    last_kept_time = -10

    frame_index = 0
    kept_frames = 0
    segments = []

    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_index / fps

        # Frame skipping (balanced)
        if frame_index % FRAME_SKIP != 0:
            frame_index += 1
            continue

        frame_small = cv2.resize(frame, (PROCESS_W, PROCESS_H))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        keep = False

        # -------- Step 1: pHash --------
        curr_hash = compute_phash(frame_small)

        if last_hash is not None:
            if abs(curr_hash - last_hash) <= 3:
                frame_index += 1
                prev_gray = gray
                continue

        # -------- Step 2: Optical Flow --------
        motion_score = 0
        if prev_gray is not None:
            motion_score = compute_motion(prev_gray, gray)
            if motion_score > 0.05:
                keep = True

        # -------- Step 3: Face Detection --------
        # run only occasionally to save time
        if not keep and frame_index % 12 == 0:
            faces = face_cascade.detectMultiScale(gray, 1.2, 4)
            if len(faces) > 0:
                keep = True

        # -------- Step 4: Context --------
        if timestamp - last_kept_time >= 3:
            keep = True

        # -------- Save --------
        if keep:
            filename = f"{FRAMES_DIR}/{kept_frames:05d}.jpg"

            out_frame = cv2.resize(frame, (OUTPUT_W, OUTPUT_H))
            cv2.imwrite(filename, out_frame)

            segments.append({
                "frame_index": int(frame_index),
                "timestamp": float(timestamp)
            })

            last_hash = curr_hash
            last_kept_time = timestamp
            kept_frames += 1

        prev_gray = gray
        frame_index += 1

    cap.release()

    # Save JSON
    with open(JSON_OUTPUT, "w") as f:
        json.dump(segments, f, indent=4)

    # -------- FFmpeg --------
    subprocess.run([
        "ffmpeg", "-y",
        "-r", "12",
        "-i", f"{FRAMES_DIR}/%05d.jpg",
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        OUTPUT_VIDEO
    ])

    generate_report()

    end = time.time()

    print("\n✅ DONE!")
    print(f"Processing time: {end - start:.2f} sec")
    print(f"Frames kept: {kept_frames}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()