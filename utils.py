import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, frame_rate=1):
    """Extract frames from video at specified frame rate"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps // frame_rate
    
    frames = []
    count = 0
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_num:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame)
            frame_num += 1
        count += 1
    
    cap.release()
    return frames

def create_video_from_frames(frames, output_path, fps=30):
    """Create video from processed frames"""
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()

def ensure_dirs():
    """Create necessary directories"""
    dirs = ['models', 'data/train', 'data/val', 'uploads', 'outputs']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
