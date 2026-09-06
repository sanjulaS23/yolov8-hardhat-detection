import cv2
import time
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = 'runs/train/hardhat_v14/weights/best.pt'
IGNORE_CLASSES = {'person'}

model = YOLO(MODEL_PATH)


def run_video(video_path, conf=0.35, save_output=True):
    """
    Run detection on a .mp4 file.
    Displays annotated video on screen.
    Saves annotated output to output_videos/ folder.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return

    # Get video properties
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps | {total} frames")

    # Setup output video writer
    writer = None
    out_path = None
    if save_output:
        Path('output_videos').mkdir(exist_ok=True)
        stem     = Path(video_path).stem
        out_path = f'output_videos/{stem}_annotated.mp4'
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Saving annotated output -> {out_path}")

    frame_num = 0
    fps_buf   = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break   # end of video

        frame_num += 1

        # Run detection
        t0      = time.perf_counter()
        results = model(frame, conf=conf, verbose=False)
        fps_buf.append(1 / (time.perf_counter() - t0))
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        avg_fps = sum(fps_buf) / len(fps_buf)

        # Draw bounding boxes (all classes, including person — visual only)
        annotated = results[0].plot()

        # Overlay: FPS + frame counter
        progress = f"Frame {frame_num}/{total}"
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}  {progress}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

        # Safety logic: 'head' (no helmet) detected -> violation
        # 'person' is ignored entirely (unreliable class, see evaluation results)
        violation = any(
            model.names[int(b.cls)] == 'head'
            for b in results[0].boxes
            if model.names[int(b.cls)] not in IGNORE_CLASSES
        )
        if violation:
            cv2.putText(annotated, "SAFETY VIOLATION - NO HELMET",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 0, 255), 3)

        # Save frame to output video
        if writer:
            writer.write(annotated)

    cap.release()
    if writer:
        writer.release()
    print(f"Done - processed {frame_num} frames")
    if save_output:
        print(f"Annotated video saved -> {out_path}")


if __name__ == '__main__':
    run_video('videos/site_footage.mp4')