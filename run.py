from ultralytics import YOLO
import cv2
import supervision as sv
import time
from pathlib import Path

BEST_WEIGHTS_PATH_STR = "yolo11n_int8.tflite" 
CONF_THRESHOLD = 0.35
MODE = "live" 
INFERENCE_RESOLUTION = 320 

INPUT_IMAGE_PATH_STR = "path/to/your/test/image.jpg"
INPUT_VIDEO_PATH_STR = "path/to/your/test/video.mp4"
OUTPUT_DIR_STR = "inference_output"
CAMERA_INDEX = 0

BEST_WEIGHTS_PATH = Path(BEST_WEIGHTS_PATH_STR)
OUTPUT_DIR = Path(OUTPUT_DIR_STR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    model = YOLO(str(BEST_WEIGHTS_PATH), task='detect')
    class_names = model.names
except Exception as e:
    exit()

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(
    text_thickness=1, text_scale=0.6, text_color=sv.Color.BLACK, text_padding=2
)

def process_frame(frame: cv2.typing.MatLike, frame_index: int) -> cv2.typing.MatLike:
    results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=INFERENCE_RESOLUTION, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    labels = [
        f"{class_names.get(class_id, str(class_id))} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame

def infer_on_live_camera(camera_index: int):
    cap = cv2.VideoCapture(camera_index)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        return

    prev_time = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = process_frame(frame, frame_count)
            frame_count += 1

            current_time = time.time()
            if (current_time - prev_time) > 0:
                fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            cv2.putText(annotated_frame, f"Pi 4 FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Live Edge Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if MODE == "live":
        infer_on_live_camera(CAMERA_INDEX)
    elif MODE == "image":
        pass
    elif MODE == "video":
        pass