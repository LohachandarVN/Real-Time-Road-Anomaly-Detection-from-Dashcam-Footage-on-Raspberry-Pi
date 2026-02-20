from ultralytics import YOLO
import cv2
import supervision as sv
import time
from pathlib import Path
import traceback  

MODEL_1_WEIGHTS_PATH_STR = "yolo11n_int8.tflite"
MODEL_2_WEIGHTS_PATH_STR = "yolov8n_int8.tflite"

CONF_THRESHOLD_MODEL_1 = 0.35
CONF_THRESHOLD_MODEL_2 = 0.40

MODE = "video" 

INPUT_IMAGE_PATH_STR = "path/to/your/test/image.jpg"
INPUT_VIDEO_PATH_STR = "Downloads/v.mp4"
OUTPUT_DIR_STR = "inference_output_two_models"
CAMERA_INDEX = 0

MODEL_1_WEIGHTS_PATH = Path(MODEL_1_WEIGHTS_PATH_STR)
MODEL_2_WEIGHTS_PATH = Path(MODEL_2_WEIGHTS_PATH_STR)
OUTPUT_DIR = Path(OUTPUT_DIR_STR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model1 = None
model2 = None
class_names1 = {}
class_names2 = {}
models_loaded = False

try:
    if MODEL_1_WEIGHTS_PATH.is_file():
        model1 = YOLO(str(MODEL_1_WEIGHTS_PATH), task='detect')
        class_names1 = model1.names

    if MODEL_2_WEIGHTS_PATH.is_file():
        model2 = YOLO(str(MODEL_2_WEIGHTS_PATH), task='detect')
        class_names2 = model2.names

    if model1 and model2:
        models_loaded = True

except Exception as e:
    traceback.print_exc()

if not models_loaded:
    exit()

box_annotator1 = sv.BoxAnnotator(thickness=2, color=sv.Color.RED)
label_annotator1 = sv.LabelAnnotator(
    text_thickness=1, text_scale=0.5, color=sv.Color.WHITE, text_color=sv.Color.BLACK, text_padding=2, text_position=sv.Position.TOP_LEFT,
)
box_annotator2 = sv.BoxAnnotator(thickness=2, color=sv.Color.BLUE)
label_annotator2 = sv.LabelAnnotator(
    text_thickness=1, text_scale=0.5, color=sv.Color.WHITE, text_color=sv.Color.BLACK, text_padding=2, text_position=sv.Position.TOP_RIGHT,
)

def process_frame_two_models(frame: cv2.typing.MatLike, frame_index: int) -> cv2.typing.MatLike:
    annotated_frame = frame.copy()

    try:
        results1 = model1.predict(frame, conf=CONF_THRESHOLD_MODEL_1, imgsz=320, verbose=False)[0]
        detections1 = sv.Detections.from_ultralytics(results1)
        labels1 = [
            f"M1:{class_names1.get(cls_id, f'cls_{cls_id}')} {conf:.2f}"
            for cls_id, conf in zip(detections1.class_id, detections1.confidence)
        ]
        annotated_frame = box_annotator1.annotate(scene=annotated_frame, detections=detections1)
        annotated_frame = label_annotator1.annotate(scene=annotated_frame, detections=detections1, labels=labels1)
    except Exception as e:
        pass

    try:
        results2 = model2.predict(frame, conf=CONF_THRESHOLD_MODEL_2, imgsz=320, verbose=False)[0]
        detections2 = sv.Detections.from_ultralytics(results2)
        labels2 = [
            f"M2:{class_names2.get(cls_id, f'cls_{cls_id}')} {conf:.2f}"
            for cls_id, conf in zip(detections2.class_id, detections2.confidence)
        ]
        annotated_frame = box_annotator2.annotate(scene=annotated_frame, detections=detections2)
        annotated_frame = label_annotator2.annotate(scene=annotated_frame, detections=detections2, labels=labels2)
    except Exception as e:
        pass

    return annotated_frame

def infer_on_live_camera(camera_index: int):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return

    prev_time = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            annotated_frame = process_frame_two_models(frame, frame_count)
            frame_count += 1

            current_time = time.time()
            fps = (1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0)
            prev_time = current_time
            
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Live Camera - Two Models", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if not models_loaded:
        exit()

    if MODE == "image":
        pass
    elif MODE == "video":
        pass
    elif MODE == "live":
        infer_on_live_camera(CAMERA_INDEX)