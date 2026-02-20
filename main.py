import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import logging

MODEL_PATHS = {
    "M1 (Edge Nano)": "yolo11n_int8.tflite",
}
MODEL_PREFIX = {
    "M1 (Edge Nano)": "M1",
}
DEFAULT_CONF = {"M1 (Edge Nano)": 0.40}

LIVE_FEED_TARGET_WIDTH = 320 

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "output_file_path" not in st.session_state:
    st.session_state.output_file_path = None
if "output_file_name" not in st.session_state:
    st.session_state.output_file_name = None

@st.cache_resource
def load_yolo_model(path: str):
    try:
        model = YOLO(path, task='detect')
        return model, model.names
    except Exception as e:
        st.error(f"Error loading Edge model at {path}: {e}")
        return None, {}

def make_annotators(color: sv.Color):
    box_annotator = sv.BoxAnnotator(thickness=1, color=color)
    label_annotator = sv.LabelAnnotator(
        text_thickness=1, text_scale=0.4, color=sv.Color.WHITE, text_color=sv.Color.BLACK, text_padding=2,
    )
    return box_annotator, label_annotator

def process_frame(frame: np.ndarray, models: dict[str, tuple], thresholds: dict[str, float]) -> np.ndarray:
    annotated_frame = frame.copy()
    for model_name, (model, names_map, box_ann, label_ann) in models.items():
        try:
            results = model.predict(frame, conf=thresholds[model_name], imgsz=320, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            labels = [
                f"{MODEL_PREFIX[model_name]}:{names_map.get(cls_id, str(cls_id))} {conf:.2f}"
                for cls_id, conf in zip(detections.class_id, detections.confidence)
            ]
            annotated_frame = box_ann.annotate(annotated_frame, detections)
            annotated_frame = label_ann.annotate(annotated_frame, detections, labels=labels)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
    return annotated_frame

def main():
    st.set_page_config(layout="wide", page_title="Edge Road Anomaly Detection")
    st.title("Raspberry Pi Dashcam Detection")
    st.markdown("Optimized for Arm Cortex CPU via INT8 TFLite")

    st.sidebar.header("Configuration")
    st.sidebar.subheader("Models")
    
    use_m1 = st.sidebar.checkbox("M1 (Edge Nano TFLite)", value=True, key="cb_m1")

    models_to_load = {}
    if use_m1:
        models_to_load["M1 (Edge Nano)"] = MODEL_PATHS["M1 (Edge Nano)"]

    loaded_models = {}
    thresholds = {}
    model_load_failed = False

    if not models_to_load:
        st.sidebar.warning("Select the edge model to start.")
        st.stop()

    for name, path in models_to_load.items():
        model, names_map = load_yolo_model(path)
        if model and names_map:
            color = sv.Color.RED
            box_ann, label_ann = make_annotators(color)
            loaded_models[name] = (model, names_map, box_ann, label_ann)
            thresholds[name] = st.sidebar.slider(f"{name} Confidence", 0.1, 1.0, DEFAULT_CONF[name], 0.05, key=f"{name}_conf")
        else:
            model_load_failed = True

    if model_load_failed:
        st.error("Model failed to load.")
        st.stop()

    st.sidebar.subheader("Input Source")
    input_mode = st.sidebar.radio("Select Input Type", ["Image", "Video", "Live Camera"], key="input_mode_radio")

    if input_mode == "Image":
        image_placeholder = st.empty()
    elif input_mode == "Video":
        video_status_placeholder = st.empty()
    elif input_mode == "Live Camera":
        pass

if __name__ == "__main__":
    main()