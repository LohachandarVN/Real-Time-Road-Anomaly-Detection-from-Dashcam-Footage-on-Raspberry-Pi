# Real-Time-Road-Anomaly-Detection-from-Dashcam-Footage-on-Raspberry-Pi
An Edge AI application designed for the Raspberry Pi that processes dashcam footage in real-time to detect, log, and record road anomalies such as potholes and unexpected obstacles.  This project demonstrates the deployment of lightweight computer vision models on edge hardware, utilizing model optimization techniques that are effiecient.



##  Project Overview

**Objective:** To build an edge AI application capable of processing dashcam footage in real-time to detect and log road anomalies (potholes, obstacles, vehicle accidents, etc.) entirely on an ARM-based edge device.

**Alignment:** This project successfully implements a fully CPU-bound inference pipeline on a Raspberry Pi. By eliminating the reliance on cloud computing or external hardware accelerators (like Google Coral), it ensures that anomaly logging remains functional in remote areas with poor network connectivity, which is critical for real-world dashcam applications.

---

## Architecture & Pipeline

The system architecture follows a highly optimized, linear four-step pipeline:



1. **Video Input:** A USB webcam (simulating a dashcam) captures a live stream at 1080p, or the system reads a pre-recorded `.mp4` video file containing road anomalies.
2. **Pre-processing:** To ensure real-time performance on a resource-constrained CPU, OpenCV extracts frames and immediately resizes the input tensor to 320x320 pixels. Color spaces are normalized for neural network ingestion.
3. **AI Inference:** The pre-processed frame is passed to the `tflite-runtime` engine. A YOLOv11n (Nano) model, heavily quantized to 8-bit integers (INT8), executes the object detection math on the Raspberry Pi's ARM Cortex-A72 cores.
4. **Action & Logging:** Detections are filtered by a strict confidence threshold (e.g., `> 0.35`) to minimize false positives. Valid anomalies trigger the Supervision library to draw bounding boxes and labels on the frame, calculate the real-time FPS, and display the output.

---

## ⚙️ Hardware & Software Environment

### Hardware Requirements
* **Compute:** Raspberry Pi 4 Model B (4GB/8GB RAM).
* **Camera:** 1080p USB Webcam / Raspberry Pi Camera Module v2.
* **Storage:** High-write-speed MicroSD card (Class 10) to prevent I/O bottlenecks.
* **Thermal Management:** Active heat sinks and a cooling fan are mandatory to prevent CPU thermal throttling (>80°C), which degrades frame rates.

### Software Stack
* **OS:** Raspberry Pi OS 64-bit (Required for optimal memory access).
* **Language:** Python 3.9+ running in an isolated virtual environment (`.venv`).
* **Libraries:** `tflite-runtime` (for lightweight inference), `opencv-python` (video handling), `supervision` (annotations), `streamlit` (optional web UI).

---

##  Model Selection & Optimization

* **Base Model:** YOLOv11n (Nano). Standard YOLOv8/11 models are too heavy for Raspberry Pi CPUs. The Nano variant provides the best trade-off between architectural efficiency and the precision needed for complex road environments.
* **Training Data:** The model is trained to recognize potholes, accidents, and debris, utilizing diverse datasets such as the Global Road Damage Detection Challenge (GRDDC) and the India Driving Dataset (IDD) to handle challenging local infrastructure.
* **Quantization & Export:** Standard `.pt` PyTorch weights require heavy 32-bit floating-point (FP32) math. The model was exported to TensorFlow Lite (`.tflite`) using Full Integer Quantization (INT8). This reduced the model size by ~75% and shifted operations to integer arithmetic, unlocking near-real-time speeds on ARM CPUs.

---

## Performance Targets (ARM Cortex Focus)

Achieving the required **≥ 5 FPS** on a Pi 4 CPU required aggressive optimization:

* **Resolution Scaling:** We deliberately scale the OpenCV camera read parameters to 640x480, and further compress the inference tensor to `imgsz=320`.
* **Minimizing False Positives:** The `CONF_THRESHOLD` is tightly regulated (0.35 - 0.40). This ensures the system only logs high-probability anomalies, maintaining the system's credibility and preventing unnecessary read/write cycles to the SD card.
* **Throughput:** By handling video buffering sequentially and avoiding multi-threading overheads in the core `run.py` loop, we prevent frame-dropping and memory leaks.

---

## Novelty & Customization

* **Modular Pipeline Design:** The codebase features interchangeable front-ends. Users can run bare-metal via `run.py` for deployment, or boot up a fully interactive web application via `main.py` using Streamlit WebRTC for remote monitoring and demonstration.
* **Dynamic Resolution Handling:** While the neural network inference is aggressively downscaled to 320x320 to preserve CPU cycles, the Supervision bounding box logic scales the annotations back up to the original frame resolution, ensuring the saved output footage remains clean and high-fidelity.

---

## Media & Outputs



* **Live Screenshots:** 
* **Demo Video**

**Outputs Include:**
1. Real-time visual display with bounding boxes identifying the anomaly class (e.g., "Pothole", "Obstacle") and confidence scores.
2. An overlay of the real-time processing speed (FPS).
3. Annotated media saved directly to the `inference_output/` directory.

---

##  Assumptions & Limitations

**Assumptions:**
* The hardware is securely mounted on the dashboard; extreme camera shake is mitigated by physical dampening rather than computationally expensive software stabilization.
* The primary goal is offline anomaly logging for post-trip analytics, not instantaneous collision avoidance (which would require millisecond latency).

**Limitations:**
* **Lighting:** Extreme low-light conditions without street illumination will degrade bounding box accuracy due to motion blur.
* **Distance Processing:** Because the inference tensor is downsampled to 320x320 to maintain 5 FPS, smaller potholes at a far distance (>15 meters) may not trigger an immediate detection until the vehicle closes the gap.

---

## Conclusion & Future Scope

### Conclusion
This project successfully demonstrates the viability of deploying advanced machine learning models directly onto resource-constrained edge devices for real-time infrastructure monitoring. By heavily optimizing the pipeline—transitioning from heavy PyTorch tensors to an INT8 quantized TensorFlow Lite model and aggressively managing input resolution—the system achieves a stable ≥ 5 FPS entirely on the Raspberry Pi's ARM Cortex-A72 CPU. 

Ultimately, this localized, cloud-independent approach resolves critical challenges in modern dashcam analytics: it eliminates expensive cloud data transmission costs, protects user privacy by processing footage locally, and ensures zero-latency anomaly logging even in remote areas with poor or non-existent cellular coverage.

### Future Enhancements
While the current prototype meets all hackathon deliverables, the architecture is designed to be highly extensible:
* **Hardware Sensor Fusion:** Integrating the visual data with physical IMU (accelerometer/gyroscope) sensors to cross-verify physical vehicle jolts with visual pothole detections.
* **Physical GPS Integration:** Connecting a dedicated NEO-6M GPS module to the Raspberry Pi's GPIO pins to automatically generate heat maps of degraded infrastructure for municipal repair teams.
* **Night-Vision Capability:** Swapping the standard USB webcam for a NoIR (No Infrared Filter) Raspberry Pi camera module, coupled with IR illuminators.
