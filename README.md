# Aerial Target Detection & Tracking System

A multi-channel ground control station (GCS) application for tracking objects across RGB and IR (Infrared) video streams using YOLO models and robust custom tracking algorithms.

## Features
- Multi-Channel Video Support (up to 4 independent concurrent channels)
- RGB & IR Optimized Trackers
- Dynamic Target Highlight (Motion/Feature-based)
- Military/Tactical Dark-Themed User Interface
- Real-time Telemetry and Global Motion Compensation (Optical Flow)

## Architecture

- **`core/`**: Contains core tracking algorithms (`tracker_rgb`, `tracker_ir`) and inference utilities.
- **`scripts/`**: Standalone evaluation pipelines and test scripts.
- **`models/`**: Pre-trained YOLO weights for detections.
- **`main.py`**: The main entry point to start the Ground Control Station UI.

## Requirements
- Python 3.8+
- PyTorch (CUDA supported for best performance)
- PyQt6

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/CankayaUniversity/ceng-407-408-2025-2026-Aerial-Tracking-Detection-System.git
   cd ceng-407-408-2025-2026-Aerial-Tracking-Detection-System
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python main.py
```
