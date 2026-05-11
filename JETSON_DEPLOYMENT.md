# Jetson Nano Edge Node - Deployment Guide

This document outlines how to set up and deploy the Aerial Tracking Detection System **Edge Node** on a Jetson Nano using Docker.

---

## Phase 1: Preparation on Main PC (Windows/Mac)

Since the Jetson Nano environment does not have the `ultralytics` package installed, you must first convert your PyTorch (`.pt`) models to ONNX format on your Main PC.

1. Run the transform script on your main machine:
```bash
python models/transform.py
```
*(This will generate `.onnx` files in your `models/` directory using Jetson-safe settings like `opset=12` and `end2end=False`)*

2. Transfer the required files to the Jetson Nano:
```bash
# Transfer the edge code
scp -r jetson_edge/ jetson@<JETSON_IP_ADDRESS>:/home/jetson/

# Transfer the generated ONNX models
scp -r models/*.onnx jetson@<JETSON_IP_ADDRESS>:/home/jetson/jetson_edge/models/

# Transfer a test video (e.g., 5.mp4)
scp 5.mp4 jetson@<JETSON_IP_ADDRESS>:/home/jetson/jetson_edge/
```

---

## Phase 2: Docker Environment Setup (One-time only)

To ensure a stable environment without dependency conflicts, we use a custom Docker container built on top of the NVIDIA L4T PyTorch base image. 

If you haven't created the `jetson_edge_image` yet, follow these steps:

1. **Start the base NVIDIA container:**
```bash
sudo docker run -it --network host --runtime nvidia nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3 bash
```

2. **Install OpenCV inside the container:**
```bash
apt-get update
apt-get install -y python3-opencv python3-scipy python3-numpy
exit
```

3. **Save the container as a new image:**
Find the container ID and commit it to create your persistent image:
```bash
sudo docker ps -a  # Find the ID of the container you just exited
sudo docker commit <CONTAINER_ID> jetson_edge_image
```

---

## Phase 3: Starting the Container

Whenever you want to run the system, start your custom container using the following command:

```bash
sudo docker run -it --ipc=host --network host --runtime nvidia \
  -v /home/jetson:/home/jetson \
  jetson_edge_image bash
```

> **Parameter Breakdown:**
> - `--runtime nvidia`: Grants the container access to the Jetson's GPU and TensorRT.
> - `-v /home/jetson:/home/jetson`: Syncs the file system so you can access the files you transferred via SCP.
> - `--network host` & `--ipc=host`: Shares the host network stack, allowing the container to automatically discover the Main Hub via UDP broadcasts.

---

## Phase 4: TensorRT Engine Generation (Inside Jetson Container)

To get maximum FPS with hardware acceleration, we must compile the ONNX files into TensorRT `.engine` files. Since we don't use Python/Ultralytics on the Jetson, we use the native `trtexec` binary.

Run these commands inside your running container (this takes ~5-15 minutes per model):

```bash
cd /home/jetson/jetson_edge

# RGB Models
/usr/src/tensorrt/bin/trtexec --onnx=models/rgb_normal.onnx --saveEngine=models/rgb_normal.engine --workspace=1024 --fp16
/usr/src/tensorrt/bin/trtexec --onnx=models/rgb_highlight.onnx --saveEngine=models/rgb_highlight.engine --workspace=1024 --fp16

# IR Models (if applicable)
/usr/src/tensorrt/bin/trtexec --onnx=models/ir_normal.onnx --saveEngine=models/ir_normal.engine --workspace=1024 --fp16
/usr/src/tensorrt/bin/trtexec --onnx=models/ir_highlight.onnx --saveEngine=models/ir_highlight.engine --workspace=1024 --fp16
```
*(Once generated, you do not need to run this step again unless you change your `.onnx` models.)*

---

## Phase 5: Starting the Edge Node

### Step 5.1: Prepare the Main PC
1. Start your main desktop application (`python main.py`).
2. Click the **"+ Add Edge Channel"** button.
3. Click **"Listen for Connection"** to wait for the Jetson.

### Step 5.2: Launch on Jetson
Run the edge script inside the Jetson container. The script will automatically look for the `*.engine` files you generated.

**Standard Run (RGB Mode with video file):**
```bash
cd /home/jetson/jetson_edge
python3 jetson_edge_node.py --video 5.mp4
```

**Optimize with Frame Skipping (`--skip-frames`):**
To drastically improve FPS, you can force the AI model to skip frames, relying on the optical flow tracker for the gaps:
```bash
python3 jetson_edge_node.py --video 5.mp4 --skip-frames 1
```
*(Using `--skip-frames 1` means it detects 1 frame, tracks 1 frame, detects 1 frame... effectively doubling the FPS.)*

**Infrared (IR) Mode:**
```bash
python3 jetson_edge_node.py --video 0 --mode IR
```

As soon as the script starts, it will output `[*] Listening for Main Hub broadcast on UDP port 50050...` and connect automatically within a few seconds (`[+] Connected to Main Hub.`).
