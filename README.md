# Distillation of YOLO11 for Bacteria Detection in Microscopic Images

## Project Overview

This project focuses on the **distillation of YOLO11** for automated bacteria detection in microscopic images, with a specific case study on **Pseudomonas aeruginosa**. The goal is to create a lightweight, efficient model that maintains high detection accuracy while being suitable for deployment on resource-constrained embedded systems.

## 🔬 Problem Statement

Bacteria detection in microscopic images is a critical task in medical diagnostics and research. Traditional manual inspection is time-consuming and prone to human error. This project addresses the challenge of developing an automated system that can:

- Detect bacteria in real-time microscopic video streams
- Maintain high accuracy with minimal computational resources
- Enable deployment on embedded systems for point-of-care applications

## 🎯 Key Features

- **Knowledge Distillation**: Leverages a larger teacher model (YOLO11l) to train a smaller student model (YOLO11n)
- **Real-time Processing**: Optimized for real-time bacteria detection and tracking
- **Multiple Training Modes**: Supports teacher-only, student-only, and distillation training
- **Embedded System Ready**: Designed for deployment on resource-constrained devices
- **Comprehensive Evaluation**: Includes performance metrics and benchmarking tools

## 📊 Results

### Detection Performance
Our distilled YOLO11n model achieves:
- **High Detection Accuracy**: Maintains competitive performance with the teacher model
- **Real-time Processing**: Processes microscopic video streams at target FPS
- **Efficient Resource Usage**: Optimized for embedded deployment

### Demo Video
![Bacteria Detection Results](/home/yuguerten/workspace/distillation_yolo/demo/image_detection.png)

*Real-time bacteria detection results showing automated identification and tracking of Pseudomonas aeruginosa in microscopic images.*

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/distillation_yolo.git
   cd distillation_yolo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**
   - Organize your microscopic images in YOLO format
   - Update the dataset path in `dataset_sliced/data.yaml`

## 🚀 Usage

### Training Modes

The project supports three training modes:

#### 1. Distillation Training (Recommended)
Train a lightweight student model using knowledge from a teacher model:
```bash
python train.py --mode distill --epochs 100 --batch 16 --distillation_loss cwd
```

#### 2. Teacher Model Training
Train only the teacher model (YOLO11l):
```bash
python train.py --mode teacher --epochs 100 --batch 8
```

#### 3. Student Model Training
Train only the student model (YOLO11n):
```bash
python train.py --mode student --epochs 100 --batch 16
```

### Real-time Detection

Run real-time bacteria detection on microscopic video:
```bash
python realtime_bacteria_detection.py --video_path /path/to/video.avi --model_path /path/to/best.pt
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epochs` | Number of training epochs | 10 |
| `--batch` | Batch size | 16 |
| `--seed` | Random seed for reproducibility | 42 |
| `--workers` | Number of data loading workers | 0 |
| `--distillation_loss` | Type of distillation loss | 'cwd' |
| `--mode` | Training mode (distill/teacher/student) | 'distill' |

## 📁 Project Structure

```
distillation_yolo/
├── train.py                          # Main training script
├── realtime_bacteria_detection.py    # Real-time detection pipeline
├── requirements.txt                   # Python dependencies
├── gpu_performance_metrics_optimize.json  # Performance benchmarks
├── dataset_sliced/                    # Dataset configuration
│   └── data.yaml                     # Dataset paths and classes
├── runs/                             # Training results and models
│   └── detect/                       # Detection experiments
├── ultralytics/                      # Modified YOLO framework
└── docs/                            # Documentation
```

## 🔬 Methodology

### Knowledge Distillation Approach

1. **Teacher Model**: YOLO11l - Large model with high accuracy
2. **Student Model**: YOLO11n - Lightweight model for deployment
3. **Distillation Loss**: Channel-wise Distillation (CWD) for effective knowledge transfer
4. **Training Strategy**: Multi-stage training with progressive distillation

### Optimization Techniques

- **Slice-based Processing**: Handles high-resolution microscopic images
- **Tracking Integration**: Maintains object continuity across frames
- **Performance Monitoring**: Real-time metrics collection and analysis

## 📈 Performance Metrics

Our system achieves:
- **Average FPS**: 2.66 (target: 1 FPS for real-time processing)
- **Total Detections**: 148,388 across test sequences
- **Average Inference Time**: 0.19 seconds per frame
- **Detection Accuracy**: High precision on Pseudomonas aeruginosa

## 🔧 Configuration

### Model Configuration
- **Teacher Model**: YOLO11l (pre-trained)
- **Student Model**: YOLO11n (distilled)
- **Input Size**: 640x640 pixels
- **Confidence Threshold**: 0.2
- **NMS IoU Threshold**: 0.4

### Hardware Requirements
- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with CUDA support
- **Embedded**: Jetson Nano/Xavier (tested compatibility)

## 🎓 Academic Context

This project is part of a **Final Year Project (PFE)** focused on applying deep learning techniques to medical image analysis. The research contributes to the field of automated microscopy and embedded AI systems.

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: For the YOLO framework
- **Research Team**: For guidance and support
- **Open Source Community**: For tools and libraries used

## 📬 Contact

For questions or collaborations, please contact:
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)

---

*This project demonstrates the application of knowledge distillation techniques to create efficient, deployable AI systems for medical image analysis.*