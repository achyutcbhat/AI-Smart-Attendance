# AI-Based Smart Attendance System Using Face Detection and Python

## 1. Project Overview

An AI-based smart attendance system that uses face detection, face recognition, anti-spoofing, background analysis, and score fusion to automatically record student attendance.

## 2. Main Features

- Face Detection using RetinaFace
- Face Alignment using facial landmarks
- Face Recognition using InsightFace and ArcFace
- Student Registration with face embeddings
- Anti-Spoofing using MobileNetV3-based CNN
- Background Analysis
- Score Fusion
- Duplicate Attendance Prevention
- MySQL Database
- Configurable thresholds and weights
- System logging

## 3. System Workflow

Camera
↓
Face Detection
↓
Face Alignment
↓
Face Recognition
↓
Identity Matching
↓
Liveness Check
↓
Background Check
↓
Score Fusion
↓
Attendance Decision
↓
MySQL Database

## 4. Technologies Used

- Python 3.11
- OpenCV
- RetinaFace
- InsightFace
- ArcFace
- PyTorch
- MobileNetV3
- NumPy
- Pandas
- Scikit-learn
- MySQL
- PyYAML

## 5. Project Structure

```text
AI-Smart-Attendance/
├── src/
├── data/
├── models/
├── tests/
├── experiments/
├── results/
├── logs/
├── requirements.txt
├── config.yaml
└── README.md