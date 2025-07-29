# 🚀 AutoGearCheck: Smart Quality Control with Computer Vision & Clustering

Welcome to **AutoGearCheck**, a research-driven AI system that combines **Computer Vision** and **Unsupervised Learning** for **automated defect detection and quality clustering** of automotive components. This project was developed as part of a Bachelor's thesis at the Department of Industrial Engineering, Universitas Singaperbangsa Karawang.

---

## 🎯 Project Summary

Traditional manual inspection in manufacturing is prone to error, inefficiency, and subjectivity — especially for small or subtle visual defects like **cracks in gear components**. AutoGearCheck addresses this challenge by automating:

- 🔍 **Crack detection** using **Cascade Mask R-CNN**
- 📊 **Quality clustering** based on defect length using **K-Means**

> This approach significantly reduces undetected defects, boosts consistency, and enables objective product grading.

---

## 🧠 Core Technologies

| Module             | Description |
|--------------------|-------------|
| 🖼️ **Computer Vision** | Detects cracks using **Cascade Mask R-CNN**, trained on augmented defect images. |
| 📉 **Unsupervised Clustering** | Groups product quality into **Grade A, B, and C** using **K-Means** based on crack length. |
| 📏 **SQC** (X̄-R Control Charts) | Verifies the statistical control of the process before and after CV implementation. |

---

## 📦 Dataset Overview

- **Total Original Samples**: 27 crack-defective gear images (augmented to increase dataset size).
- **Defect Focus**: *Crack-type* visual defects only.
- **Measurement Basis**: Length of crack in mm via pixel-to-mm conversion (1 pixel = 0.26 mm).
- **View**: Top-down 2D projection only.

---

## 🖼️ Model Pipeline Overview

1. **Image Preprocessing**
   - Labeling via LabelMe
   - Data Augmentation
2. **Detection Training**
   - Model: Cascade Mask R-CNN
   - Accuracy: `AP50 = 96.63%` (bbox), `85.08%` (mask)
3. **Crack Length Measurement**
   - Extracted from segmentation mask
4. **Clustering**
   - Algorithm: K-Means
   - Metrics: `Silhouette Score = 0.636`, `DBI = 0.432`
   - Output: Quality Grades (A, B, C)
5. **Process Validation**
   - Statistical Quality Control (X̄ and R charts)

---

## 🧪 Demo Snapshots

| Crack Detection | Cluster Visualization |
|------------------|------------------------|
| ![Detection](assets/detection_example.png) | ![Clustering](assets/clustering_example.png) |

---

## 📁 Repository Structure

