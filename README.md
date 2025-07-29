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
| 🖼️ **Computer Vision** | Detects cracks using **Cascade Mask R-CNN**, trained on augmented defect images, measure crack length.|
| 📉 **Unsupervised Clustering** | Groups product quality into **Grade A, B, and C** using **K-Means** based on crack length. |

---

## 📦 Dataset Overview

- **Total Original Samples**: 27 crack-defective gear images (augmented to increase dataset size).
- **Defect Focus**: *Crack-type* visual defects only.
- **Measurement Basis**: Length of crack in mm via pixel-to-mm conversion (1 pixel = 0.26 mm).
- **View**: Top-down 2D projection only.

---

## 🧠 Research Flow

Flowchart for Implementing Crack Detection and Quality Classification on Gear Components using Cascade Mask R-CNN and K-Means

![Research Flow](./assets/RESEARCH%20FLOW.png)

---
## 🧪 Model Interface Demo 

<p align="center">
  <img src="assets/interface_demo.png" alt="AutoGearCheck Interface Demo" width="600"/>
  <br>
  <em>Figure. Interface preview of defect detection and quality clustering system.</em>
</p>

*Try the AutoGearCheck application live via Streamlit:*

👉 🔗 [Launch Specura Streamlit App]([https://specura.streamlit.app/](https://crack-detection1.streamlit.app/))
---

## 📁 Repository Structure

