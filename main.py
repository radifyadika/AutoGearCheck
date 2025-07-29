import streamlit as st 
import numpy as np
from PIL import Image
import cv2
import pickle
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from util import set_background, visualize_with_mask
import gdown
import os

model_path = './model/model_final.pth'

# Cek apakah model sudah ada
if not os.path.exists(model_path):
    # Download dari Google Drive
    file_id = '10Be-WNw8svw1G-VZw-qmDeIjr5YVrHsr'
    url = f'https://drive.google.com/uc?id={file_id}'
    os.makedirs('./model', exist_ok=True)
    gdown.download(url, model_path, quiet=False)

def visualize_with_mask(original, detected, mask):
    # Custom CSS untuk memastikan teks tab terlihat di dark mode
    st.markdown("""
        <style>
            .stRadio > div {
                flex-direction: row;
                justify-content: center;
            }
            .stRadio label {
                font-weight: 600;
                color: white !important;
                background-color: #1f1f1f;
                border: 1px solid #444;
                padding: 0.4em 1.2em;
                border-radius: 0.5em;
                margin-right: 0.5em;
            }
            .stRadio div[role="radiogroup"] > label[data-selected="true"] {
                background-color: #ffffff10 !important;
                color: #ddd !important;
                border: 1px solid #888 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    option = st.radio("Pilih Tampilan Gambar:", options=["Original", "Deteksi", "Mask"], horizontal=True, label_visibility="collapsed")

    if option == "Original":
        st.image(original, caption="Gambar Asli", use_container_width=True)
    elif option == "Deteksi":
        st.image(detected, caption="Hasil Deteksi", use_container_width=True)
    elif option == "Mask":
        st.image(mask, caption="Hasil Mask", use_container_width=True)
# --- Setup background ---
set_background('./bg.png')

# --- Load KMeans model ---
with open('./kmeans_model_3_clusters.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# --- Register dataset & load model ---
DATA_SET_NAME = "my_dataset"
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = './model/model_final.pth'
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.DATASETS.TEST = (f"{DATA_SET_NAME}_valid",)
predictor = DefaultPredictor(cfg)
train_metadata = MetadataCatalog.get(f"{DATA_SET_NAME}_train")

# --- Sidebar Configuration ---
st.sidebar.header("Gear Crack Detection")

# --- Upload Image ---
uploaded_file = st.sidebar.file_uploader("📤 Upload Gambar Gear (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Prediction
    outputs = predictor(image_np)
    v = Visualizer(image_np[:, :, ::-1], metadata=train_metadata, scale=0.8)
    vis_output = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    vis_image = vis_output.get_image()[:, :, ::-1] 

    # Combined Mask
    masks = outputs["instances"].pred_masks.cpu().numpy()
    combined_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    for m in masks:
        combined_mask = np.maximum(combined_mask, m.astype(np.uint8))
    combined_mask_img = (combined_mask * 255).astype(np.uint8)

    # Crack Length
    pixel_to_mm = 0.26
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_length_px = sum(cv2.arcLength(cnt, True) for cnt in contours)
    total_length_mm = total_length_px * pixel_to_mm / 15

    # Clustering
    features = np.array([[total_length_mm]])
    cluster = kmeans.predict(features)[0]
    cluster_to_grade = {
        1: "Grade A (Kualitas Terbaik) ✅",
        2: "Grade B (Kualitas Sedang) ⚠️",
        0: "Grade C (Kualitas Rendah) ❌"
    }
    grade = cluster_to_grade.get(cluster, "Tidak diketahui")

    # --- Sidebar Results ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<h4>📏 Estimasi Panjang Retakan:</h4><p style='color:#e74c3c;font-size:25px'><b>{total_length_mm:.2f} mm</b></p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<h4>🏷️ Grade Kualitas:</h4><p style='color:#3498db;font-size:25px'><b>{grade}</b></p>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <hr>
    <div style='font-size: 13px;'>
        <strong>ℹ️ Disclaimer:</strong><br>
        Hasil ini merupakan estimasi berdasarkan deteksi visual menggunakan segmentasi & kontur.
        <i>Semakin panjang retakan</i>, kemungkinan kerusakan struktural meningkat. Disarankan untuk pemeriksaan atau perbaikan lebih lanjut.<br>
        
    </div>
    """, unsafe_allow_html=True)

    # --- Main Content Area: Display Visualization ---
    visualize_with_mask(image_np, vis_image, combined_mask_img)
