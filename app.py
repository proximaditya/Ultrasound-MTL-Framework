import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import time

# --- 1. PAGE SETUP (Professional) ---
# Switched to 'centered' layout for a much better vertical UI flow
st.set_page_config(page_title="MT-CNN Ultrasound AI", page_icon="⚕️", layout="centered")

# Initialize Session State (AI Memory)
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main-header {font-size: 40px; font-weight: bold; color: #1E88E5; margin-bottom: -10px;}
    .sub-header {font-size: 18px; color: #666666; margin-bottom: 30px;}
    .stButton>button {width: 100%; border-radius: 8px; font-size: 18px; font-weight: bold; background-color: #1E88E5; color: white;}
    .stButton>button:hover {background-color: #1565C0; color: white;}
    .report-card {background-color: #f8f9fa; padding: 25px; border-radius: 12px; border-left: 6px solid #1E88E5; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); margin-bottom: 15px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚕️ Multi-Task Clinical Ultrasound Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Multi-Branch ConvNeXt Architecture for Anatomical Classification and Pathology Detection.</div>', unsafe_allow_html=True)

# --- 2. DEFINE THE AI ARCHITECTURE ---
class DualHeadConvNeXt(nn.Module):
    def __init__(self, num_organs, num_statuses):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        self.organ_head = nn.Linear(in_features, num_organs)
        self.status_head = nn.Linear(in_features, num_statuses)

    def forward(self, x):
        features = self.backbone(x)
        return self.organ_head(features), self.status_head(features)

# --- 3. LABELS & CACHING ---
ORGAN_LABELS = {
    0: 'Spleen', 1: 'Pancreas', 2: 'Ovary', 3: 'Gallbladder', 
    4: 'Portal Vein', 5: 'Liver', 6: 'Abdominal Aorta (AA)', 
    7: 'Kidney', 8: 'UB / Prostate / Uterus', 9: 'Hepatic Vein', 10: 'Ascites'
}
STATUS_LABELS = {0: '✅ Normal (Healthy Tissue)', 1: '⚠️ Abnormal (Pathology Detected)'}

@st.cache_resource
def load_model():
    model = DualHeadConvNeXt(num_organs=11, num_statuses=2)
    model.load_state_dict(torch.load('best_multitask_usg.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- 4. IMAGE PREPROCESSING ---
transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --- 5. THE WEBSITE UI (Vertical Stacked Layout) ---

st.markdown("### 1. Upload Scan")
uploaded_file = st.file_uploader("Drag & Drop or Browse (JPG/PNG/WEBP)", type=["jpg", "jpeg", "png", "webp"])

# 1. IMAGE & BUTTON SECTION
if uploaded_file is not None:
    # Reset AI memory if a new file is uploaded
    if uploaded_file.name != st.session_state.last_uploaded_file:
        st.session_state.analyzed = False
        st.session_state.last_uploaded_file = uploaded_file.name

    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Ultrasound', use_container_width=True, clamp=True)
    
    # The Run Button directly under the image
    if st.button("🚀 Run AI Analysis"):
        st.session_state.analyzed = True

# Visual Divider
st.markdown("---")

# 2. CLINICAL REPORT SECTION
st.markdown("### 2. Clinical Report")

if uploaded_file is None:
    st.info("👈 Please upload an ultrasound image to begin analysis.")
else:
    if st.session_state.analyzed:
        # Only run the heavy AI math if we haven't saved it yet
        if 'organ_idx' not in st.session_state or st.session_state.last_uploaded_file == uploaded_file.name:
            with st.spinner('Extracting hierarchical features via ConvNeXt...'):
                time.sleep(0.5) 
                
                img_array = np.array(image)
                tensor_img = transform(image=img_array)['image'].unsqueeze(0)
                
                with torch.no_grad():
                    organ_pred, status_pred = model(tensor_img)
                    
                    st.session_state.organ_idx = organ_pred.argmax(dim=1).item()
                    st.session_state.status_idx = status_pred.argmax(dim=1).item()
                    st.session_state.organ_conf = torch.nn.functional.softmax(organ_pred, dim=1)[0][st.session_state.organ_idx].item() * 100
                    st.session_state.status_conf = torch.nn.functional.softmax(status_pred, dim=1)[0][st.session_state.status_idx].item() * 100
                    st.session_state.organ_raw = organ_pred.numpy().tolist()[0]
                    st.session_state.status_raw = status_pred.numpy().tolist()[0]

        # UX FIX: Dynamic Confidence Warnings!
        if st.session_state.organ_conf < 65.0:
            organ_display = f"⚠️ Unsure (Best Guess: {ORGAN_LABELS[st.session_state.organ_idx]})"
            organ_color = "#F57C00" # Orange
        else:
            organ_display = ORGAN_LABELS[st.session_state.organ_idx]
            organ_color = "#1E88E5" # Blue

        if st.session_state.status_conf < 65.0:
            status_display = f"⚠️ Unsure (Best Guess: {STATUS_LABELS[st.session_state.status_idx]})"
            status_color = "#F57C00" # Orange
        else:
            status_display = STATUS_LABELS[st.session_state.status_idx]
            status_color = '#2E7D32' if st.session_state.status_idx == 0 else '#D32F2F'
        
        # Beautiful HTML Report Card
        st.markdown(f"""
            <div class="report-card">
                <h4 style='color: #555; margin-bottom: 5px;'>Anatomical Identification</h4>
                <h2 style='color: {organ_color}; margin-top: 0px;'>{organ_display}</h2>
                <p style='font-size: 16px;'><b>AI Confidence:</b> {st.session_state.organ_conf:.2f}%</p>
                <hr style='border: 1px solid #ddd;'>
                <h4 style='color: #555; margin-bottom: 5px;'>Pathological Diagnosis</h4>
                <h2 style='color: {status_color}; margin-top: 0px;'>{status_display}</h2>
                <p style='font-size: 16px;'><b>AI Confidence:</b> {st.session_state.status_conf:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Technical Details Dropdown
        with st.expander("⚙️ View Technical AI Details"):
            st.write("**Architecture:** ConvNeXt-Tiny (Multi-Task)")
            st.write(f"**Organ Raw Logits:** {st.session_state.organ_raw}")
            st.write(f"**Pathology Raw Logits:** {st.session_state.status_raw}")

# Anti-Vibration Padding (Forces scrollbar to stay stable)
st.markdown("<br><br><br><br>", unsafe_allow_html=True)