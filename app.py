# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import streamlit as st
# import torch
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import segmentation_models_pytorch as smp
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import time
# import sys

# # ─────────────────────────────────────────
# # Add E:\python-packages to path
# # ─────────────────────────────────────────
# sys.path.insert(0, r'E:\python-packages')

# # ─────────────────────────────────────────
# # Page Config
# # ─────────────────────────────────────────
# st.set_page_config(
#     page_title="NeuroScan AI — Brain Tumor Segmentation",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ─────────────────────────────────────────
# # Full Custom CSS
# # ─────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

# /* ── Base ── */
# *, *::before, *::after { box-sizing: border-box; margin: 0; }

# html, body, .stApp {
#     background-color: #050810 !important;
#     color: #e2e8f0;
#     font-family: 'DM Mono', monospace;
# }

# /* ── Animated background grid ── */
# .stApp::before {
#     content: '';
#     position: fixed;
#     inset: 0;
#     background-image:
#         linear-gradient(rgba(0,200,255,0.03) 1px, transparent 1px),
#         linear-gradient(90deg, rgba(0,200,255,0.03) 1px, transparent 1px);
#     background-size: 40px 40px;
#     pointer-events: none;
#     z-index: 0;
# }

# /* ── Hide default Streamlit elements ── */
# #MainMenu, footer, header { visibility: hidden; }
# .block-container {
#     padding: 2rem 3rem !important;
#     max-width: 1400px !important;
# }

# /* ── Hero Header ── */
# .hero-header {
#     text-align: center;
#     padding: 2.5rem 0 1.5rem;
#     position: relative;
# }
# .hero-badge {
#     display: inline-block;
#     background: linear-gradient(135deg, rgba(0,200,255,0.15), rgba(0,100,255,0.1));
#     border: 1px solid rgba(0,200,255,0.3);
#     color: #00c8ff;
#     font-family: 'DM Mono', monospace;
#     font-size: 0.7rem;
#     letter-spacing: 0.2em;
#     padding: 0.3rem 1rem;
#     border-radius: 2rem;
#     margin-bottom: 1rem;
#     text-transform: uppercase;
# }
# .hero-title {
#     font-family: 'Syne', sans-serif;
#     font-size: 3rem;
#     font-weight: 800;
#     background: linear-gradient(135deg, #ffffff 0%, #00c8ff 50%, #0066ff 100%);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     background-clip: text;
#     line-height: 1.1;
#     margin-bottom: 0.75rem;
# }
# .hero-subtitle {
#     color: #64748b;
#     font-size: 0.9rem;
#     letter-spacing: 0.05em;
#     font-family: 'DM Mono', monospace;
# }

# /* ── Divider ── */
# .neon-divider {
#     height: 1px;
#     background: linear-gradient(90deg, transparent, #00c8ff40, #0066ff60, #00c8ff40, transparent);
#     margin: 1.5rem 0;
#     border: none;
# }

# /* ── Cards ── */
# .glass-card {
#     background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
#     border: 1px solid rgba(255,255,255,0.08);
#     border-radius: 16px;
#     padding: 1.5rem;
#     backdrop-filter: blur(10px);
#     margin-bottom: 1rem;
#     transition: border-color 0.3s;
# }
# .glass-card:hover { border-color: rgba(0,200,255,0.2); }

# /* ── Metric Cards ── */
# .metric-row {
#     display: grid;
#     grid-template-columns: repeat(3, 1fr);
#     gap: 1rem;
#     margin: 1.5rem 0;
# }
# .metric-card {
#     background: linear-gradient(135deg, rgba(0,200,255,0.08), rgba(0,100,255,0.05));
#     border: 1px solid rgba(0,200,255,0.2);
#     border-radius: 12px;
#     padding: 1.2rem;
#     text-align: center;
#     position: relative;
#     overflow: hidden;
# }
# .metric-card::before {
#     content: '';
#     position: absolute;
#     top: 0; left: 0; right: 0;
#     height: 2px;
#     background: linear-gradient(90deg, transparent, #00c8ff, transparent);
# }
# .metric-value {
#     font-family: 'Syne', sans-serif;
#     font-size: 2rem;
#     font-weight: 700;
#     color: #00c8ff;
#     line-height: 1;
#     margin-bottom: 0.3rem;
# }
# .metric-label {
#     font-size: 0.7rem;
#     color: #64748b;
#     letter-spacing: 0.1em;
#     text-transform: uppercase;
# }

# /* ── Status Badges ── */
# .status-ok {
#     display: inline-flex;
#     align-items: center;
#     gap: 0.4rem;
#     background: rgba(0,255,150,0.1);
#     border: 1px solid rgba(0,255,150,0.3);
#     color: #00ff96;
#     padding: 0.3rem 0.8rem;
#     border-radius: 2rem;
#     font-size: 0.75rem;
#     font-family: 'DM Mono', monospace;
# }
# .status-err {
#     display: inline-flex;
#     align-items: center;
#     gap: 0.4rem;
#     background: rgba(255,50,50,0.1);
#     border: 1px solid rgba(255,50,50,0.3);
#     color: #ff5555;
#     padding: 0.3rem 0.8rem;
#     border-radius: 2rem;
#     font-size: 0.75rem;
#     font-family: 'DM Mono', monospace;
# }

# /* ── Tumor Alert ── */
# .tumor-alert {
#     background: linear-gradient(135deg, rgba(255,60,60,0.12), rgba(255,100,0,0.08));
#     border: 1px solid rgba(255,60,60,0.35);
#     border-left: 4px solid #ff3c3c;
#     border-radius: 10px;
#     padding: 1rem 1.2rem;
#     margin: 1rem 0;
#     color: #fca5a5;
#     font-size: 0.85rem;
# }
# .clear-alert {
#     background: linear-gradient(135deg, rgba(0,255,150,0.08), rgba(0,200,100,0.05));
#     border: 1px solid rgba(0,255,150,0.25);
#     border-left: 4px solid #00ff96;
#     border-radius: 10px;
#     padding: 1rem 1.2rem;
#     margin: 1rem 0;
#     color: #86efac;
#     font-size: 0.85rem;
# }

# /* ── Section Headers ── */
# .section-header {
#     font-family: 'Syne', sans-serif;
#     font-size: 1.1rem;
#     font-weight: 700;
#     color: #e2e8f0;
#     margin-bottom: 0.75rem;
#     display: flex;
#     align-items: center;
#     gap: 0.5rem;
# }
# .section-header::after {
#     content: '';
#     flex: 1;
#     height: 1px;
#     background: linear-gradient(90deg, rgba(0,200,255,0.3), transparent);
#     margin-left: 0.5rem;
# }

# /* ── Progress Bar ── */
# .prog-wrap {
#     background: rgba(255,255,255,0.05);
#     border-radius: 999px;
#     height: 8px;
#     overflow: hidden;
#     margin: 0.3rem 0 0.8rem;
# }
# .prog-fill {
#     height: 100%;
#     border-radius: 999px;
#     background: linear-gradient(90deg, #0066ff, #00c8ff);
#     transition: width 0.8s ease;
# }
# .prog-fill-red {
#     height: 100%;
#     border-radius: 999px;
#     background: linear-gradient(90deg, #ff3c3c, #ff8c00);
#     transition: width 0.8s ease;
# }

# /* ── Tabs ── */
# .stTabs [data-baseweb="tab-list"] {
#     gap: 0.5rem;
#     background: rgba(255,255,255,0.03);
#     border-radius: 12px;
#     padding: 0.3rem;
#     border: 1px solid rgba(255,255,255,0.06);
# }
# .stTabs [data-baseweb="tab"] {
#     background: transparent !important;
#     color: #64748b !important;
#     border-radius: 8px !important;
#     font-family: 'DM Mono', monospace !important;
#     font-size: 0.8rem !important;
#     padding: 0.5rem 1.2rem !important;
#     border: none !important;
#     transition: all 0.2s !important;
# }
# .stTabs [aria-selected="true"] {
#     background: linear-gradient(135deg, rgba(0,200,255,0.2), rgba(0,100,255,0.15)) !important;
#     color: #00c8ff !important;
#     border: 1px solid rgba(0,200,255,0.3) !important;
# }
# .stTabs [data-baseweb="tab-panel"] {
#     padding-top: 1.5rem !important;
# }

# /* ── File Uploader ── */
# [data-testid="stFileUploader"] {
#     background: rgba(0,200,255,0.03) !important;
#     border: 2px dashed rgba(0,200,255,0.2) !important;
#     border-radius: 12px !important;
#     transition: border-color 0.3s !important;
# }
# [data-testid="stFileUploader"]:hover {
#     border-color: rgba(0,200,255,0.5) !important;
# }

# /* ── Buttons ── */
# .stButton > button {
#     background: linear-gradient(135deg, #0066ff, #00c8ff) !important;
#     color: white !important;
#     border: none !important;
#     border-radius: 10px !important;
#     padding: 0.7rem 2rem !important;
#     font-family: 'Syne', sans-serif !important;
#     font-weight: 600 !important;
#     font-size: 0.95rem !important;
#     letter-spacing: 0.05em !important;
#     width: 100% !important;
#     transition: all 0.3s !important;
#     box-shadow: 0 4px 20px rgba(0,100,255,0.3) !important;
# }
# .stButton > button:hover {
#     transform: translateY(-2px) !important;
#     box-shadow: 0 8px 30px rgba(0,200,255,0.4) !important;
# }

# /* ── Sidebar ── */
# [data-testid="stSidebar"] {
#     background: linear-gradient(180deg, #080d1a 0%, #050810 100%) !important;
#     border-right: 1px solid rgba(0,200,255,0.1) !important;
# }
# [data-testid="stSidebar"] .block-container {
#     padding: 1.5rem 1rem !important;
# }

# /* ── Selectbox / Radio ── */
# .stSelectbox [data-baseweb="select"] > div,
# .stRadio > div {
#     background: rgba(255,255,255,0.04) !important;
#     border: 1px solid rgba(255,255,255,0.1) !important;
#     border-radius: 8px !important;
#     color: #e2e8f0 !important;
# }

# /* ── Scrollbar ── */
# ::-webkit-scrollbar { width: 4px; }
# ::-webkit-scrollbar-track { background: #050810; }
# ::-webkit-scrollbar-thumb {
#     background: linear-gradient(#0066ff, #00c8ff);
#     border-radius: 4px;
# }

# /* ── Info boxes ── */
# .info-box {
#     background: rgba(0,100,255,0.08);
#     border: 1px solid rgba(0,100,255,0.2);
#     border-radius: 10px;
#     padding: 1rem;
#     font-size: 0.82rem;
#     color: #94a3b8;
#     line-height: 1.6;
# }
# .spec-row {
#     display: flex;
#     justify-content: space-between;
#     padding: 0.4rem 0;
#     border-bottom: 1px solid rgba(255,255,255,0.05);
#     font-size: 0.8rem;
# }
# .spec-key { color: #64748b; }
# .spec-val { color: #00c8ff; font-weight: 500; }

# </style>
# """, unsafe_allow_html=True)

# # ─────────────────────────────────────────
# # Constants
# # ─────────────────────────────────────────
# CHECKPOINT_PATH = r"C:\Users\Ismail\Documents\medical-segmentation\notebooks\outputs\checkpoints\best_model.pth"
# IMG_SIZE        = 128

# # ─────────────────────────────────────────
# # Load Model
# # ─────────────────────────────────────────
# @st.cache_resource
# def load_model():
#     if not os.path.exists(CHECKPOINT_PATH):
#         return None
#     model = smp.Unet(
#         encoder_name    = "mobilenet_v2",
#         encoder_weights = None,
#         in_channels     = 1,
#         classes         = 2,
#         activation      = None
#     )
#     model.load_state_dict(
#         torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
#     )
#     model.eval()
#     return model

# # ─────────────────────────────────────────
# # Helper Functions
# # ─────────────────────────────────────────
# def preprocess_image(img_array, size=128):
#     transform = A.Compose([A.Resize(size, size), ToTensorV2()])
#     img       = img_array.astype(np.float32) / 255.0
#     result    = transform(image=img)
#     return result['image'].float().unsqueeze(0)

# def predict(model, tensor):
#     with torch.no_grad():
#         output    = model(tensor)
#         pred_mask = output.argmax(dim=1).squeeze().numpy()
#     return pred_mask

# def dice_score(pred, target, smooth=1.0):
#     p = (pred == 1).astype(float)
#     t = (target == 1).astype(float)
#     return (2.0*(p*t).sum() + smooth) / (p.sum() + t.sum() + smooth)

# def iou_score(pred, target, smooth=1.0):
#     p = (pred == 1).astype(float)
#     t = (target == 1).astype(float)
#     intersection = (p*t).sum()
#     union        = p.sum() + t.sum() - intersection
#     return (intersection + smooth) / (union + smooth)

# def make_colored_mask(mask):
#     """Convert binary mask to RGB colored image."""
#     colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
#     colored[mask == 1] = [255, 60, 60]
#     return colored

# def plot_results(img_np, pred_mask, true_mask=None):
#     has_gt = true_mask is not None
#     cols   = 4 if has_gt else 3
#     fig, axes = plt.subplots(1, cols, figsize=(cols * 5, 5))
#     fig.patch.set_facecolor('#050810')

#     for ax in axes:
#         ax.set_facecolor('#050810')
#         ax.axis('off')
#         for spine in ax.spines.values():
#             spine.set_visible(False)

#     # Input
#     axes[0].imshow(img_np, cmap='gray')
#     axes[0].set_title('MRI Input', color='#94a3b8',
#                       fontsize=11, fontfamily='monospace', pad=10)

#     if has_gt:
#         axes[1].imshow(true_mask, cmap='Reds', vmin=0, vmax=1)
#         axes[1].set_title('Ground Truth', color='#94a3b8',
#                           fontsize=11, fontfamily='monospace', pad=10)
#         axes[2].imshow(pred_mask, cmap='Reds', vmin=0, vmax=1)
#         axes[2].set_title('AI Prediction', color='#00c8ff',
#                           fontsize=11, fontfamily='monospace', pad=10)
#         axes[3].imshow(img_np, cmap='gray')
#         overlay = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
#         overlay[pred_mask == 1] = [1.0, 0.24, 0.24, 0.65]
#         axes[3].imshow(overlay)
#         axes[3].set_title('Overlay', color='#94a3b8',
#                           fontsize=11, fontfamily='monospace', pad=10)
#     else:
#         axes[1].imshow(pred_mask, cmap='Reds', vmin=0, vmax=1)
#         axes[1].set_title('AI Prediction', color='#00c8ff',
#                           fontsize=11, fontfamily='monospace', pad=10)
#         axes[2].imshow(img_np, cmap='gray')
#         overlay = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
#         overlay[pred_mask == 1] = [1.0, 0.24, 0.24, 0.65]
#         axes[2].imshow(overlay)
#         axes[2].set_title('Overlay', color='#94a3b8',
#                           fontsize=11, fontfamily='monospace', pad=10)

#     plt.tight_layout(pad=1.5)
#     return fig

# # ─────────────────────────────────────────
# # Hero Header
# # ─────────────────────────────────────────
# st.markdown("""
# <div class="hero-header">
#     <div class="hero-badge">⚕ AI-Powered Medical Imaging</div>
#     <div class="hero-title">NeuroScan AI</div>
#     <div class="hero-subtitle">Brain Tumor Detection & Segmentation · U-Net + MobileNetV2</div>
# </div>
# <hr class="neon-divider">
# """, unsafe_allow_html=True)

# # ─────────────────────────────────────────
# # Sidebar
# # ─────────────────────────────────────────
# with st.sidebar:
#     st.markdown("""
#     <div style="text-align:center; padding: 1rem 0;">
#         <div style="font-size:3rem; margin-bottom:0.5rem;">🧠</div>
#         <div style="font-family:'Syne',sans-serif; font-size:1.2rem;
#                     font-weight:700; color:#e2e8f0;">NeuroScan AI</div>
#         <div style="font-size:0.7rem; color:#64748b; letter-spacing:0.1em;">v1.0 · DIAGNOSTIC TOOL</div>
#     </div>
#     <hr class="neon-divider">
#     """, unsafe_allow_html=True)

#     # Model Status
#     model = load_model()
#     if model is not None:
#         st.markdown('<div class="status-ok">● Model Ready</div>', unsafe_allow_html=True)
#     else:
#         st.markdown('<div class="status-err">● Model Not Found</div>', unsafe_allow_html=True)
#         st.markdown("""
#         <div class="info-box">
#         ⚠️ Run <b>Notebook 3</b> to train and save the model first.
#         </div>
#         """, unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)

#     # Settings
#     st.markdown('<div class="section-header">⚙ Settings</div>', unsafe_allow_html=True)

#     colormap = st.selectbox(
#         "Prediction Colormap",
#         ["Hot Red", "Plasma", "Cyan Glow", "Classic Jet"],
#         index=0
#     )
#     cmap_map = {
#         "Hot Red"    : "Reds",
#         "Plasma"     : "plasma",
#         "Cyan Glow"  : "cool",
#         "Classic Jet": "jet"
#     }
#     selected_cmap = cmap_map[colormap]

#     show_confidence = st.toggle("Show Confidence Heatmap", value=True)
#     show_contour    = st.toggle("Show Tumor Contour",      value=True)

#     st.markdown("<br>", unsafe_allow_html=True)

#     # Model Specs
#     st.markdown('<div class="section-header">🔬 Model Specs</div>', unsafe_allow_html=True)
#     st.markdown("""
#     <div class="glass-card">
#         <div class="spec-row"><span class="spec-key">Architecture</span><span class="spec-val">U-Net</span></div>
#         <div class="spec-row"><span class="spec-key">Encoder</span><span class="spec-val">MobileNetV2</span></div>
#         <div class="spec-row"><span class="spec-key">Input Size</span><span class="spec-val">128×128</span></div>
#         <div class="spec-row"><span class="spec-key">Classes</span><span class="spec-val">2 (BG + Tumor)</span></div>
#         <div class="spec-row"><span class="spec-key">Loss</span><span class="spec-val">Dice + CE</span></div>
#         <div class="spec-row"><span class="spec-key">Optimizer</span><span class="spec-val">Adam</span></div>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)

#     # Legend
#     st.markdown('<div class="section-header">🏷 Legend</div>', unsafe_allow_html=True)
#     st.markdown("""
#     <div class="glass-card">
#         <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
#             <div style="width:14px;height:14px;background:#1e293b;border:1px solid #334155;border-radius:3px;"></div>
#             <span style="font-size:0.8rem; color:#94a3b8;">Background — Class 0</span>
#         </div>
#         <div style="display:flex; align-items:center; gap:0.5rem;">
#             <div style="width:14px;height:14px;background:#ff3c3c;border-radius:3px;"></div>
#             <span style="font-size:0.8rem; color:#94a3b8;">Tumor Region — Class 1</span>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# # ─────────────────────────────────────────
# # Main Tabs
# # ─────────────────────────────────────────
# tab1, tab2, tab3, tab4 = st.tabs([
#     "🔬  Scan & Detect",
#     "📊  Metrics & Analysis",
#     "🗂  Batch Processing",
#     "ℹ️  About & Guide"
# ])

# # ══════════════════════════════════════════
# # TAB 1 — Scan & Detect
# # ══════════════════════════════════════════
# with tab1:
#     col_upload, col_preview = st.columns([1, 1], gap="large")

#     with col_upload:
#         st.markdown('<div class="section-header">📁 Upload MRI Scan</div>',
#                     unsafe_allow_html=True)
#         scan_file = st.file_uploader(
#             "Drop your MRI scan here",
#             type=["png", "jpg", "jpeg"],
#             key="scan",
#             help="Upload a grayscale brain MRI slice (.png or .jpg)"
#         )
#         mask_file = st.file_uploader(
#             "Ground Truth Mask (optional)",
#             type=["png", "jpg", "jpeg"],
#             key="mask",
#             help="Upload the corresponding segmentation mask for metric evaluation"
#         )

#         if scan_file:
#             st.markdown("<br>", unsafe_allow_html=True)
#             run_btn = st.button("⚡ Run AI Segmentation", use_container_width=True)
#         else:
#             st.markdown("""
#             <div class="info-box" style="margin-top:1rem;">
#             📌 <b>How to use:</b><br><br>
#             1. Upload a brain MRI scan image<br>
#             2. Optionally upload the ground truth mask<br>
#             3. Click <b>Run AI Segmentation</b><br>
#             4. View results, overlay & metrics
#             </div>
#             """, unsafe_allow_html=True)
#             run_btn = False

#     with col_preview:
#         st.markdown('<div class="section-header">🖼 Preview</div>',
#                     unsafe_allow_html=True)
#         if scan_file:
#             img_pil    = Image.open(scan_file).convert('L')
#             img_np_raw = np.array(img_pil)

#             fig_prev, ax = plt.subplots(figsize=(4, 4))
#             fig_prev.patch.set_facecolor('#0a0f1e')
#             ax.set_facecolor('#0a0f1e')
#             ax.imshow(img_np_raw, cmap='gray')
#             ax.axis('off')
#             ax.set_title('Uploaded Scan', color='#94a3b8',
#                          fontsize=10, fontfamily='monospace')
#             plt.tight_layout(pad=0.5)
#             st.pyplot(fig_prev)
#             plt.close()

#             # Image info
#             st.markdown(f"""
#             <div class="glass-card" style="margin-top:0.5rem;">
#                 <div class="spec-row">
#                     <span class="spec-key">Filename</span>
#                     <span class="spec-val">{scan_file.name}</span>
#                 </div>
#                 <div class="spec-row">
#                     <span class="spec-key">Dimensions</span>
#                     <span class="spec-val">{img_np_raw.shape[1]} × {img_np_raw.shape[0]} px</span>
#                 </div>
#                 <div class="spec-row">
#                     <span class="spec-key">Channels</span>
#                     <span class="spec-val">Grayscale</span>
#                 </div>
#                 <div class="spec-row">
#                     <span class="spec-key">Intensity Range</span>
#                     <span class="spec-val">[{img_np_raw.min()}, {img_np_raw.max()}]</span>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown("""
#             <div style="height:280px; display:flex; align-items:center;
#                         justify-content:center; border:2px dashed rgba(0,200,255,0.1);
#                         border-radius:12px; color:#334155; font-size:0.85rem;
#                         text-align:center; flex-direction:column; gap:0.5rem;">
#                 <div style="font-size:2.5rem;">🧠</div>
#                 <div>Upload a scan to preview</div>
#             </div>
#             """, unsafe_allow_html=True)

#     # ── Run Segmentation ──
#     if scan_file and run_btn:
#         if model is None:
#             st.markdown("""
#             <div class="tumor-alert">
#             ❌ <b>Model checkpoint not found!</b><br>
#             Please complete training in Notebook 3 first.
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             # Progress animation
#             progress_bar = st.progress(0)
#             status_text  = st.empty()

#             steps = [
#                 (15,  "🔄 Preprocessing MRI scan..."),
#                 (40,  "🧠 Running U-Net encoder..."),
#                 (65,  "🔀 Decoding feature maps..."),
#                 (85,  "🎯 Generating segmentation mask..."),
#                 (100, "✅ Analysis complete!")
#             ]
#             for pct, msg in steps:
#                 progress_bar.progress(pct)
#                 status_text.markdown(f"<div style='color:#00c8ff;font-size:0.85rem;'>{msg}</div>",
#                                      unsafe_allow_html=True)
#                 time.sleep(0.3)

#             progress_bar.empty()
#             status_text.empty()

#             # Inference
#             img_np      = np.array(Image.open(scan_file).convert('L'))
#             img_resized = cv2.resize(img_np.astype(np.float32),
#                                      (IMG_SIZE, IMG_SIZE))
#             tensor      = preprocess_image(img_np, IMG_SIZE)
#             pred_mask   = predict(model, tensor)

#             # Ground truth
#             true_mask = None
#             if mask_file:
#                 mask_np   = np.array(Image.open(mask_file).convert('L'))
#                 mask_res  = cv2.resize(mask_np.astype(np.float32),
#                                        (IMG_SIZE, IMG_SIZE),
#                                        interpolation=cv2.INTER_NEAREST)
#                 true_mask = (mask_res > 127).astype(int)

#             # Stats
#             total     = pred_mask.size
#             tumor_pct = (pred_mask == 1).sum() / total * 100
#             bg_pct    = (pred_mask == 0).sum() / total * 100

#             st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)
#             st.markdown('<div class="section-header">🧬 Segmentation Results</div>',
#                         unsafe_allow_html=True)

#             # Result image
#             fig = plot_results(img_resized / 255.0, pred_mask, true_mask)
#             st.pyplot(fig)
#             plt.close()

#             st.markdown("<br>", unsafe_allow_html=True)

#             # Stats row
#             st.markdown(f"""
#             <div class="metric-row">
#                 <div class="metric-card">
#                     <div class="metric-value">{tumor_pct:.1f}%</div>
#                     <div class="metric-label">🔴 Tumor Coverage</div>
#                 </div>
#                 <div class="metric-card">
#                     <div class="metric-value">{bg_pct:.1f}%</div>
#                     <div class="metric-label">⬛ Background</div>
#                 </div>
#                 <div class="metric-card">
#                     <div class="metric-value">{(pred_mask==1).sum()}</div>
#                     <div class="metric-label">📍 Tumor Pixels</div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#             # Tumor presence alert
#             if tumor_pct > 1.0:
#                 st.markdown(f"""
#                 <div class="tumor-alert">
#                 ⚠️ <b>Tumor Region Detected</b> — AI identified a tumor region
#                 covering <b>{tumor_pct:.2f}%</b> of the scan.
#                 This is an AI-assisted tool. Please consult a medical professional for diagnosis.
#                 </div>
#                 """, unsafe_allow_html=True)
#             else:
#                 st.markdown("""
#                 <div class="clear-alert">
#                 ✅ <b>No Significant Tumor Detected</b> — The AI model did not identify
#                 a significant tumor region in this scan slice.
#                 Always verify with a qualified radiologist.
#                 </div>
#                 """, unsafe_allow_html=True)

#             # Confidence heatmap
#             if show_confidence:
#                 st.markdown('<div class="section-header" style="margin-top:1rem;">🌡 Confidence Heatmap</div>',
#                             unsafe_allow_html=True)
#                 with torch.no_grad():
#                     raw_out    = model(tensor)
#                     probs      = torch.softmax(raw_out, dim=1)
#                     tumor_prob = probs[0, 1].numpy()

#                 fig_heat, ax = plt.subplots(1, 2, figsize=(10, 4))
#                 fig_heat.patch.set_facecolor('#050810')
#                 for a in ax: a.set_facecolor('#050810'); a.axis('off')

#                 ax[0].imshow(img_resized / 255.0, cmap='gray')
#                 ax[0].set_title('MRI Scan', color='#94a3b8',
#                                 fontsize=10, fontfamily='monospace')

#                 im = ax[1].imshow(tumor_prob, cmap='hot', vmin=0, vmax=1)
#                 ax[1].set_title('Tumor Probability Map', color='#00c8ff',
#                                 fontsize=10, fontfamily='monospace')
#                 cbar = fig_heat.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
#                 cbar.ax.yaxis.set_tick_params(color='#64748b')
#                 plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#64748b', fontsize=8)

#                 plt.tight_layout()
#                 st.pyplot(fig_heat)
#                 plt.close()

#             # Contour overlay
#             if show_contour and (pred_mask == 1).any():
#                 st.markdown('<div class="section-header" style="margin-top:1rem;">🔍 Tumor Contour</div>',
#                             unsafe_allow_html=True)
#                 img_bgr    = cv2.cvtColor(
#                     (img_resized / img_resized.max() * 255).astype(np.uint8),
#                     cv2.COLOR_GRAY2BGR
#                 )
#                 mask_uint8 = (pred_mask * 255).astype(np.uint8)
#                 contours, _ = cv2.findContours(
#                     mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#                 )
#                 cv2.drawContours(img_bgr, contours, -1, (0, 200, 255), 2)

#                 fig_cont, ax = plt.subplots(figsize=(4, 4))
#                 fig_cont.patch.set_facecolor('#050810')
#                 ax.set_facecolor('#050810')
#                 ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
#                 ax.axis('off')
#                 ax.set_title('Contour Detection', color='#00c8ff',
#                              fontsize=10, fontfamily='monospace')
#                 plt.tight_layout()
#                 st.pyplot(fig_cont)
#                 plt.close()

#             # Store for metrics tab
#             st.session_state['pred_mask'] = pred_mask
#             st.session_state['true_mask'] = true_mask
#             st.session_state['tumor_pct'] = tumor_pct
#             st.session_state['img_np']    = img_resized

# # ══════════════════════════════════════════
# # TAB 2 — Metrics & Analysis
# # ══════════════════════════════════════════
# with tab2:
#     st.markdown('<div class="section-header">📊 Segmentation Metrics</div>',
#                 unsafe_allow_html=True)

#     has_pred = 'pred_mask' in st.session_state
#     has_true = has_pred and st.session_state.get('true_mask') is not None

#     if has_pred and has_true:
#         pred = st.session_state['pred_mask']
#         true = st.session_state['true_mask']

#         d_score = dice_score(pred, true)
#         i_score = iou_score(pred,  true)
#         p_acc   = (pred == true).sum() / true.size
#         sensitivity = ((pred==1) & (true==1)).sum() / max((true==1).sum(), 1)
#         specificity = ((pred==0) & (true==0)).sum() / max((true==0).sum(), 1)

#         # Big metrics
#         st.markdown(f"""
#         <div class="metric-row">
#             <div class="metric-card">
#                 <div class="metric-value">{d_score:.3f}</div>
#                 <div class="metric-label">🎯 Dice Score</div>
#             </div>
#             <div class="metric-card">
#                 <div class="metric-value">{i_score:.3f}</div>
#                 <div class="metric-label">📐 IoU Score</div>
#             </div>
#             <div class="metric-card">
#                 <div class="metric-value">{p_acc:.3f}</div>
#                 <div class="metric-label">✅ Pixel Accuracy</div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

#         # Progress bars
#         col1, col2 = st.columns(2)

#         with col1:
#             st.markdown('<div class="section-header">📈 Score Breakdown</div>',
#                         unsafe_allow_html=True)
#             for label, value in [
#                 ("Dice Score",     d_score),
#                 ("IoU Score",      i_score),
#                 ("Pixel Accuracy", p_acc),
#                 ("Sensitivity",    float(sensitivity)),
#                 ("Specificity",    float(specificity)),
#             ]:
#                 pct = int(value * 100)
#                 color_class = "prog-fill" if value >= 0.5 else "prog-fill-red"
#                 st.markdown(f"""
#                 <div style="margin-bottom:0.8rem;">
#                     <div style="display:flex; justify-content:space-between;
#                                 font-size:0.78rem; margin-bottom:0.3rem;">
#                         <span style="color:#94a3b8;">{label}</span>
#                         <span style="color:#00c8ff; font-weight:600;">{value:.4f}</span>
#                     </div>
#                     <div class="prog-wrap">
#                         <div class="{color_class}" style="width:{pct}%;"></div>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)

#         with col2:
#             st.markdown('<div class="section-header">📊 Metrics Chart</div>',
#                         unsafe_allow_html=True)
#             fig_bar, ax = plt.subplots(figsize=(5, 4))
#             fig_bar.patch.set_facecolor('#0a0f1e')
#             ax.set_facecolor('#0a0f1e')

#             labels  = ['Dice', 'IoU', 'Accuracy', 'Sensitivity', 'Specificity']
#             values  = [d_score, i_score, p_acc,
#                        float(sensitivity), float(specificity)]
#             colors  = ['#0066ff', '#00c8ff', '#00ff96', '#ff8c00', '#a855f7']

#             bars = ax.bar(labels, values, color=colors,
#                           edgecolor='none', width=0.6)
#             ax.set_ylim(0, 1.15)
#             ax.tick_params(colors='#64748b', labelsize=8)
#             ax.spines['bottom'].set_color('#1e293b')
#             ax.spines['left'].set_color('#1e293b')
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.set_facecolor('#0a0f1e')
#             ax.yaxis.label.set_color('#64748b')

#             for bar, val in zip(bars, values):
#                 ax.text(bar.get_x() + bar.get_width()/2,
#                         bar.get_height() + 0.02,
#                         f'{val:.3f}', ha='center',
#                         color='white', fontsize=8, fontweight='bold')

#             plt.xticks(rotation=15)
#             plt.tight_layout()
#             st.pyplot(fig_bar)
#             plt.close()

#         # Confusion matrix style
#         st.markdown('<div class="section-header" style="margin-top:1rem;">🔢 Pixel Classification</div>',
#                     unsafe_allow_html=True)

#         tp = int(((pred==1) & (true==1)).sum())
#         fp = int(((pred==1) & (true==0)).sum())
#         fn = int(((pred==0) & (true==1)).sum())
#         tn = int(((pred==0) & (true==0)).sum())

#         st.markdown(f"""
#         <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; max-width:400px;">
#             <div class="metric-card" style="background:rgba(0,255,150,0.08); border-color:rgba(0,255,150,0.3);">
#                 <div class="metric-value" style="color:#00ff96; font-size:1.5rem;">{tp}</div>
#                 <div class="metric-label">True Positive</div>
#             </div>
#             <div class="metric-card" style="background:rgba(255,100,0,0.08); border-color:rgba(255,100,0,0.3);">
#                 <div class="metric-value" style="color:#ff8c00; font-size:1.5rem;">{fp}</div>
#                 <div class="metric-label">False Positive</div>
#             </div>
#             <div class="metric-card" style="background:rgba(255,60,60,0.08); border-color:rgba(255,60,60,0.3);">
#                 <div class="metric-value" style="color:#ff3c3c; font-size:1.5rem;">{fn}</div>
#                 <div class="metric-label">False Negative</div>
#             </div>
#             <div class="metric-card" style="background:rgba(0,100,255,0.08); border-color:rgba(0,100,255,0.3);">
#                 <div class="metric-value" style="color:#0066ff; font-size:1.5rem;">{tn}</div>
#                 <div class="metric-label">True Negative</div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

#     elif has_pred and not has_true:
#         tumor_pct = st.session_state.get('tumor_pct', 0)
#         st.markdown(f"""
#         <div class="metric-row">
#             <div class="metric-card">
#                 <div class="metric-value">{tumor_pct:.1f}%</div>
#                 <div class="metric-label">🔴 Tumor Coverage</div>
#             </div>
#         </div>
#         <div class="info-box">
#         ℹ️ Upload a <b>ground truth mask</b> along with your scan
#         to unlock full metrics (Dice, IoU, Sensitivity, Specificity).
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown("""
#         <div class="info-box" style="text-align:center; padding:2rem;">
#             <div style="font-size:2rem; margin-bottom:0.5rem;">📊</div>
#             Run a prediction in the <b>Scan & Detect</b> tab first to see metrics here.
#         </div>
#         """, unsafe_allow_html=True)

# # ══════════════════════════════════════════
# # TAB 3 — Batch Processing
# # ══════════════════════════════════════════
# with tab3:
#     st.markdown('<div class="section-header">🗂 Batch MRI Processing</div>',
#                 unsafe_allow_html=True)
#     st.markdown("""
#     <div class="info-box" style="margin-bottom:1.5rem;">
#     Upload multiple MRI scans at once. The AI will process each one and
#     generate a summary report of tumor detections.
#     </div>
#     """, unsafe_allow_html=True)

#     batch_files = st.file_uploader(
#         "Upload multiple MRI scans",
#         type=["png", "jpg", "jpeg"],
#         accept_multiple_files=True,
#         key="batch"
#     )

#     if batch_files and st.button("⚡ Process All Scans", key="batch_run"):
#         if model is None:
#             st.error("Model not loaded!")
#         else:
#             results = []
#             prog    = st.progress(0)

#             for i, f in enumerate(batch_files):
#                 img_np    = np.array(Image.open(f).convert('L'))
#                 tensor    = preprocess_image(img_np, IMG_SIZE)
#                 pred_mask = predict(model, tensor)
#                 tumor_pct = (pred_mask == 1).sum() / pred_mask.size * 100
#                 results.append({
#                     'filename'  : f.name,
#                     'tumor_pct' : tumor_pct,
#                     'detected'  : tumor_pct > 1.0,
#                     'pixels'    : int((pred_mask == 1).sum())
#                 })
#                 prog.progress(int((i+1) / len(batch_files) * 100))

#             prog.empty()

#             # Summary
#             detected = sum(1 for r in results if r['detected'])
#             st.markdown(f"""
#             <div class="metric-row">
#                 <div class="metric-card">
#                     <div class="metric-value">{len(results)}</div>
#                     <div class="metric-label">Total Scans</div>
#                 </div>
#                 <div class="metric-card" style="border-color:rgba(255,60,60,0.3);">
#                     <div class="metric-value" style="color:#ff3c3c;">{detected}</div>
#                     <div class="metric-label">Tumor Detected</div>
#                 </div>
#                 <div class="metric-card" style="border-color:rgba(0,255,150,0.3);">
#                     <div class="metric-value" style="color:#00ff96;">{len(results)-detected}</div>
#                     <div class="metric-label">Clear Scans</div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#             # Results table
#             st.markdown('<div class="section-header">📋 Scan Results</div>',
#                         unsafe_allow_html=True)
#             for r in results:
#                 status = "🔴 Tumor Detected" if r['detected'] else "✅ Clear"
#                 color  = "#ff3c3c" if r['detected'] else "#00ff96"
#                 st.markdown(f"""
#                 <div class="glass-card" style="display:flex; justify-content:space-between;
#                             align-items:center; padding:0.8rem 1.2rem; margin-bottom:0.5rem;">
#                     <span style="font-size:0.85rem; color:#94a3b8;">📄 {r['filename']}</span>
#                     <span style="font-size:0.8rem; color:{color};">{status}</span>
#                     <span style="font-size:0.8rem; color:#00c8ff;">{r['tumor_pct']:.1f}% coverage</span>
#                 </div>
#                 """, unsafe_allow_html=True)

# # ══════════════════════════════════════════
# # TAB 4 — About & Guide
# # ══════════════════════════════════════════
# with tab4:
#     col1, col2 = st.columns(2, gap="large")

#     with col1:
#         st.markdown('<div class="section-header">🧠 About This Project</div>',
#                     unsafe_allow_html=True)
#         st.markdown("""
#         <div class="glass-card">
#         <p style="font-size:0.85rem; color:#94a3b8; line-height:1.8;">
#         <b style="color:#e2e8f0;">NeuroScan AI</b> is a deep learning-powered brain tumor
#         segmentation system built using the U-Net architecture with a MobileNetV2 encoder.
#         It was trained on the Brain Tumor Segmentation dataset from Kaggle.<br><br>
#         The model performs <b style="color:#00c8ff;">pixel-wise binary classification</b>
#         — identifying each pixel as either background or tumor tissue.
#         </p>
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown('<div class="section-header" style="margin-top:1rem;">📖 How to Use</div>',
#                     unsafe_allow_html=True)
#         for i, step in enumerate([
#             ("Upload Scan",     "Go to Scan & Detect tab and upload a brain MRI PNG/JPG"),
#             ("Optional Mask",   "Upload a ground truth mask for full metric evaluation"),
#             ("Run Detection",   "Click 'Run AI Segmentation' button"),
#             ("View Results",    "See prediction, overlay, confidence heatmap & contours"),
#             ("Check Metrics",   "Switch to Metrics tab for Dice, IoU and more"),
#             ("Batch Mode",      "Use Batch Processing tab to analyze multiple scans at once"),
#         ], 1):
#             st.markdown(f"""
#             <div style="display:flex; gap:1rem; margin-bottom:0.8rem; align-items:flex-start;">
#                 <div style="min-width:28px; height:28px; background:linear-gradient(135deg,#0066ff,#00c8ff);
#                             border-radius:50%; display:flex; align-items:center; justify-content:center;
#                             font-size:0.75rem; font-weight:700; color:white; margin-top:2px;">{i}</div>
#                 <div>
#                     <div style="font-size:0.85rem; font-weight:600; color:#e2e8f0;">{step[0]}</div>
#                     <div style="font-size:0.78rem; color:#64748b; margin-top:2px;">{step[1]}</div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#     with col2:
#         st.markdown('<div class="section-header">⚙️ Technical Details</div>',
#                     unsafe_allow_html=True)
#         st.markdown("""
#         <div class="glass-card">
#             <div class="spec-row"><span class="spec-key">Architecture</span><span class="spec-val">U-Net</span></div>
#             <div class="spec-row"><span class="spec-key">Encoder</span><span class="spec-val">MobileNetV2 (ImageNet)</span></div>
#             <div class="spec-row"><span class="spec-key">Input Resolution</span><span class="spec-val">128 × 128 px</span></div>
#             <div class="spec-row"><span class="spec-key">Input Channels</span><span class="spec-val">1 (Grayscale)</span></div>
#             <div class="spec-row"><span class="spec-key">Output Classes</span><span class="spec-val">2 (BG + Tumor)</span></div>
#             <div class="spec-row"><span class="spec-key">Loss Function</span><span class="spec-val">0.5×Dice + 0.5×CE</span></div>
#             <div class="spec-row"><span class="spec-key">Optimizer</span><span class="spec-val">Adam (lr=1e-4)</span></div>
#             <div class="spec-row"><span class="spec-key">Scheduler</span><span class="spec-val">ReduceLROnPlateau</span></div>
#             <div class="spec-row"><span class="spec-key">Dataset</span><span class="spec-val">nikhilroxtomar/brain-tumor-segmentation</span></div>
#             <div class="spec-row"><span class="spec-key">Framework</span><span class="spec-val">PyTorch + SMP</span></div>
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown('<div class="section-header" style="margin-top:1rem;">⚠️ Disclaimer</div>',
#                     unsafe_allow_html=True)
#         st.markdown("""
#         <div class="tumor-alert">
#         This tool is developed for <b>academic and educational purposes only</b>.
#         It is <b>NOT</b> a certified medical device and should <b>NOT</b> be used
#         for clinical diagnosis. Always consult a qualified radiologist or
#         medical professional for diagnosis and treatment decisions.
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown('<div class="section-header" style="margin-top:1rem;">🛠 Tech Stack</div>',
#                     unsafe_allow_html=True)
#         st.markdown("""
#         <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.5rem;">
#         """ + "".join([
#             f'<span style="background:rgba(0,200,255,0.1); border:1px solid rgba(0,200,255,0.2); '
#             f'color:#00c8ff; padding:0.25rem 0.7rem; border-radius:999px; font-size:0.75rem;">{t}</span>'
#             for t in ["PyTorch", "U-Net", "MobileNetV2", "Albumentations",
#                       "Streamlit", "OpenCV", "NumPy", "Matplotlib", "SMP"]
#         ]) + "</div>", unsafe_allow_html=True)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import sys

sys.path.insert(0, r'E:\python-packages')

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body, .stApp {
    background-color: #050810 !important;
    color: #e2e8f0;
    font-family: 'DM Mono', monospace;
}
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(0,200,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

/* Hero */
.hero-header { text-align: center; padding: 2.5rem 0 1.5rem; }
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,200,255,0.15), rgba(0,100,255,0.1));
    border: 1px solid rgba(0,200,255,0.3); color: #00c8ff;
    font-size: 0.7rem; letter-spacing: 0.2em;
    padding: 0.3rem 1rem; border-radius: 2rem;
    margin-bottom: 1rem; text-transform: uppercase;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #00c8ff 50%, #0066ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 0.75rem;
}
.hero-subtitle { color: #64748b; font-size: 0.9rem; letter-spacing: 0.05em; }

/* Divider */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00c8ff40, #0066ff60, #00c8ff40, transparent);
    margin: 1.5rem 0; border: none;
}

/* Cards */
.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08); border-radius: 16px;
    padding: 1.5rem; backdrop-filter: blur(10px);
    margin-bottom: 1rem; transition: border-color 0.3s;
}
.glass-card:hover { border-color: rgba(0,200,255,0.2); }

/* ═══ CUSTOM UPLOAD ZONES ═══ */
.upload-zone-wrap { position: relative; margin-bottom: 0.75rem; }

.upload-zone-ui {
    border: 2px dashed rgba(0,200,255,0.3);
    border-radius: 16px;
    padding: 2.2rem 1.5rem;
    text-align: center;
    background: linear-gradient(135deg, rgba(0,200,255,0.05), rgba(0,100,255,0.02));
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.upload-zone-ui::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(0,200,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.upload-zone-ui:hover {
    border-color: rgba(0,200,255,0.7);
    background: linear-gradient(135deg, rgba(0,200,255,0.1), rgba(0,100,255,0.06));
    box-shadow: 0 0 30px rgba(0,200,255,0.12), inset 0 0 30px rgba(0,200,255,0.03);
    transform: translateY(-2px);
}
.upload-icon-big {
    font-size: 3.2rem; display: block; margin-bottom: 0.9rem;
    filter: drop-shadow(0 0 12px rgba(0,200,255,0.6));
    animation: pulse-icon 2.5s ease-in-out infinite;
}
@keyframes pulse-icon {
    0%, 100% { filter: drop-shadow(0 0 8px rgba(0,200,255,0.4)); transform: scale(1); }
    50%       { filter: drop-shadow(0 0 18px rgba(0,200,255,0.8)); transform: scale(1.05); }
}
.upload-zone-title {
    font-family: 'Syne', sans-serif; font-size: 1.05rem;
    font-weight: 700; color: #e2e8f0; margin-bottom: 0.3rem;
}
.upload-zone-sub { font-size: 0.75rem; color: #64748b; margin-bottom: 1rem; }
.fmt-badges { display: flex; gap: 0.4rem; justify-content: center; flex-wrap: wrap; }
.fmt-badge {
    background: rgba(0,200,255,0.1); border: 1px solid rgba(0,200,255,0.25);
    color: #00c8ff; font-size: 0.62rem; letter-spacing: 0.12em;
    padding: 0.18rem 0.55rem; border-radius: 999px; text-transform: uppercase;
}

/* Optional mask upload — smaller, muted */
.upload-mask-ui {
    border: 1px dashed rgba(100,116,139,0.35);
    border-radius: 12px; padding: 1.2rem 1.5rem;
    text-align: center;
    background: rgba(255,255,255,0.015);
    transition: all 0.3s ease; cursor: pointer;
    position: relative;
}
.upload-mask-ui:hover {
    border-color: rgba(100,116,139,0.7);
    background: rgba(255,255,255,0.035);
}
.upload-mask-icon { font-size: 1.9rem; display: block; margin-bottom: 0.4rem; }
.upload-mask-title { font-size: 0.88rem; font-weight: 600; color: #94a3b8; margin-bottom: 0.2rem; }
.upload-mask-sub { font-size: 0.7rem; color: #475569; }

/* Hide Streamlit's default uploader widget chrome */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"] > label { display: none !important; }
[data-testid="stFileUploader"] > div > div:first-child { display: none !important; }
[data-testid="stFileUploader"] > div { border: none !important; background: transparent !important; padding: 0 !important; }
[data-testid="stFileUploader"] section {
    background: transparent !important; border: none !important; padding: 0 !important;
}
[data-testid="stFileUploader"] section > div > div { display: none !important; }
[data-testid="stFileUploader"] section > button { display: none !important; }

/* File name pill shown after upload */
.file-uploaded-pill {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: linear-gradient(135deg, rgba(0,200,255,0.12), rgba(0,100,255,0.08));
    border: 1px solid rgba(0,200,255,0.3); border-radius: 999px;
    padding: 0.35rem 0.9rem; font-size: 0.76rem; color: #00c8ff;
    margin-top: 0.6rem; font-family: 'DM Mono', monospace;
}
.file-uploaded-pill-mask {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: linear-gradient(135deg, rgba(0,255,150,0.1), rgba(0,200,100,0.06));
    border: 1px solid rgba(0,255,150,0.25); border-radius: 999px;
    padding: 0.35rem 0.9rem; font-size: 0.76rem; color: #00ff96;
    margin-top: 0.6rem; font-family: 'DM Mono', monospace;
}

/* Metrics */
.metric-row {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1rem; margin: 1.5rem 0;
}
.metric-card {
    background: linear-gradient(135deg, rgba(0,200,255,0.08), rgba(0,100,255,0.05));
    border: 1px solid rgba(0,200,255,0.2); border-radius: 12px;
    padding: 1.2rem; text-align: center; position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00c8ff, transparent);
}
.metric-value {
    font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700;
    color: #00c8ff; line-height: 1; margin-bottom: 0.3rem;
}
.metric-label { font-size: 0.7rem; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase; }

/* Status */
.status-ok {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(0,255,150,0.1); border: 1px solid rgba(0,255,150,0.3);
    color: #00ff96; padding: 0.3rem 0.8rem; border-radius: 2rem; font-size: 0.75rem;
}
.status-err {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(255,50,50,0.1); border: 1px solid rgba(255,50,50,0.3);
    color: #ff5555; padding: 0.3rem 0.8rem; border-radius: 2rem; font-size: 0.75rem;
}

/* Alerts */
.tumor-alert {
    background: linear-gradient(135deg, rgba(255,60,60,0.12), rgba(255,100,0,0.08));
    border: 1px solid rgba(255,60,60,0.35); border-left: 4px solid #ff3c3c;
    border-radius: 10px; padding: 1rem 1.2rem; margin: 1rem 0;
    color: #fca5a5; font-size: 0.85rem;
}
.clear-alert {
    background: linear-gradient(135deg, rgba(0,255,150,0.08), rgba(0,200,100,0.05));
    border: 1px solid rgba(0,255,150,0.25); border-left: 4px solid #00ff96;
    border-radius: 10px; padding: 1rem 1.2rem; margin: 1rem 0;
    color: #86efac; font-size: 0.85rem;
}

/* Section Headers */
.section-header {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
    color: #e2e8f0; margin-bottom: 0.75rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-header::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,200,255,0.3), transparent);
    margin-left: 0.5rem;
}

/* Progress bars */
.prog-wrap { background: rgba(255,255,255,0.05); border-radius: 999px; height: 8px; overflow: hidden; margin: 0.3rem 0 0.8rem; }
.prog-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #0066ff, #00c8ff); }
.prog-fill-red { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #ff3c3c, #ff8c00); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem; background: rgba(255,255,255,0.03); border-radius: 12px;
    padding: 0.3rem; border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #64748b !important;
    border-radius: 8px !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important; padding: 0.5rem 1.2rem !important;
    border: none !important; transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,200,255,0.2), rgba(0,100,255,0.15)) !important;
    color: #00c8ff !important; border: 1px solid rgba(0,200,255,0.3) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0066ff, #00c8ff) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    padding: 0.7rem 2rem !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    letter-spacing: 0.05em !important; width: 100% !important;
    transition: all 0.3s !important; box-shadow: 0 4px 20px rgba(0,100,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,200,255,0.4) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d1a 0%, #050810 100%) !important;
    border-right: 1px solid rgba(0,200,255,0.1) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }

/* Info boxes */
.info-box {
    background: rgba(0,100,255,0.08); border: 1px solid rgba(0,100,255,0.2);
    border-radius: 10px; padding: 1rem; font-size: 0.82rem;
    color: #94a3b8; line-height: 1.6;
}
.spec-row {
    display: flex; justify-content: space-between; padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.8rem;
}
.spec-key { color: #64748b; }
.spec-val { color: #00c8ff; font-weight: 500; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #050810; }
::-webkit-scrollbar-thumb { background: linear-gradient(#0066ff, #00c8ff); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
CHECKPOINT_PATH = r"E:\NeuroScan-AI Brain Tumor Segmentation\notebooks\outputs\checkpoints\best_model.pth"
IMG_SIZE = 128

# ─────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    model = smp.Unet(
        encoder_name="mobilenet_v2", encoder_weights=None,
        in_channels=1, classes=2, activation=None
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def preprocess_image(img_array, size=128):
    t = A.Compose([A.Resize(size, size), ToTensorV2()])
    return t(image=img_array.astype(np.float32)/255.0)['image'].float().unsqueeze(0)

def predict(model, tensor):
    with torch.no_grad():
        return model(tensor).argmax(dim=1).squeeze().numpy()

def dice_score(pred, target, smooth=1.0):
    p = (pred==1).astype(float); t = (target==1).astype(float)
    return (2.0*(p*t).sum()+smooth) / (p.sum()+t.sum()+smooth)

def iou_score(pred, target, smooth=1.0):
    p = (pred==1).astype(float); t = (target==1).astype(float)
    inter = (p*t).sum()
    return (inter+smooth) / (p.sum()+t.sum()-inter+smooth)

def plot_results(img_np, pred_mask, true_mask=None):
    has_gt = true_mask is not None
    cols = 4 if has_gt else 3
    fig, axes = plt.subplots(1, cols, figsize=(cols*5, 5))
    fig.patch.set_facecolor('#050810')
    for ax in axes:
        ax.set_facecolor('#050810'); ax.axis('off')
        for s in ax.spines.values(): s.set_visible(False)
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('MRI Input', color='#94a3b8', fontsize=11, fontfamily='monospace', pad=10)
    if has_gt:
        axes[1].imshow(true_mask, cmap='Reds', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth', color='#94a3b8', fontsize=11, fontfamily='monospace', pad=10)
        axes[2].imshow(pred_mask, cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title('AI Prediction', color='#00c8ff', fontsize=11, fontfamily='monospace', pad=10)
        ov = np.zeros((*pred_mask.shape,4), dtype=np.float32)
        ov[pred_mask==1] = [1.0, 0.24, 0.24, 0.65]
        axes[3].imshow(img_np, cmap='gray'); axes[3].imshow(ov)
        axes[3].set_title('Overlay', color='#94a3b8', fontsize=11, fontfamily='monospace', pad=10)
    else:
        axes[1].imshow(pred_mask, cmap='Reds', vmin=0, vmax=1)
        axes[1].set_title('AI Prediction', color='#00c8ff', fontsize=11, fontfamily='monospace', pad=10)
        ov = np.zeros((*pred_mask.shape,4), dtype=np.float32)
        ov[pred_mask==1] = [1.0, 0.24, 0.24, 0.65]
        axes[2].imshow(img_np, cmap='gray'); axes[2].imshow(ov)
        axes[2].set_title('Overlay', color='#94a3b8', fontsize=11, fontfamily='monospace', pad=10)
    plt.tight_layout(pad=1.5)
    return fig

# ─────────────────────────────────────────
# Hero
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">⚕ AI-Powered Medical Imaging</div>
    <div class="hero-title">NeuroScan AI</div>
    <div class="hero-subtitle">Brain Tumor Detection & Segmentation · U-Net + MobileNetV2</div>
</div>
<hr class="neon-divider">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-size:3rem;margin-bottom:0.5rem;">🧠</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#e2e8f0;">NeuroScan AI</div>
        <div style="font-size:0.7rem;color:#64748b;letter-spacing:0.1em;">v2.0 · DIAGNOSTIC TOOL</div>
    </div>
    <hr class="neon-divider">
    """, unsafe_allow_html=True)

    model = load_model()
    if model is not None:
        st.markdown('<div class="status-ok">● Model Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-err">● Model Not Found</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box" style="margin-top:0.5rem;">⚠️ Run <b>Notebook 3</b> to train first.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚙ Settings</div>', unsafe_allow_html=True)
    colormap = st.selectbox("Prediction Colormap", ["Hot Red","Plasma","Cyan Glow","Classic Jet"])
    cmap_map = {"Hot Red":"Reds","Plasma":"plasma","Cyan Glow":"cool","Classic Jet":"jet"}
    selected_cmap = cmap_map[colormap]
    show_confidence = st.toggle("Show Confidence Heatmap", value=True)
    show_contour    = st.toggle("Show Tumor Contour",      value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔬 Model Specs</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="spec-row"><span class="spec-key">Architecture</span><span class="spec-val">U-Net</span></div>
        <div class="spec-row"><span class="spec-key">Encoder</span><span class="spec-val">MobileNetV2</span></div>
        <div class="spec-row"><span class="spec-key">Input Size</span><span class="spec-val">128×128</span></div>
        <div class="spec-row"><span class="spec-key">Classes</span><span class="spec-val">2 (BG + Tumor)</span></div>
        <div class="spec-row"><span class="spec-key">Loss</span><span class="spec-val">Dice + CE</span></div>
        <div class="spec-row"><span class="spec-key">Optimizer</span><span class="spec-val">Adam</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🏷 Legend</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
            <div style="width:14px;height:14px;background:#1e293b;border:1px solid #334155;border-radius:3px;"></div>
            <span style="font-size:0.8rem;color:#94a3b8;">Background — Class 0</span>
        </div>
        <div style="display:flex;align-items:center;gap:0.5rem;">
            <div style="width:14px;height:14px;background:#ff3c3c;border-radius:3px;"></div>
            <span style="font-size:0.8rem;color:#94a3b8;">Tumor Region — Class 1</span>
        </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬  Scan & Detect", "📊  Metrics & Analysis",
    "🗂  Batch Processing", "ℹ️  About & Guide"
])

# ══════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════
with tab1:
    col_upload, col_preview = st.columns([1,1], gap="large")

    with col_upload:
        st.markdown('<div class="section-header">📁 Upload MRI Scan</div>', unsafe_allow_html=True)

        # ── Beautiful MRI upload zone ──
        st.markdown("""
        <div class="upload-zone-ui">
            <span class="upload-icon-big">🧠</span>
            <div class="upload-zone-title">Drop your MRI Scan here</div>
            <div class="upload-zone-sub">Drag & drop or click to browse files</div>
            <div class="fmt-badges">
                <span class="fmt-badge">PNG</span>
                <span class="fmt-badge">JPG</span>
                <span class="fmt-badge">JPEG</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        scan_file = st.file_uploader(
            "upload_scan", type=["png","jpg","jpeg"],
            key="scan", label_visibility="collapsed"
        )
        if scan_file:
            st.markdown(
                f'<div class="file-uploaded-pill">✅ &nbsp;&nbsp;{scan_file.name}</div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Mask upload zone ──
        st.markdown("""
        <div class="upload-mask-ui">
            <span class="upload-mask-icon">🎭</span>
            <div class="upload-mask-title">Ground Truth Mask &nbsp;<span style="font-size:0.65rem;color:#475569;font-weight:400;">Optional</span></div>
            <div class="upload-mask-sub">Upload segmentation mask for metric evaluation</div>
        </div>
        """, unsafe_allow_html=True)

        mask_file = st.file_uploader(
            "upload_mask", type=["png","jpg","jpeg"],
            key="mask", label_visibility="collapsed"
        )
        if mask_file:
            st.markdown(
                f'<div class="file-uploaded-pill-mask">✅ &nbsp;&nbsp;{mask_file.name}</div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        if scan_file:
            run_btn = st.button("⚡ Run AI Segmentation", use_container_width=True)
        else:
            st.markdown("""
            <div class="info-box">
            📌 <b>How to use:</b><br><br>
            1. Upload a brain MRI scan above<br>
            2. Optionally upload a ground truth mask<br>
            3. Click <b>Run AI Segmentation</b><br>
            4. View results, overlay & metrics
            </div>""", unsafe_allow_html=True)
            run_btn = False

    with col_preview:
        st.markdown('<div class="section-header">🖼 Preview</div>', unsafe_allow_html=True)
        if scan_file:
            img_pil    = Image.open(scan_file).convert('L')
            img_np_raw = np.array(img_pil)
            fig_prev, ax = plt.subplots(figsize=(4,4))
            fig_prev.patch.set_facecolor('#0a0f1e')
            ax.set_facecolor('#0a0f1e')
            ax.imshow(img_np_raw, cmap='gray'); ax.axis('off')
            ax.set_title('Uploaded Scan', color='#94a3b8', fontsize=10, fontfamily='monospace')
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_prev); plt.close()
            st.markdown(f"""
            <div class="glass-card" style="margin-top:0.5rem;">
                <div class="spec-row"><span class="spec-key">Filename</span><span class="spec-val">{scan_file.name}</span></div>
                <div class="spec-row"><span class="spec-key">Dimensions</span><span class="spec-val">{img_np_raw.shape[1]} × {img_np_raw.shape[0]} px</span></div>
                <div class="spec-row"><span class="spec-key">Channels</span><span class="spec-val">Grayscale</span></div>
                <div class="spec-row"><span class="spec-key">Intensity Range</span><span class="spec-val">[{img_np_raw.min()}, {img_np_raw.max()}]</span></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:300px;display:flex;align-items:center;justify-content:center;
                        border:2px dashed rgba(0,200,255,0.08);border-radius:16px;
                        flex-direction:column;gap:0.75rem;">
                <div style="font-size:3rem;opacity:0.2;filter:drop-shadow(0 0 10px rgba(0,200,255,0.3));">🧠</div>
                <div style="font-size:0.8rem;color:#1e293b;text-align:center;">
                    Upload a scan<br>to preview here
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Run Segmentation ──
    if scan_file and run_btn:
        if model is None:
            st.markdown('<div class="tumor-alert">❌ <b>Model not found!</b> Run Notebook 3 first.</div>', unsafe_allow_html=True)
        else:
            prog = st.progress(0); status = st.empty()
            for pct, msg in [(15,"🔄 Preprocessing MRI scan..."),(40,"🧠 Running U-Net encoder..."),
                             (65,"🔀 Decoding feature maps..."),(85,"🎯 Generating mask..."),(100,"✅ Done!")]:
                prog.progress(pct)
                status.markdown(f"<div style='color:#00c8ff;font-size:0.85rem;'>{msg}</div>", unsafe_allow_html=True)
                time.sleep(0.3)
            prog.empty(); status.empty()

            img_np      = np.array(Image.open(scan_file).convert('L'))
            img_resized = cv2.resize(img_np.astype(np.float32), (IMG_SIZE, IMG_SIZE))
            tensor      = preprocess_image(img_np, IMG_SIZE)
            pred_mask   = predict(model, tensor)

            true_mask = None
            if mask_file:
                mask_np  = np.array(Image.open(mask_file).convert('L'))
                mask_res = cv2.resize(mask_np.astype(np.float32),(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                true_mask = (mask_res > 127).astype(int)

            tumor_pct = (pred_mask==1).sum() / pred_mask.size * 100
            bg_pct    = (pred_mask==0).sum() / pred_mask.size * 100

            st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">🧬 Segmentation Results</div>', unsafe_allow_html=True)
            fig = plot_results(img_resized/255.0, pred_mask, true_mask)
            st.pyplot(fig); plt.close()

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-value">{tumor_pct:.1f}%</div>
                    <div class="metric-label">🔴 Tumor Coverage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{bg_pct:.1f}%</div>
                    <div class="metric-label">⬛ Background</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{int((pred_mask==1).sum())}</div>
                    <div class="metric-label">📍 Tumor Pixels</div>
                </div>
            </div>""", unsafe_allow_html=True)

            if tumor_pct > 1.0:
                st.markdown(f'<div class="tumor-alert">⚠️ <b>Tumor Detected</b> — covering <b>{tumor_pct:.2f}%</b> of scan. Consult a medical professional.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="clear-alert">✅ <b>No Significant Tumor Detected</b> — Always verify with a qualified radiologist.</div>', unsafe_allow_html=True)

            if show_confidence:
                st.markdown('<div class="section-header" style="margin-top:1rem;">🌡 Confidence Heatmap</div>', unsafe_allow_html=True)
                with torch.no_grad():
                    tumor_prob = torch.softmax(model(tensor), dim=1)[0,1].numpy()
                fig_heat, ax = plt.subplots(1,2,figsize=(10,4))
                fig_heat.patch.set_facecolor('#050810')
                for a in ax: a.set_facecolor('#050810'); a.axis('off')
                ax[0].imshow(img_resized/255.0, cmap='gray')
                ax[0].set_title('MRI Scan', color='#94a3b8', fontsize=10, fontfamily='monospace')
                im = ax[1].imshow(tumor_prob, cmap='hot', vmin=0, vmax=1)
                ax[1].set_title('Tumor Probability Map', color='#00c8ff', fontsize=10, fontfamily='monospace')
                cbar = fig_heat.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
                cbar.ax.yaxis.set_tick_params(color='#64748b')
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#64748b', fontsize=8)
                plt.tight_layout(); st.pyplot(fig_heat); plt.close()

            if show_contour and (pred_mask==1).any():
                st.markdown('<div class="section-header" style="margin-top:1rem;">🔍 Tumor Contour</div>', unsafe_allow_html=True)
                img_bgr = cv2.cvtColor((img_resized/img_resized.max()*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                contours, _ = cv2.findContours((pred_mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_bgr, contours, -1, (0,200,255), 2)
                fig_c, ax = plt.subplots(figsize=(4,4))
                fig_c.patch.set_facecolor('#050810'); ax.set_facecolor('#050810')
                ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); ax.axis('off')
                ax.set_title('Contour Detection', color='#00c8ff', fontsize=10, fontfamily='monospace')
                plt.tight_layout(); st.pyplot(fig_c); plt.close()

            st.session_state.update({'pred_mask':pred_mask,'true_mask':true_mask,'tumor_pct':tumor_pct,'img_np':img_resized})

# ══════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Segmentation Metrics</div>', unsafe_allow_html=True)
    has_pred = 'pred_mask' in st.session_state
    has_true = has_pred and st.session_state.get('true_mask') is not None

    if has_pred and has_true:
        pred=st.session_state['pred_mask']; true=st.session_state['true_mask']
        d=dice_score(pred,true); i=iou_score(pred,true)
        p=(pred==true).sum()/true.size
        sens=((pred==1)&(true==1)).sum()/max((true==1).sum(),1)
        spec=((pred==0)&(true==0)).sum()/max((true==0).sum(),1)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card"><div class="metric-value">{d:.3f}</div><div class="metric-label">🎯 Dice Score</div></div>
            <div class="metric-card"><div class="metric-value">{i:.3f}</div><div class="metric-label">📐 IoU Score</div></div>
            <div class="metric-card"><div class="metric-value">{p:.3f}</div><div class="metric-label">✅ Pixel Accuracy</div></div>
        </div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">📈 Score Breakdown</div>', unsafe_allow_html=True)
            for lbl, val in [("Dice Score",d),("IoU Score",i),("Pixel Accuracy",p),("Sensitivity",float(sens)),("Specificity",float(spec))]:
                cls = "prog-fill" if val>=0.5 else "prog-fill-red"
                st.markdown(f"""<div style="margin-bottom:0.8rem;">
                    <div style="display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:0.3rem;">
                        <span style="color:#94a3b8;">{lbl}</span><span style="color:#00c8ff;font-weight:600;">{val:.4f}</span>
                    </div>
                    <div class="prog-wrap"><div class="{cls}" style="width:{int(val*100)}%;"></div></div>
                </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="section-header">📊 Metrics Chart</div>', unsafe_allow_html=True)
            fig_b, ax = plt.subplots(figsize=(5,4))
            fig_b.patch.set_facecolor('#0a0f1e'); ax.set_facecolor('#0a0f1e')
            vals=[d,i,p,float(sens),float(spec)]
            bars=ax.bar(['Dice','IoU','Acc','Sens','Spec'],vals,color=['#0066ff','#00c8ff','#00ff96','#ff8c00','#a855f7'],edgecolor='none',width=0.6)
            ax.set_ylim(0,1.15); ax.tick_params(colors='#64748b',labelsize=8)
            for s in ['top','right']: ax.spines[s].set_visible(False)
            for s in ['bottom','left']: ax.spines[s].set_color('#1e293b')
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,f'{v:.3f}',ha='center',color='white',fontsize=8,fontweight='bold')
            plt.xticks(rotation=15); plt.tight_layout(); st.pyplot(fig_b); plt.close()
        tp=int(((pred==1)&(true==1)).sum()); fp=int(((pred==1)&(true==0)).sum())
        fn=int(((pred==0)&(true==1)).sum()); tn=int(((pred==0)&(true==0)).sum())
        st.markdown('<div class="section-header" style="margin-top:1rem;">🔢 Pixel Classification</div>', unsafe_allow_html=True)
        st.markdown(f"""<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;max-width:400px;">
            <div class="metric-card" style="background:rgba(0,255,150,0.08);border-color:rgba(0,255,150,0.3);"><div class="metric-value" style="color:#00ff96;font-size:1.5rem;">{tp}</div><div class="metric-label">True Positive</div></div>
            <div class="metric-card" style="background:rgba(255,100,0,0.08);border-color:rgba(255,100,0,0.3);"><div class="metric-value" style="color:#ff8c00;font-size:1.5rem;">{fp}</div><div class="metric-label">False Positive</div></div>
            <div class="metric-card" style="background:rgba(255,60,60,0.08);border-color:rgba(255,60,60,0.3);"><div class="metric-value" style="color:#ff3c3c;font-size:1.5rem;">{fn}</div><div class="metric-label">False Negative</div></div>
            <div class="metric-card" style="background:rgba(0,100,255,0.08);border-color:rgba(0,100,255,0.3);"><div class="metric-value" style="color:#0066ff;font-size:1.5rem;">{tn}</div><div class="metric-label">True Negative</div></div>
        </div>""", unsafe_allow_html=True)
    elif has_pred:
        st.markdown(f"""<div class="metric-row"><div class="metric-card"><div class="metric-value">{st.session_state.get('tumor_pct',0):.1f}%</div><div class="metric-label">🔴 Tumor Coverage</div></div></div>
        <div class="info-box">ℹ️ Upload a <b>ground truth mask</b> to unlock full metrics.</div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box" style="text-align:center;padding:2rem;"><div style="font-size:2rem;margin-bottom:0.5rem;">📊</div>Run a prediction in <b>Scan & Detect</b> tab first.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🗂 Batch MRI Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box" style="margin-bottom:1.5rem;">Upload multiple MRI scans at once. The AI processes each and generates a tumor detection summary.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-zone-ui" style="border-color:rgba(168,85,247,0.35);
         background:linear-gradient(135deg,rgba(168,85,247,0.06),rgba(99,102,241,0.03));">
        <span class="upload-icon-big" style="filter:drop-shadow(0 0 12px rgba(168,85,247,0.7));">🗂️</span>
        <div class="upload-zone-title">Drop Multiple MRI Scans</div>
        <div class="upload-zone-sub">Select multiple files at once for batch analysis</div>
        <div class="fmt-badges">
            <span class="fmt-badge" style="border-color:rgba(168,85,247,0.3);color:#a855f7;background:rgba(168,85,247,0.1);">PNG</span>
            <span class="fmt-badge" style="border-color:rgba(168,85,247,0.3);color:#a855f7;background:rgba(168,85,247,0.1);">JPG</span>
            <span class="fmt-badge" style="border-color:rgba(168,85,247,0.3);color:#a855f7;background:rgba(168,85,247,0.1);">MULTI-FILE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader("batch_hidden", type=["png","jpg","jpeg"],
                                    accept_multiple_files=True, key="batch", label_visibility="collapsed")
    if batch_files:
        pills = "".join([f'<span class="file-uploaded-pill">📄 &nbsp;{f.name}</span>' for f in batch_files])
        st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin:0.5rem 0 1rem;">{pills}</div>', unsafe_allow_html=True)

    if batch_files and st.button("⚡ Process All Scans", key="batch_run"):
        if model is None:
            st.error("Model not loaded!")
        else:
            results=[]; prog=st.progress(0)
            for i, f in enumerate(batch_files):
                img_np=np.array(Image.open(f).convert('L'))
                pred_mask=predict(model, preprocess_image(img_np,IMG_SIZE))
                t_pct=(pred_mask==1).sum()/pred_mask.size*100
                results.append({'filename':f.name,'tumor_pct':t_pct,'detected':t_pct>1.0,'pixels':int((pred_mask==1).sum())})
                prog.progress(int((i+1)/len(batch_files)*100))
            prog.empty()
            detected=sum(1 for r in results if r['detected'])
            st.markdown(f"""<div class="metric-row">
                <div class="metric-card"><div class="metric-value">{len(results)}</div><div class="metric-label">Total Scans</div></div>
                <div class="metric-card" style="border-color:rgba(255,60,60,0.3);"><div class="metric-value" style="color:#ff3c3c;">{detected}</div><div class="metric-label">Tumor Detected</div></div>
                <div class="metric-card" style="border-color:rgba(0,255,150,0.3);"><div class="metric-value" style="color:#00ff96;">{len(results)-detected}</div><div class="metric-label">Clear Scans</div></div>
            </div>""", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📋 Scan Results</div>', unsafe_allow_html=True)
            for r in results:
                color="#ff3c3c" if r['detected'] else "#00ff96"
                status="🔴 Tumor Detected" if r['detected'] else "✅ Clear"
                st.markdown(f"""<div class="glass-card" style="display:flex;justify-content:space-between;align-items:center;padding:0.8rem 1.2rem;margin-bottom:0.5rem;">
                    <span style="font-size:0.85rem;color:#94a3b8;">📄 {r['filename']}</span>
                    <span style="font-size:0.8rem;color:{color};">{status}</span>
                    <span style="font-size:0.8rem;color:#00c8ff;">{r['tumor_pct']:.1f}% coverage</span>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 4
# ══════════════════════════════════════════
with tab4:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="section-header">🧠 About This Project</div>', unsafe_allow_html=True)
        st.markdown("""<div class="glass-card"><p style="font-size:0.85rem;color:#94a3b8;line-height:1.8;">
        <b style="color:#e2e8f0;">NeuroScan AI</b> is a deep learning brain tumor segmentation system
        using U-Net + MobileNetV2, trained on the Kaggle Brain Tumor Segmentation dataset.<br><br>
        Performs <b style="color:#00c8ff;">pixel-wise binary classification</b> — background vs tumor tissue.
        </p></div>""", unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="margin-top:1rem;">📖 How to Use</div>', unsafe_allow_html=True)
        for i,(t,d) in enumerate([("Upload Scan","Upload brain MRI PNG/JPG in Scan & Detect tab"),
                                    ("Optional Mask","Upload mask for full metric evaluation"),
                                    ("Run Detection","Click Run AI Segmentation"),
                                    ("View Results","See prediction, overlay, heatmap & contours"),
                                    ("Check Metrics","Switch to Metrics tab for Dice, IoU and more"),
                                    ("Batch Mode","Use Batch Processing for multiple scans")],1):
            st.markdown(f"""<div style="display:flex;gap:1rem;margin-bottom:0.8rem;align-items:flex-start;">
                <div style="min-width:28px;height:28px;background:linear-gradient(135deg,#0066ff,#00c8ff);
                            border-radius:50%;display:flex;align-items:center;justify-content:center;
                            font-size:0.75rem;font-weight:700;color:white;margin-top:2px;">{i}</div>
                <div><div style="font-size:0.85rem;font-weight:600;color:#e2e8f0;">{t}</div>
                     <div style="font-size:0.78rem;color:#64748b;margin-top:2px;">{d}</div></div>
            </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-header">⚙️ Technical Details</div>', unsafe_allow_html=True)
        st.markdown("""<div class="glass-card">
            <div class="spec-row"><span class="spec-key">Architecture</span><span class="spec-val">U-Net</span></div>
            <div class="spec-row"><span class="spec-key">Encoder</span><span class="spec-val">MobileNetV2</span></div>
            <div class="spec-row"><span class="spec-key">Input Resolution</span><span class="spec-val">128 × 128 px</span></div>
            <div class="spec-row"><span class="spec-key">Input Channels</span><span class="spec-val">1 (Grayscale)</span></div>
            <div class="spec-row"><span class="spec-key">Output Classes</span><span class="spec-val">2 (BG + Tumor)</span></div>
            <div class="spec-row"><span class="spec-key">Loss Function</span><span class="spec-val">0.5×Dice + 0.5×CE</span></div>
            <div class="spec-row"><span class="spec-key">Optimizer</span><span class="spec-val">Adam (lr=1e-4)</span></div>
            <div class="spec-row"><span class="spec-key">Dataset</span><span class="spec-val">nikhilroxtomar/brain-tumor</span></div>
            <div class="spec-row"><span class="spec-key">Framework</span><span class="spec-val">PyTorch + SMP</span></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="margin-top:1rem;">⚠️ Disclaimer</div>', unsafe_allow_html=True)
        st.markdown('<div class="tumor-alert">For <b>academic purposes only</b>. <b>NOT</b> a certified medical device. Do <b>NOT</b> use for clinical diagnosis. Always consult a qualified radiologist.</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="margin-top:1rem;">🛠 Tech Stack</div>', unsafe_allow_html=True)
        st.markdown("<div style='display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:0.5rem;'>" +
            "".join([f'<span style="background:rgba(0,200,255,0.1);border:1px solid rgba(0,200,255,0.2);color:#00c8ff;padding:0.25rem 0.7rem;border-radius:999px;font-size:0.75rem;">{t}</span>'
                     for t in ["PyTorch","U-Net","MobileNetV2","Albumentations","Streamlit","OpenCV","NumPy","Matplotlib","SMP"]]) +
            "</div>", unsafe_allow_html=True)