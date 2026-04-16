import os
import io
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import streamlit as st
from models.stegnet import StegNet
from models.unet_purifier import UNetPurifier
import datetime
STEGNET_CHECKPOINT = 'checkpoints/stegnet_best.pth'
UNET_CHECKPOINT = 'checkpoints/unet_best.pth'
IMAGE_SIZE = 256
MOCK_USERS = {'admin': {'password': 'admin123', 'role': 'admin'}, 'user': {'password': 'user123', 'role': 'user'}}

if __name__ == '__main__':
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        from streamlit.web import cli as stcli
        import sys
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
st.set_page_config(page_title='Stego-GAN: Adversarial Steganography Hunter', page_icon='🔍', layout='wide', initial_sidebar_state='expanded')
st.markdown('\n<style>\n    /* Main container */\n    .main { background-color: #0e1117; }\n\n    /* Title styling */\n    .hero-title {\n        font-size: 2.4rem;\n        font-weight: 800;\n        background: linear-gradient(90deg, #ff4b4b, #ff8c00, #ffd700);\n        -webkit-background-clip: text;\n        -webkit-text-fill-color: transparent;\n        margin-bottom: 0.2rem;\n    }\n    .hero-sub {\n        color: #8d9db6;\n        font-size: 1.0rem;\n        margin-bottom: 1.5rem;\n    }\n\n    /* Metric cards */\n    .metric-card {\n        background: #1c2333;\n        border: 1px solid #2d3748;\n        border-radius: 12px;\n        padding: 1.2rem 1.5rem;\n        text-align: center;\n        margin-bottom: 1rem;\n    }\n    .metric-value {\n        font-size: 2.5rem;\n        font-weight: 700;\n    }\n    .metric-label {\n        color: #8d9db6;\n        font-size: 0.85rem;\n        margin-top: 0.2rem;\n    }\n\n    /* Status badges */\n    .badge-clean   { color: #22c55e; font-weight: 700; }\n    .badge-stego   { color: #ef4444; font-weight: 700; }\n    .badge-warning { color: #f59e0b; font-weight: 700; }\n\n    /* Section headers */\n    .section-header {\n        font-size: 1.1rem;\n        font-weight: 600;\n        color: #e2e8f0;\n        border-left: 3px solid #ff4b4b;\n        padding-left: 0.75rem;\n        margin: 1rem 0 0.5rem 0;\n    }\n\n    /* Info box */\n    .info-box {\n        background: #1a2035;\n        border: 1px solid #2a3555;\n        border-radius: 8px;\n        padding: 1rem;\n        color: #94a3b8;\n        font-size: 0.875rem;\n    }\n\n    /* Streamlit element overrides */\n    div[data-testid="stMetric"] { background: #1c2333; border-radius: 10px; padding: 1rem; }\n    .stProgress > div > div { background: linear-gradient(90deg, #22c55e, #ef4444); }\n</style>\n', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading StegNet detector…')
def load_stegnet() -> Tuple[StegNet, str]:
    device = torch.device('cpu')
    model = StegNet().to(device)
    model.eval()
    if os.path.isfile(STEGNET_CHECKPOINT):
        ckpt = torch.load(STEGNET_CHECKPOINT, map_location=device)
        state_dict = ckpt['model_state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        epoch = ckpt.get('epoch', '?')
        val_loss = ckpt.get('val_loss', float('nan'))
        status = f'✅ Loaded (epoch {epoch}, val_loss={val_loss:.4f})'
    else:
        status = '⚠️ No checkpoint found — using random weights (train first)'
    return (model, status)

@st.cache_resource(show_spinner='Loading U-Net purifier…')
def load_unet() -> Tuple[UNetPurifier, str]:
    device = torch.device('cpu')
    model = UNetPurifier().to(device)
    model.eval()
    if os.path.isfile(UNET_CHECKPOINT):
        ckpt = torch.load(UNET_CHECKPOINT, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            if k.startswith('head.'):
                k = k.replace('head.', 'output_head.')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        epoch = ckpt.get('epoch', '?')
        val_loss = ckpt.get('val_loss', float('nan'))
        status = f'✅ Loaded (epoch {epoch}, val_loss={val_loss:.4f})'
    else:
        status = '⚠️ No checkpoint found — using random weights (train first)'
    return (model, status)

def preprocess_image(pil_image: Image.Image, size: int=IMAGE_SIZE) -> torch.Tensor:
    img = pil_image.convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor.unsqueeze(0)
_SRM_KERNEL = None

def _get_srm_kernel() -> torch.Tensor:
    global _SRM_KERNEL
    if _SRM_KERNEL is None:
        k = torch.tensor([[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 1.0, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]], dtype=torch.float32)
        _SRM_KERNEL = k.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    return _SRM_KERNEL

def preprocess_image_for_detection(pil_image: Image.Image, size: int=IMAGE_SIZE) -> torch.Tensor:
    raw = preprocess_image(pil_image, size)
    with torch.no_grad():
        filtered = F.conv2d(raw, _get_srm_kernel(), padding=1, groups=3)
    return filtered

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')

def overlay_heatmap(original_pil: Image.Image, heatmap: np.ndarray, alpha: float=0.5, colormap: str='jet') -> Image.Image:
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap)[:, :, :3]
    orig_w, orig_h = original_pil.size
    colored_uint8 = (colored * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(colored_uint8, mode='RGB').resize((orig_w, orig_h), Image.LANCZOS)
    orig_arr = np.array(original_pil.convert('RGB'), dtype=np.float32) / 255.0
    heatmap_arr = np.array(heatmap_pil, dtype=np.float32) / 255.0
    blended = alpha * heatmap_arr + (1.0 - alpha) * orig_arr
    blended = np.clip(blended, 0.0, 1.0)
    blended_uint8 = (blended * 255).astype(np.uint8)
    return Image.fromarray(blended_uint8, mode='RGB')

def pil_to_bytes(pil_image: Image.Image, fmt: str='PNG') -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return buf.getvalue()

def run_stegnet(model: StegNet, image_tensor: torch.Tensor, original_pil: Image.Image) -> Tuple[float, Image.Image]:
    with torch.inference_mode(False):
        probability, heatmap = model.generate_gradcam(image_tensor, target_class=1)
    overlay = overlay_heatmap(original_pil, heatmap, alpha=0.45, colormap='inferno')
    return (probability, overlay)

@torch.no_grad()
def run_unet(model: UNetPurifier, image_tensor: torch.Tensor) -> Image.Image:
    purified_tensor = model(image_tensor)
    out_std = purified_tensor.std().item()
    if out_std < 0.02:
        return tensor_to_pil(image_tensor)
    return tensor_to_pil(purified_tensor)

def render_sidebar(stegnet_status: str, unet_status: str) -> dict:
    with st.sidebar:
        st.markdown('## 🛡️ Stego-GAN')
        st.caption('Adversarial Steganography Hunter')
        st.divider()
        st.markdown('### Model Status')
        st.markdown(f'**StegNet Detector**  \n{stegnet_status}')
        st.markdown(f'**U-Net Purifier**   \n{unet_status}')
        st.divider()
        st.markdown('### Settings')
        heatmap_alpha = st.slider('Heatmap Opacity', min_value=0.1, max_value=0.9, value=0.45, step=0.05, help='Controls how strongly the Grad-CAM heatmap overlays the image.')
        colormap = st.selectbox('Heatmap Colour Map', options=['inferno', 'jet', 'hot', 'RdYlGn_r', 'plasma'], index=0, help='Matplotlib colourmap applied to the Grad-CAM output.')
        detection_threshold = st.slider('Detection Threshold', min_value=0.1, max_value=0.9, value=0.5, step=0.05, help='Probability above this value is classified as STEGO.')
        st.divider()
        st.markdown('### About')
        st.markdown("\n        <div class='info-box'>\n        <b>StegNet</b>: CNN classifier trained to detect\n        LSB steganographic payloads.<br><br>\n        <b>Grad-CAM</b>: Shows <em>where</em> in the image the\n        network suspects hidden data.<br><br>\n        <b>U-Net Purifier</b>: Encoder-decoder that strips\n        the embedding noise while preserving visual quality.\n        </div>\n        ", unsafe_allow_html=True)
    return {'heatmap_alpha': heatmap_alpha, 'colormap': colormap, 'detection_threshold': detection_threshold}

def render_detector(stegnet_model, stegnet_status, unet_model, unet_status, settings) -> None:
    st.markdown("<p class='hero-title'>🔍 Stego-GAN: Adversarial Steganography Hunter</p><p class='hero-sub'>Upload an image to detect hidden payloads with Grad-CAM forensics and sanitize it with neural purification.</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader('Upload an image to analyse', type=['jpg', 'jpeg', 'png', 'bmp', 'webp'], help='Supports JPEG, PNG, BMP, WebP. Max size depends on your browser.')
    if uploaded_file is None:
        st.markdown('---')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("\n            <div class='metric-card'>\n                <div style='font-size:2rem'>🧠</div>\n                <div class='metric-label'>CNN Detector</div>\n                <div style='color:#94a3b8;font-size:0.8rem;margin-top:0.5rem'>\n                    4-block conv network with Grad-CAM localisation\n                </div>\n            </div>\n            ", unsafe_allow_html=True)
        with col2:
            st.markdown("\n            <div class='metric-card'>\n                <div style='font-size:2rem'>🗺️</div>\n                <div class='metric-label'>Grad-CAM Heatmap</div>\n                <div style='color:#94a3b8;font-size:0.8rem;margin-top:0.5rem'>\n                    Highlights exactly where the model suspects hidden data\n                </div>\n            </div>\n            ", unsafe_allow_html=True)
        with col3:
            st.markdown("\n            <div class='metric-card'>\n                <div style='font-size:2rem'>✨</div>\n                <div class='metric-label'>U-Net Purifier</div>\n                <div style='color:#94a3b8;font-size:0.8rem;margin-top:0.5rem'>\n                    Encoder-decoder strips embedding noise, restores texture\n                </div>\n            </div>\n            ", unsafe_allow_html=True)
        st.info('👆 Upload an image above to begin steganalysis.')
        return
    original_pil = Image.open(uploaded_file).convert('RGB')
    image_tensor = preprocess_image(original_pil, size=IMAGE_SIZE)
    image_tensor_detected = preprocess_image_for_detection(original_pil, size=IMAGE_SIZE)
    st.markdown('---')
    st.markdown("<p class='section-header'>① Detection & Forensic Analysis</p>", unsafe_allow_html=True)
    with st.spinner('Running StegNet + Grad-CAM…'):
        probability, gradcam_overlay = run_stegnet(stegnet_model, image_tensor_detected, original_pil)
        _, heatmap_arr = stegnet_model.generate_gradcam(image_tensor_detected, target_class=1)
        gradcam_overlay = overlay_heatmap(original_pil, heatmap_arr, alpha=settings['heatmap_alpha'], colormap=settings['colormap'])
    threshold = settings['detection_threshold']
    is_stego = probability >= threshold
    if is_stego:
        verdict_html = f"<span class='badge-stego'>⚠ STEGO DETECTED</span>"
        verdict_color = '#ef4444'
    else:
        verdict_html = f"<span class='badge-clean'>✅ CLEAN</span>"
        verdict_color = '#22c55e'
    col_img, col_heat, col_meta = st.columns([2, 2, 1.2])
    with col_img:
        st.markdown('**Original Image**')
        st.image(original_pil, use_container_width=True)
    with col_heat:
        st.markdown('**Grad-CAM Heatmap Overlay**')
        st.image(gradcam_overlay, use_container_width=True)
        st.caption('🔴 Hot regions = where the model suspects embedded payload.')
    with col_meta:
        st.markdown(f'**Stego Probability**')
        prob_pct = int(probability * 100)
        st.progress(probability)
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{verdict_color}'>{prob_pct}%</div><div class='metric-label'>Stego Confidence</div></div>", unsafe_allow_html=True)
        st.markdown(f'**Verdict:** {verdict_html}', unsafe_allow_html=True)
        st.markdown(f'*Threshold: {threshold:.0%}*')
        st.divider()
        w, h = original_pil.size
        st.markdown('**Image Info**')
        st.markdown(f'- **Size:** {w} × {h} px')
        st.markdown(f'- **Mode:** {original_pil.mode}')
        total_px = w * h
        st.markdown(f'- **Pixels:** {total_px:,}')
    st.markdown('---')
    st.markdown("<p class='section-header'>② Neural Purification (U-Net)</p>", unsafe_allow_html=True)
    sanitize_col, _ = st.columns([1, 3])
    with sanitize_col:
        sanitize_clicked = st.button('🧹 Sanitize Image', type='primary', use_container_width=True, help='Run the U-Net purifier to strip any hidden payload while preserving visual quality.')
    if sanitize_clicked:
        with st.spinner('Running U-Net Purifier — removing embedding noise…'):
            purified_pil = run_unet(unet_model, image_tensor)
        orig_256 = original_pil.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        purif_256 = purified_pil.convert('RGB')
        diff_check = np.abs(np.array(orig_256, dtype=np.float32) - np.array(purif_256, dtype=np.float32)).mean()
        if diff_check < 0.5:
            st.error('🚫 **U-Net output was blank — your checkpoint was likely trained with the SRM filter enabled (old bug).** Please re-train the U-Net using the updated `kaggel_train_unet.ipynb` (SRM filter removed from U-Net dataset) and replace `checkpoints/unet_best.pth`. The original image is shown below.')
        else:
            st.success('Purification complete! Compare the images below.')
        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown('**Before (Input)**')
            st.image(original_pil, use_container_width=True)
        with col_after:
            st.markdown('**After (Purified)**')
            purified_display = purified_pil.resize(original_pil.size, Image.LANCZOS)
            st.image(purified_display, use_container_width=True)
        st.markdown('**Difference Map** *(amplified ×5 for visibility)*')
        orig_arr = np.array(original_pil.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB'), dtype=np.float32) / 255.0
        purified_arr = np.array(purified_pil.convert('RGB'), dtype=np.float32) / 255.0
        if orig_arr.shape != purified_arr.shape:
            purified_arr = np.array(purified_pil.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB'), dtype=np.float32) / 255.0
        diff = np.abs(orig_arr - purified_arr)
        diff_amplified = np.clip(diff * 5.0, 0.0, 1.0)
        diff_pil = Image.fromarray((diff_amplified * 255).astype(np.uint8), mode='RGB')
        diff_col, info_col = st.columns([2, 2])
        with diff_col:
            st.image(diff_pil, use_container_width=True)
            st.caption('Bright areas = where the purifier made changes. In a trained model, these align with the Grad-CAM heatmap.')
        with info_col:
            mean_diff = float(diff.mean()) * 255.0
            max_diff = float(diff.max()) * 255.0
            st.markdown('**Purification Statistics**')
            st.metric('Mean Pixel Δ', f'{mean_diff:.3f}')
            st.metric('Max Pixel Δ', f'{max_diff:.1f} / 255')
            st.metric('% Pixels Changed', f'{100 * (diff.sum(axis=2) > 0.002).mean():.1f}%')
        st.markdown('---')
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            img_bytes = pil_to_bytes(purified_display, fmt='PNG')
            st.download_button(label='⬇ Download Purified Image', data=img_bytes, file_name=f"purified_{uploaded_file.name.rsplit('.', 1)[0]}.png", mime='image/png', use_container_width=True)
    elif is_stego:
        st.warning('⚠️ **Stego content detected!** Click **Sanitize Image** above to strip the hidden payload from this image.')
    else:
        st.info('ℹ️ The detector did not flag this image as stego. You can still run the purifier as a precaution.')
    st.markdown('---')
    st.caption('Stego-GAN | StegNet CNN + Grad-CAM + U-Net Purifier | PyTorch · Streamlit')

def render_admin_panel():
    st.markdown("<p class='hero-title'>⚙️ Admin Dashboard</p>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>System Management & Telemetry</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown('### 📊 Usage Logs & Reports')
    st.markdown('*(Mock Data represented for UI purposes)*')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><div class='metric-value'>1,245</div><div class='metric-label'>Total Images Scanned</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='metric-value' style='color:#ef4444'>342</div><div class='metric-label'>Stego Detected</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='metric-value' style='color:#22c55e'>298</div><div class='metric-label'>Sanitized Successfully</div></div>", unsafe_allow_html=True)
    st.markdown('**Recent Activity**')
    logs = [{'ts': (datetime.datetime.now() - datetime.timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'), 'user': 'user123', 'action': 'Uploaded img_src_01.png', 'status': 'CLEAN'}, {'ts': (datetime.datetime.now() - datetime.timedelta(minutes=22)).strftime('%Y-%m-%d %H:%M:%S'), 'user': 'admin123', 'action': 'Uploaded payload_hidden.jpg', 'status': 'STEGO'}, {'ts': (datetime.datetime.now() - datetime.timedelta(minutes=23)).strftime('%Y-%m-%d %H:%M:%S'), 'user': 'admin123', 'action': 'Sanitized payload_hidden.jpg', 'status': 'CLEANED'}]
    st.table(logs)
    st.divider()
    st.markdown('### 🔄 Model Updates')
    st.info('Current active models in production environment.')
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown('**StegNet (Detector)**')
        st.code('Version: v2.1.0\\nLast updated: 2026-02-15\\nAccuracy: 94.2%')
        st.button('Check for StegNet Updates')
    with m_col2:
        st.markdown('**U-Net (Purifier)**')
        st.code('Version: v1.8.4\\nLast updated: 2026-02-10\\nPSNR: 32.5 dB')
        st.button('Check for U-Net Updates')

def render_login():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("<p class='hero-title'>🔒 Stego-GAN Login</p>", unsafe_allow_html=True)
        st.markdown("<p class='hero-sub'>Please authenticate to access the system.</p>", unsafe_allow_html=True)
        with st.form('login_form'):
            username = st.text_input('Username')
            password = st.text_input('Password', type='password')
            submitted = st.form_submit_button('Login')
            if submitted:
                if username in MOCK_USERS and MOCK_USERS[username]['password'] == password:
                    st.session_state.logged_in = True
                    st.session_state.role = MOCK_USERS[username]['role']
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error('Invalid username or password')

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.username = None
    if not st.session_state.logged_in:
        render_login()
        return
    stegnet_model, stegnet_status = load_stegnet()
    unet_model, unet_status = load_unet()
    with st.sidebar:
        st.markdown(f'**Logged in as:** `{st.session_state.username}` ({st.session_state.role})')
        if st.button('Logout', use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.session_state.username = None
            st.rerun()
        st.divider()
    if st.session_state.role == 'admin':
        tab1, tab2 = st.tabs(['🔍 Detector', '⚙️ Admin Panel'])
        with tab1:
            settings = render_sidebar(stegnet_status, unet_status)
            render_detector(stegnet_model, stegnet_status, unet_model, unet_status, settings)
        with tab2:
            render_admin_panel()
    else:
        settings = render_sidebar(stegnet_status, unet_status)
        render_detector(stegnet_model, stegnet_status, unet_model, unet_status, settings)
if __name__ == '__main__':
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        main()
    else:
        import sys
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
