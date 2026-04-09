import io
import sys
import os
import struct
import numpy as np
from PIL import Image
import streamlit as st
st.set_page_config(page_title='Stego-GAN: Injection Lab', page_icon='💉', layout='wide', initial_sidebar_state='expanded')
st.markdown('\n<style>\n    @import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap\');\n\n    html, body, [class*="css"] {\n        font-family: \'Inter\', sans-serif;\n    }\n\n    /* Main container */\n    .main { background-color: #080d14; }\n\n    /* Hero */\n    .hero-title {\n        font-size: 2.6rem;\n        font-weight: 800;\n        background: linear-gradient(135deg, #00f5a0, #00d9f5, #7b2ff7);\n        -webkit-background-clip: text;\n        -webkit-text-fill-color: transparent;\n        margin-bottom: 0.3rem;\n        letter-spacing: -0.5px;\n    }\n    .hero-sub {\n        color: #657b96;\n        font-size: 1.05rem;\n        margin-bottom: 1.5rem;\n    }\n\n    /* Metric cards */\n    .metric-card {\n        background: linear-gradient(135deg, #111827, #1a2235);\n        border: 1px solid #1e2d45;\n        border-radius: 14px;\n        padding: 1.3rem 1.5rem;\n        text-align: center;\n        margin-bottom: 1rem;\n        box-shadow: 0 4px 24px rgba(0,0,0,0.3);\n    }\n    .metric-value {\n        font-size: 2.4rem;\n        font-weight: 700;\n        font-family: \'JetBrains Mono\', monospace;\n    }\n    .metric-label {\n        color: #657b96;\n        font-size: 0.82rem;\n        margin-top: 0.3rem;\n        text-transform: uppercase;\n        letter-spacing: 0.5px;\n    }\n\n    /* Section headers */\n    .section-header {\n        font-size: 1.15rem;\n        font-weight: 700;\n        color: #e2e8f0;\n        border-left: 4px solid #00f5a0;\n        padding-left: 0.8rem;\n        margin: 1.2rem 0 0.6rem 0;\n        letter-spacing: 0.2px;\n    }\n\n    /* Technique badge */\n    .technique-badge {\n        display: inline-block;\n        background: linear-gradient(135deg, #0d1f2d, #13283d);\n        border: 1px solid #00d9f5;\n        border-radius: 6px;\n        padding: 0.2rem 0.7rem;\n        color: #00d9f5;\n        font-family: \'JetBrains Mono\', monospace;\n        font-size: 0.8rem;\n        font-weight: 600;\n    }\n\n    /* Info box */\n    .info-box {\n        background: #0e1724;\n        border: 1px solid #1e2d45;\n        border-left: 3px solid #00f5a0;\n        border-radius: 8px;\n        padding: 1rem 1.2rem;\n        color: #8ba3bf;\n        font-size: 0.875rem;\n        line-height: 1.6;\n    }\n\n    /* Warning / danger */\n    .warn-box {\n        background: #1a1200;\n        border: 1px solid #f59e0b;\n        border-left: 3px solid #f59e0b;\n        border-radius: 8px;\n        padding: 0.8rem 1.2rem;\n        color: #fbbf24;\n        font-size: 0.875rem;\n    }\n\n    /* Code mono */\n    code {\n        font-family: \'JetBrains Mono\', monospace !important;\n        font-size: 0.8rem;\n        background: #0e1724;\n        padding: 0.1rem 0.3rem;\n        border-radius: 4px;\n        color: #00f5a0;\n    }\n\n    /* Streamlit overrides */\n    div[data-testid="stMetric"] {\n        background: #111827;\n        border-radius: 10px;\n        padding: 1rem;\n        border: 1px solid #1e2d45;\n    }\n    .stButton > button {\n        border-radius: 8px;\n        font-weight: 600;\n        letter-spacing: 0.3px;\n        transition: all 0.2s ease;\n    }\n    .stButton > button:hover {\n        transform: translateY(-1px);\n        box-shadow: 0 4px 16px rgba(0, 245, 160, 0.2);\n    }\n    div[data-testid="stFileUploader"] {\n        border: 2px dashed #1e2d45;\n        border-radius: 12px;\n        padding: 1rem;\n        background: #0b1120;\n    }\n</style>\n', unsafe_allow_html=True)

def lsb_inject_random(image: Image.Image, payload_fraction: float) -> Image.Image:
    img_rgb = image.convert('RGB')
    arr = np.array(img_rgb, dtype=np.uint8).copy()
    H, W, C = arr.shape
    total = H * W * C
    n_bits = int(total * payload_fraction)
    payload = np.random.randint(0, 2, size=n_bits, dtype=np.uint8)
    positions = np.random.choice(total, size=n_bits, replace=False)
    flat = arr.reshape(-1)
    flat[positions] = flat[positions] & np.uint8(254) | payload
    return Image.fromarray(flat.reshape(H, W, C), mode='RGB')

def lsb_inject_text(image: Image.Image, message: str) -> tuple[Image.Image, int]:
    img_rgb = image.convert('RGB')
    arr = np.array(img_rgb, dtype=np.uint8).copy()
    H, W, C = arr.shape
    msg_bytes = message.encode('utf-8')
    length_bytes = struct.pack('>I', len(msg_bytes))
    all_bytes = length_bytes + msg_bytes
    bits = np.unpackbits(np.frombuffer(all_bytes, dtype=np.uint8))
    n_bits = len(bits)
    total_capacity = H * W * C
    if n_bits > total_capacity:
        raise ValueError(f'Message too long: needs {n_bits} bits, image has {total_capacity} bits capacity.')
    flat = arr.reshape(-1)
    flat[:n_bits] = flat[:n_bits] & np.uint8(254) | bits.astype(np.uint8)
    return (Image.fromarray(flat.reshape(H, W, C), mode='RGB'), n_bits)

def lsb_spread_spectrum(image: Image.Image, payload_fraction: float, seed: int=42) -> Image.Image:
    img_rgb = image.convert('RGB')
    arr = np.array(img_rgb, dtype=np.uint8).copy()
    H, W, C = arr.shape
    total = H * W * C
    n_bits = int(total * payload_fraction)
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
    positions = rng.choice(total, size=n_bits, replace=False)
    flat = arr.reshape(-1)
    flat[positions] = flat[positions] & np.uint8(254) | payload
    return Image.fromarray(flat.reshape(H, W, C), mode='RGB')

def dct_like_inject(image: Image.Image, strength: float=8.0, payload_fraction: float=0.3) -> Image.Image:
    try:
        from scipy.fft import dct, idct
    except ImportError:

        def dct(x, norm=None):
            N = len(x)
            v = np.zeros(2 * N)
            v[:N] = x
            v[N:] = x[::-1]
            V = np.fft.fft(v)
            k = np.arange(N)
            W = np.exp(-1j * np.pi * k / (2 * N))
            return np.real(W * V[:N])

        def idct(X, norm=None):
            N = len(X)
            k = np.arange(N)
            W = np.exp(1j * np.pi * k / (2 * N))
            V = W * X
            v = np.real(np.fft.ifft(np.concatenate([V, np.conj(V[-1:0:-1])])))
            return v[:N]
    img_rgb = image.convert('RGB')
    img_ycbcr = img_rgb.convert('YCbCr')
    arr_ycbcr = np.array(img_ycbcr, dtype=np.float32)
    Y = arr_ycbcr[:, :, 0]
    H, W = Y.shape
    pH = (H + 7) // 8 * 8
    pW = (W + 7) // 8 * 8
    Yp = np.pad(Y, ((0, pH - H), (0, pW - W)), mode='edge')
    n_blocks_h = pH // 8
    n_blocks_w = pW // 8
    total_blocks = n_blocks_h * n_blocks_w
    n_embed = int(total_blocks * payload_fraction)
    rng = np.random.default_rng(None)
    block_indices = rng.choice(total_blocks, size=n_embed, replace=False)
    mid_freq_positions = [10, 11, 12, 13, 14, 15, 17, 18, 20]
    for bi in block_indices:
        r = bi // n_blocks_w * 8
        c = bi % n_blocks_w * 8
        block = Yp[r:r + 8, c:c + 8]
        dct_block = np.zeros_like(block)
        for row in range(8):
            dct_block[row, :] = dct(block[row, :], norm='ortho')
        for col in range(8):
            dct_block[:, col] = dct(dct_block[:, col], norm='ortho')
        flat_dct = dct_block.reshape(-1)
        payload_bit = rng.integers(0, 2)
        for pos in mid_freq_positions[:3]:
            coeff = flat_dct[pos]
            quantised = np.round(coeff / strength)
            if payload_bit == 1:
                quantised = quantised if quantised % 2 == 1 else quantised + 1
            else:
                quantised = quantised if quantised % 2 == 0 else quantised + 1
            flat_dct[pos] = quantised * strength
        dct_block = flat_dct.reshape(8, 8)
        idct_block = np.zeros_like(dct_block)
        for col in range(8):
            idct_block[:, col] = idct(dct_block[:, col], norm='ortho')
        for row in range(8):
            idct_block[row, :] = idct(idct_block[row, :], norm='ortho')
        Yp[r:r + 8, c:c + 8] = idct_block
    Y_new = np.clip(Yp[:H, :W], 0, 255)
    arr_ycbcr[:, :, 0] = Y_new
    stego_ycbcr = Image.fromarray(arr_ycbcr.astype(np.uint8), mode='YCbCr')
    return stego_ycbcr.convert('RGB')

def compute_diff_map(original: Image.Image, stego: Image.Image, amplify: float=10.0) -> Image.Image:
    size = (min(original.width, stego.width), min(original.height, stego.height))
    orig_arr = np.array(original.convert('RGB').resize(size), dtype=np.float32) / 255.0
    stego_arr = np.array(stego.convert('RGB').resize(size), dtype=np.float32) / 255.0
    diff = np.abs(orig_arr - stego_arr)
    diff_amplified = np.clip(diff * amplify, 0.0, 1.0)
    return Image.fromarray((diff_amplified * 255).astype(np.uint8), mode='RGB')

def pil_to_bytes(img: Image.Image, fmt: str='PNG') -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def payload_capacity_text(image: Image.Image) -> str:
    W, H = image.size
    total_bits = H * W * 3 - 32
    max_bytes = total_bits // 8
    if max_bytes > 1024 * 1024:
        return f'~{max_bytes / 1024 / 1024:.1f} MB'
    elif max_bytes > 1024:
        return f'~{max_bytes / 1024:.1f} KB'
    return f'~{max_bytes} bytes'

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown('## 💉 Injection Lab')
        st.caption('Stego-GAN Attack Tool')
        st.divider()
        st.markdown('### Technique')
        technique = st.selectbox('Select Injection Method', options=['LSB Random (standard)', 'LSB Spread Spectrum (keyed)', 'LSB Text Embed (message)', 'DCT-Like (mid-freq, JPEG-style)'], index=0, help='Each technique embeds data differently — some are harder for the detector to catch.')
        st.divider()
        st.markdown('### Parameters')
        params = {}
        tech_key = technique.split(' ')[0]
        if technique == 'LSB Random (standard)':
            params['payload_fraction'] = st.slider('Payload Density', 0.05, 1.0, 0.5, 0.05, help='Fraction of pixel channel values to inject into. 0.5 = half of all LSBs replaced.')
        elif technique == 'LSB Spread Spectrum (keyed)':
            params['payload_fraction'] = st.slider('Payload Density', 0.05, 1.0, 0.4, 0.05)
            params['seed'] = st.number_input('Embedding Seed (key)', min_value=0, max_value=999999, value=42, help="Change the seed to simulate different 'keys'. The receiver must know this seed to extract the payload.")
        elif technique == 'LSB Text Embed (message)':
            params['message'] = st.text_area('Secret Message', value='This image contains a hidden message injected by Stego-GAN Injection Lab.', height=120, help='Your text will be encoded as UTF-8 and embedded via LSB.')
        elif technique == 'DCT-Like (mid-freq, JPEG-style)':
            params['payload_fraction'] = st.slider('Block Coverage', 0.05, 0.8, 0.25, 0.05, help='Fraction of 8×8 blocks to embed into.')
            params['strength'] = st.slider('Coefficient Perturbation', 2.0, 20.0, 8.0, 1.0, help='Larger = more detectable but more robust. Smaller = more fragile but stealthier.')
        st.divider()
        st.markdown('### Amplification')
        params['diff_amplify'] = st.slider('Difference Map Amplify ×', 1.0, 30.0, 10.0, 1.0, help='Multiply the difference image for visibility.')
        st.divider()
        st.markdown("\n        <div class='info-box'>\n        <b>How to use:</b><br><br>\n        1. Upload a clean image<br>\n        2. Choose an injection technique<br>\n        3. Configure parameters<br>\n        4. Click <b>Inject Steganography</b><br>\n        5. Download the stego image<br>\n        6. Open <code>app.py</code> (detector) and upload it there to test detection!\n        </div>\n        ", unsafe_allow_html=True)
    return {'technique': technique, **params}

def main():
    st.markdown("<p class='hero-title'>💉 Stego-GAN: Injection Lab</p><p class='hero-sub'>Embed hidden payloads into images to test the adversarial detector in <code>app.py</code>.</p>", unsafe_allow_html=True)
    settings = render_sidebar()
    technique = settings['technique']
    col1, col2, col3, col4 = st.columns(4)
    card_data = [('🔡', 'LSB Random', 'Inject random bits into pixel LSBs — the standard benchmark attack'), ('🔐', 'Spread Spectrum', 'Keyed pseudo-random LSB order — simulates real-world keyed embedding'), ('✉️', 'Text Embed', 'Hide a real UTF-8 message inside the image channels'), ('📐', 'DCT-Like', 'Embed in 8×8 block frequency coefficients — JPEG-style covert channel')]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], card_data):
        with col:
            st.markdown(f"\n            <div class='metric-card'>\n                <div style='font-size:1.8rem'>{icon}</div>\n                <div class='metric-label' style='font-weight:700;color:#c5d1e0;font-size:0.85rem;margin-top:0.4rem'>{title}</div>\n                <div style='color:#4a6070;font-size:0.75rem;margin-top:0.4rem;line-height:1.4'>{desc}</div>\n            </div>\n            ", unsafe_allow_html=True)
    st.markdown('---')
    st.markdown("<p class='section-header'>① Upload a Clean Image</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader('Choose an image to inject steganography into', type=['jpg', 'jpeg', 'png', 'bmp', 'webp'], help='Any common image format. Larger images can carry more payload.')
    if uploaded is None:
        st.info('👆 Upload an image to get started. Then configure the injection parameters in the sidebar.')
        st.markdown("\n        <div class='warn-box'>\n        ⚡ <b>Tip:</b> After generating a stego image here, download it and\n        upload it to <code>streamlit run app.py</code> (the detector) to see\n        if StegNet catches it!\n        </div>\n        ", unsafe_allow_html=True)
        return
    original_pil = Image.open(uploaded).convert('RGB')
    W_orig, H_orig = original_pil.size
    st.markdown('---')
    st.markdown("<p class='section-header'>② Configure &amp; Inject</p>", unsafe_allow_html=True)
    st.markdown(f"Selected technique: <span class='technique-badge'>{technique}</span>", unsafe_allow_html=True)
    cap_text = payload_capacity_text(original_pil)
    st.markdown(f"\n    <div class='info-box' style='margin-top:0.7rem;margin-bottom:0.7rem'>\n    📷 Image: <code>{W_orig} × {H_orig} px</code> &nbsp;|&nbsp;\n    Channels: <code>RGB (3)</code> &nbsp;|&nbsp;\n    LSB capacity: <code>{cap_text}</code>\n    </div>\n    ", unsafe_allow_html=True)
    inject_col, _ = st.columns([1, 3])
    with inject_col:
        inject_clicked = st.button('💉 Inject Steganography', type='primary', use_container_width=True, help='Apply the selected embedding technique to the uploaded image.')
    if not inject_clicked:
        prev_col, info_col = st.columns([2, 1])
        with prev_col:
            st.markdown('**Uploaded Image (clean)**')
            st.image(original_pil, use_container_width=True)
        with info_col:
            st.markdown('**Image Details**')
            st.metric('Width', f'{W_orig} px')
            st.metric('Height', f'{H_orig} px')
            total_px = W_orig * H_orig
            st.metric('Total Pixels', f'{total_px:,}')
            st.metric('LSB Capacity', cap_text)
        return
    error_msg = None
    stego_pil = None
    bits_embedded = None
    with st.spinner(f"Injecting payload using '{technique}'…"):
        try:
            if technique == 'LSB Random (standard)':
                stego_pil = lsb_inject_random(original_pil, payload_fraction=settings['payload_fraction'])
            elif technique == 'LSB Spread Spectrum (keyed)':
                stego_pil = lsb_spread_spectrum(original_pil, payload_fraction=settings['payload_fraction'], seed=int(settings['seed']))
            elif technique == 'LSB Text Embed (message)':
                message = settings.get('message', '').strip()
                if not message:
                    st.error('⚠️ Please enter a message to embed in the sidebar.')
                    return
                stego_pil, bits_embedded = lsb_inject_text(original_pil, message)
            elif technique == 'DCT-Like (mid-freq, JPEG-style)':
                stego_pil = dct_like_inject(original_pil, strength=settings['strength'], payload_fraction=settings['payload_fraction'])
        except ValueError as e:
            error_msg = str(e)
    if error_msg:
        st.error(f'❌ Injection failed: {error_msg}')
        return
    st.success('✅ Steganography injected successfully!')
    st.markdown('---')
    st.markdown("<p class='section-header'>③ Results — Original vs Stego</p>", unsafe_allow_html=True)
    col_orig, col_stego = st.columns(2)
    with col_orig:
        st.markdown('**🟢 Original (Clean)**')
        st.image(original_pil, use_container_width=True)
        st.caption('No hidden data — safe to share.')
    with col_stego:
        st.markdown('**🔴 Stego Image (Payload Embedded)**')
        st.image(stego_pil, use_container_width=True)
        st.caption('Visually identical — payload hidden in pixel LSBs or DCT coefficients.')
    st.markdown('---')
    st.markdown("<p class='section-header'>④ Difference Map (Amplified)</p>", unsafe_allow_html=True)
    diff_pil = compute_diff_map(original_pil, stego_pil, amplify=settings['diff_amplify'])
    diff_col, stats_col = st.columns([2, 1])
    with diff_col:
        st.image(diff_pil, use_container_width=True)
        st.caption(f"Pixel-level difference amplified ×{settings['diff_amplify']:.0f}. Bright = modified pixels. For LSB this is nearly invisible at ×1.")
    with stats_col:
        orig_arr = np.array(original_pil.convert('RGB'), dtype=np.float32)
        stego_arr = np.array(stego_pil.convert('RGB').resize(original_pil.size), dtype=np.float32)
        diff_arr = np.abs(orig_arr - stego_arr)
        mean_delta = float(diff_arr.mean())
        max_delta = float(diff_arr.max())
        pct_changed = float((diff_arr.max(axis=2) > 0.5).mean()) * 100
        mse = float((diff_arr ** 2).mean())
        psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
        st.markdown('**📊 Injection Statistics**')
        st.metric('Mean Pixel Δ', f'{mean_delta:.4f}')
        st.metric('Max Pixel Δ', f'{max_delta:.1f} / 255')
        st.metric('% Pixels Modified', f'{pct_changed:.2f}%')
        psnr_str = f'{psnr:.1f} dB' if psnr != float('inf') else '∞ (no change)'
        st.metric('PSNR (quality)', psnr_str)
        if bits_embedded is not None:
            n_chars = bits_embedded // 8
            st.metric('Bits Embedded', f'{bits_embedded:,}')
            st.metric('Chars Embedded', f'{n_chars:,}')
    st.markdown('---')
    st.markdown("<p class='section-header'>⑤ Payload Summary</p>", unsafe_allow_html=True)
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.markdown(f"\n        <div class='metric-card'>\n            <div class='metric-value' style='color:#00f5a0;font-size:1.8rem'>{technique.split(' ')[0]}</div>\n            <div class='metric-label'>Method</div>\n        </div>\n        ", unsafe_allow_html=True)
    with summary_cols[1]:
        psnr_display = f'{psnr:.1f}' if psnr != float('inf') else '∞'
        color = '#22c55e' if psnr > 45 else '#f59e0b' if psnr > 35 else '#ef4444'
        st.markdown(f"\n        <div class='metric-card'>\n            <div class='metric-value' style='color:{color}'>{psnr_display} dB</div>\n            <div class='metric-label'>PSNR (higher = stealthier)</div>\n        </div>\n        ", unsafe_allow_html=True)
    with summary_cols[2]:
        st.markdown(f"\n        <div class='metric-card'>\n            <div class='metric-value' style='color:#00d9f5'>{pct_changed:.1f}%</div>\n            <div class='metric-label'>Pixels Modified</div>\n        </div>\n        ", unsafe_allow_html=True)
    if psnr > 50:
        level_label = '🟢 Very Hard to Detect'
        level_color = '#22c55e'
        level_desc = 'Payload density is very low — StegNet will likely classify this as CLEAN.'
    elif psnr > 40:
        level_label = '🟡 Moderate Detectability'
        level_color = '#f59e0b'
        level_desc = 'Reasonable payload size — StegNet will show elevated suspicion (~40–70%).'
    else:
        level_label = '🔴 Easily Detected'
        level_color = '#ef4444'
        level_desc = 'High payload density — StegNet will likely flag this as STEGO.'
    st.markdown(f"\n    <div style='background:#0e1724;border:1px solid {level_color};border-left:4px solid {level_color};\n                border-radius:8px;padding:1rem 1.2rem;margin-top:0.5rem'>\n        <span style='color:{level_color};font-weight:700;font-size:1.05rem'>{level_label}</span><br>\n        <span style='color:#8ba3bf;font-size:0.875rem'>{level_desc}</span>\n    </div>\n    ", unsafe_allow_html=True)
    st.markdown('---')
    st.markdown("<p class='section-header'>⑥ Download &amp; Test in Detector</p>", unsafe_allow_html=True)
    dl_col, info_col = st.columns([1, 2])
    with dl_col:
        img_bytes = pil_to_bytes(stego_pil, fmt='PNG')
        base_name = uploaded.name.rsplit('.', 1)[0]
        dl_name = f'stego_{base_name}.png'
        st.download_button(label='⬇ Download Stego Image', data=img_bytes, file_name=dl_name, mime='image/png', use_container_width=True)
    with info_col:
        st.markdown(f"\n        <div class='info-box'>\n        ✅ <b>Stego image ready: <code>{dl_name}</code></b><br><br>\n        Next steps:<br>\n        1. Download the image above<br>\n        2. Open a terminal in the project folder<br>\n        3. Run: <code>streamlit run app.py</code><br>\n        4. Upload <code>{dl_name}</code> in the detector<br>\n        5. See if StegNet catches the hidden payload! 🔍\n        </div>\n        ", unsafe_allow_html=True)
    st.markdown('---')
    st.caption('Stego-GAN Injection Lab | LSB · Spread Spectrum · Text Embed · DCT-Like | PyTorch · Streamlit')
if __name__ == '__main__':
    main()
