# Stego-GAN: Adversarial Steganography Hunter 🔍

Stego-GAN is a deep learning-based security pipeline designed to **detect** and **neutralize** hidden steganographic payloads in digital images. It specifically targets LSB (Least Significant Bit) injection techniques, providing both forensic localization via Grad-CAM and neural purification via a U-Net architecture.

---

## 🚀 Key Features

- **Dynamic LSB Injection:** A robust data pipeline that generates stego images on-the-fly during training, ensuring the models never see the same payload twice.
- **StegNet Detector:** A 4-block CNN classifier that identifies steganographic content with high precision.
- **Grad-CAM Forensics:** Visualizes suspicion heatmaps to show exactly *where* the model believes data is hidden.
- **U-Net Purifier:** An encoder-decoder network with skip connections that "cleans" images, stripping away embedding noise while preserving textures.
- **Triple-Objective Loss:** The purifier is trained using a composite of **L1 Reconstruction**, **Perceptual (VGG-16)**, and **SSIM** losses for maximum visual fidelity.
- **Interactive Dashboard:** A premium Streamlit frontend for real-time analysis and image sanitization.

---

## 📂 File Structure

```text
Stego-Gan/
├── app.py                # Streamlit dashboard (UI)
├── train.py              # Main training script for StegNet and U-Net
├── requirements.txt      # Project dependencies
├── data/
│   └── injector.py       # LSB injection logic and PyTorch Dataset
├── models/
│   ├── stegnet.py        # StegNet architecture + Grad-CAM implementation
│   └── unet_purifier.py  # U-Net architecture for image cleaning
└── utils/
    └── losses.py         # Custom composite loss (L1 + Perceptual + SSIM)
```

---

## 🛠️ How it Works

### 1. The Detection Core (StegNet)
The detector is a deep CNN that processes images at $256 \times 256$ resolution. It uses **Global Average Pooling (GAP)** and a dropout-stabilized classifier head. 
- **Explainability:** It uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to compute the importance of feature maps in the final convolutional layer, producing a heatmap that highlights suspicious pixel regions.

### 2. The Purification Engine (U-Net)
The purifier uses a **U-Net** architecture. Unlike standard autoencoders, its **skip connections** wire fine-grained textural details from the encoder directly to the decoder.
- **What it does:** It treats steganographic bits as high-frequency noise and learns to map the "stego" distribution back to the "clean" distribution.
- **Loss Function:** 
  - **L1 Loss:** Ensures pixel-level accuracy.
  - **Perceptual Loss:** Matches features from a frozen VGG-16 network to maintain semantic realism.
  - **SSIM Loss:** Preserves structural integrity and contrast.

### 3. Training Strategy
The project uses a "Dynamic Injection" approach. Instead of keeping a static dataset of stego images, the `StegoDataset` injects random binary payloads into clean images in RAM during every training step. This forces the models to learn the *concept* of steganography rather than memorizing specific patterns.

---

## 🚦 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support for faster training)

### Setup
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your clean carrier images (JPG/PNG) in `data/raw/`.

### Training
Train both models sequentially:
```bash
python train.py --data_dir data/raw --epochs 30
```

### Running the App
Launch the interactive dashboard:
```bash
streamlit run app.py
```

---

## 🛡️ Security Disclaimer
This tool is designed for research and educational purposes in steganography and digital forensics. It is intended to help security researchers understand and defend against covert communication channels.
