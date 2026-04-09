import torch
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
if device.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1000000000.0:.1f} GB')
else:
    print('⚠️  No GPU detected. Go to Settings → Accelerator → GPU T4 x2')
INPUT_BASE = '/kaggle/input'
print(f'\nAttached datasets in {INPUT_BASE}:')
for name in os.listdir(INPUT_BASE):
    print(f'  → {os.path.join(INPUT_BASE, name)}')
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

def count_images(folder):
    count = 0
    for root, _, files in os.walk(folder):
        count += sum((1 for f in files if os.path.splitext(f)[1].lower() in VALID_EXTS))
    return count
print('\nImage counts per attached dataset:')
for name in os.listdir(INPUT_BASE):
    fp = os.path.join(INPUT_BASE, name)
    n = count_images(fp)
    print(f'  {name}: {n} images')
DATASET_NAME = 'your-dataset-name'
DATASET_PATH = f'/kaggle/input/{DATASET_NAME}'
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
train_dir = os.path.join(DATASET_PATH, 'train')
val_dir = os.path.join(DATASET_PATH, 'val')
assert os.path.isdir(DATASET_PATH), f'❌ Dataset not found at {DATASET_PATH}'
assert os.path.isdir(train_dir), f'❌ train/ folder not found at {train_dir}'
assert os.path.isdir(val_dir), f'❌ val/ folder not found at {val_dir}'
print(f'✅ Dataset found!')
print(f'   Train images : {count_images(train_dir)}')
print(f'   Val   images : {count_images(val_dir)}')
print(f'   Checkpoints  : {CHECKPOINT_DIR}')
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class LSBInjector:

    def inject(self, image: Image.Image, payload_fraction: float=0.5) -> Image.Image:
        image = image.convert('RGB')
        pixel_array = np.array(image, dtype=np.uint8).copy()
        H, W, C = pixel_array.shape
        total = H * W * C
        n_bits = int(total * payload_fraction)
        bits = np.random.randint(0, 2, size=n_bits, dtype=np.uint8)
        indices = np.random.choice(total, size=n_bits, replace=False)
        flat = pixel_array.reshape(-1)
        flat[indices] = flat[indices] & np.uint8(254) | bits
        return Image.fromarray(flat.reshape(H, W, C), mode='RGB')

class StegoDataset(Dataset):
    _VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, root_dir, image_size=256, payload_fraction=0.5, augment=True):
        self.payload_fraction = payload_fraction
        self.augment = augment
        self.injector = LSBInjector()
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in self._VALID_EXTS:
                    self.image_paths.append(os.path.join(root, fname))
        self.image_paths.sort()
        if not self.image_paths:
            raise FileNotFoundError(f'No images found in {root_dir!r}')
        spatial = [T.Resize((image_size, image_size))]
        if augment:
            spatial.append(T.RandomHorizontalFlip(p=0.5))
        self.spatial_transform = T.Compose(spatial)
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        clean_pil = Image.open(self.image_paths[idx]).convert('RGB')
        stego_pil = self.injector.inject(clean_pil, payload_fraction=self.payload_fraction)
        seed = random.randint(0, 2 ** 31)
        random.seed(seed)
        torch.manual_seed(seed)
        clean_t = self.spatial_transform(clean_pil)
        random.seed(seed)
        torch.manual_seed(seed)
        stego_t = self.spatial_transform(stego_pil)
        c = self.to_tensor(clean_t)
        s = self.to_tensor(stego_t)
        return {'clean': c, 'stego': s, 'label': 1}
print('✅ StegoDataset defined (pixel-space mode — no SRM)')
import torch.nn as nn

def _double_conv(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

class _EncoderStage(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = _double_conv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

class _DecoderStage(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _double_conv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))

class UNetPurifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.enc0 = _double_conv(3, 64)
        self.enc1 = _EncoderStage(64, 128)
        self.enc2 = _EncoderStage(128, 256)
        self.enc3 = _EncoderStage(256, 512)
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2, 2), _double_conv(512, 1024))
        self.dec3 = _DecoderStage(1024, 512)
        self.dec2 = _DecoderStage(512, 256)
        self.dec1 = _DecoderStage(256, 128)
        self.dec0 = _DecoderStage(128, 64)
        self.output_head = nn.Sequential(nn.Conv2d(64, 3, 1), nn.Tanh())

    def forward(self, x):
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.bottleneck(s3)
        return self.output_head(self.dec0(self.dec1(self.dec2(self.dec3(b, s3), s2), s1), s0))
print('✅ UNetPurifier defined')
import torchvision.models as tv_models
from typing import Optional

class ReconstructionLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.l1(pred, target)

class PerceptualLoss(nn.Module):
    _LAYER_IDX = {'relu1_2': 4, 'relu2_2': 9, 'relu3_3': 16}
    _LAYER_W = {'relu1_2': 1.0, 'relu2_2': 0.75, 'relu3_3': 0.5}

    def __init__(self):
        super().__init__()
        try:
            vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        except AttributeError:
            vgg = tv_models.vgg16(pretrained=True)
        feats = list(vgg.features.children())
        self.slice_names = sorted(self._LAYER_IDX.keys())
        self.slices = nn.ModuleList()
        prev = 0
        for idx in sorted(self._LAYER_IDX.values()):
            self.slices.append(nn.Sequential(*feats[prev:idx + 1]))
            prev = idx + 1
        for p in self.parameters():
            p.requires_grad = False
        self.mse = nn.MSELoss()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _pre(self, x):
        return ((x + 1.0) / 2.0 - self.mean) / self.std

    def forward(self, pred, target):
        px, tx = (self._pre(pred), self._pre(target))
        loss = torch.tensor(0.0, device=pred.device)
        for s, n in zip(self.slices, self.slice_names):
            px = s(px)
            with torch.no_grad():
                tx = s(tx)
            loss = loss + self._LAYER_W[n] * self.mse(px, tx)
        return loss

class SSIMLoss(nn.Module):

    def __init__(self, ws=11, sigma=1.5):
        super().__init__()
        self.ws, self.sigma = (ws, sigma)
        self._win: Optional[torch.Tensor] = None

    def _build(self, device):
        c = torch.arange(self.ws, dtype=torch.float32, device=device) - self.ws // 2
        g = torch.exp(-c ** 2 / (2 * self.sigma ** 2))
        g /= g.sum()
        return (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    def _ssim(self, x, y):
        B, C, H, W = x.shape
        p = self.ws // 2
        if self._win is None or self._win.device != x.device:
            self._win = self._build(x.device)
        w = self._win.expand(C, 1, self.ws, self.ws)
        mx = F.conv2d(x, w, padding=p, groups=C)
        my = F.conv2d(y, w, padding=p, groups=C)
        sx = F.conv2d(x * x, w, padding=p, groups=C) - mx ** 2
        sy = F.conv2d(y * y, w, padding=p, groups=C) - my ** 2
        sxy = F.conv2d(x * y, w, padding=p, groups=C) - mx * my
        C1, C2 = ((0.01 * 2) ** 2, (0.03 * 2) ** 2)
        return ((2 * mx * my + C1) * (2 * sxy + C2) / ((mx ** 2 + my ** 2 + C1) * (sx + sy + C2) + 1e-08)).mean()

    def forward(self, pred, target):
        return 1.0 - self._ssim(pred, target)

class UNetLoss(nn.Module):

    def __init__(self, lr=1.0, lp=0.1, ls=0.5):
        super().__init__()
        self.lr, self.lp, self.ls = (lr, lp, ls)
        self.recon = ReconstructionLoss()
        self.perc = PerceptualLoss()
        self.ssim = SSIMLoss()

    def forward(self, pred, target):
        return self.lr * self.recon(pred, target) + self.lp * self.perc(pred, target) + self.ls * self.ssim(pred, target)

    def breakdown(self, pred, target):
        l1 = self.recon(pred, target)
        lp = self.perc(pred, target)
        ls = self.ssim(pred, target)
        total = self.lr * l1 + self.lp * lp + self.ls * ls
        return {'total': total.item(), 'l1': l1.item(), 'perceptual': lp.item(), 'ssim': ls.item()}
print('✅ Loss functions defined')
import time
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
EPOCHS = 20
BATCH_SIZE = 8
LR = 0.0005
IMAGE_SIZE = 256
PAYLOAD_FRAC = 0.5
NUM_WORKERS = 2
SAVE_EVERY = 5
train_set = StegoDataset(train_dir, IMAGE_SIZE, PAYLOAD_FRAC, augment=True)
val_set = StegoDataset(val_dir, IMAGE_SIZE, PAYLOAD_FRAC, augment=False)
print(f'Train samples : {len(train_set)}')
print(f'Val samples   : {len(val_set)}')
pin = device.type == 'cuda'
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
model = UNetPurifier().to(device)
criterion = UNetLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-05)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
n_params = sum((p.numel() for p in model.parameters()))
print(f'Model parameters : {n_params:,}')
print('✅ Ready to train!')
best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}
print('=' * 65)
print('  U-Net Purifier Training — Kaggle (FIXED: no SRM in dataset)')
print(f'  Epochs:{EPOCHS}  Batch:{BATCH_SIZE}  LR:{LR}  Device:{device}')
print('=' * 65)
for epoch in range(EPOCHS):
    model.train()
    running = {'total': 0.0, 'l1': 0.0, 'perceptual': 0.0, 'ssim': 0.0}
    t0 = time.time()
    for i, batch in enumerate(train_loader):
        stego = batch['stego'].to(device)
        clean = batch['clean'].to(device)
        optimizer.zero_grad()
        purified = model(stego)
        loss = criterion(purified, clean)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        d = criterion.breakdown(purified.detach(), clean.detach())
        for k in running:
            running[k] += d[k]
        if (i + 1) % 20 == 0:
            print(f"  [Epoch {epoch + 1}/{EPOCHS}] Step {i + 1}/{len(train_loader)} | Loss:{d['total']:.4f} L1:{d['l1']:.4f} Perc:{d['perceptual']:.4f} SSIM:{d['ssim']:.4f}")
    model.eval()
    vtot = {'total': 0.0, 'l1': 0.0, 'perceptual': 0.0, 'ssim': 0.0}
    with torch.no_grad():
        for batch in val_loader:
            s = batch['stego'].to(device)
            c = batch['clean'].to(device)
            d = criterion.breakdown(model(s), c)
            for k in vtot:
                vtot[k] += d[k]
    nv = len(val_loader)
    nt = len(train_loader)
    val_loss = vtot['total'] / nv
    train_loss = running['total'] / nt
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    print(f"\n[Epoch {epoch + 1}/{EPOCHS}] Train:{train_loss:.4f} | Val:{val_loss:.4f} (L1:{vtot['l1'] / nv:.4f} Perc:{vtot['perceptual'] / nv:.4f} SSIM:{vtot['ssim'] / nv:.4f}) | Time:{time.time() - t0:.1f}s")
    scheduler.step(val_loss)
    state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_loss, 'best_val_loss': best_val_loss}
    torch.save(state, f'{CHECKPOINT_DIR}/unet_latest.pth')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(state, f'{CHECKPOINT_DIR}/unet_best.pth')
        print(f'  ✅ New best! Val Loss: {best_val_loss:.4f} → unet_best.pth saved')
    if (epoch + 1) % SAVE_EVERY == 0:
        torch.save(state, f'{CHECKPOINT_DIR}/unet_epoch{epoch + 1}.pth')
        print(f'  💾 Periodic checkpoint saved (epoch {epoch + 1})')
    print()
print(f'🎉 Training complete! Best Val Loss: {best_val_loss:.4f}')
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
plt.plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('U-Net Purifier — Training Curve (Fixed: pixel-space mode)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{CHECKPOINT_DIR}/training_curve.png', dpi=150)
plt.show()
print('Plot saved to /kaggle/working/checkpoints/training_curve.png')
import zipfile
ZIP_PATH = '/kaggle/working/unet_checkpoints.zip'
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(CHECKPOINT_DIR):
        fpath = os.path.join(CHECKPOINT_DIR, fname)
        zf.write(fpath, arcname=fname)
        print(f'  Added: {fname}')
zip_mb = os.path.getsize(ZIP_PATH) / 1000000.0
print(f'\n✅ Zip created: {ZIP_PATH}  ({zip_mb:.1f} MB)')
print('Download it from the OUTPUT tab → unet_checkpoints.zip')
print()
print('Then put unet_best.pth in your local: Stego-Gan/checkpoints/unet_best.pth')
