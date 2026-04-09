import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data.injector import StegoDataset
from models.stegnet import StegNet
from models.unet_purifier import UNetPurifier
from utils.losses import UNetLoss

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stego-GAN Training Script')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Path to the folder containing clean training images (JPG/PNG).')
    parser.add_argument('--image_size', type=int, default=256, help='Resize all images to this square size before training (default: 256).')
    parser.add_argument('--payload_fraction', type=float, default=0.5, help='Fraction of pixel channel LSBs to inject (0.0–1.0, default: 0.5).')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of dataset to hold out for validation (Only used if no train/val subfolders exist).')
    parser.add_argument('--model', type=str, default='both', choices=['stegnet', 'unet', 'both'], help='Which model to train (default: both).')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs (default: 30).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16).')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate (default: 1e-4).')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader worker processes (default: 2).')
    parser.add_argument('--lambda_recon', type=float, default=1.0, help='L1 reconstruction loss weight.')
    parser.add_argument('--lambda_perc', type=float, default=0.1, help='Perceptual loss weight.')
    parser.add_argument('--lambda_ssim', type=float, default=0.5, help='SSIM loss weight.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints (default: checkpoints/).')
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint .pth file to resume training from.')
    parser.add_argument('--save_every', type=int, default=5, help='Save a checkpoint every N epochs (default: 5).')
    return parser.parse_args()

def build_dataloaders(args: argparse.Namespace, device: torch.device, for_unet: bool=False):
    apply_srm = not for_unet
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_set = StegoDataset(root_dir=train_dir, image_size=args.image_size, payload_fraction=args.payload_fraction, augment=True, apply_srm=apply_srm)
        val_set = StegoDataset(root_dir=val_dir, image_size=args.image_size, payload_fraction=args.payload_fraction, augment=False, apply_srm=apply_srm)
        n_train = len(train_set)
        n_val = len(val_set)
        n_total = n_train + n_val
        print(f"[Data] Found explicit 'train' and 'val' folders in {args.data_dir}.")
        print(f'[Data] Total images: {n_total}  |  Train: {n_train}  |  Val: {n_val}')
        print(f"[Data] SRM filter : {('ON (StegNet mode)' if apply_srm else 'OFF (U-Net pixel-space mode)')}")
    else:
        full_dataset = StegoDataset(root_dir=args.data_dir, image_size=args.image_size, payload_fraction=args.payload_fraction, augment=True, apply_srm=apply_srm)
        n_total = len(full_dataset)
        n_val = max(1, int(n_total * args.val_split))
        n_train = n_total - n_val
        print(f'[Data] Total images: {n_total}  |  Train: {n_train}  |  Val: {n_val}')
        print(f"[Data] SRM filter : {('ON (StegNet mode)' if apply_srm else 'OFF (U-Net pixel-space mode)')}")
        train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        val_set.dataset.augment = False
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    return (train_loader, val_loader, n_total)

def train_stegnet(args: argparse.Namespace, device: torch.device) -> None:
    print('\n' + '═' * 60)
    print('  Training StegNet (Binary CNN Detector)')
    print('═' * 60)
    model = StegNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"[StegNet] Resumed from '{args.resume}' (epoch {start_epoch})")
    train_loader, val_loader, _ = build_dataloaders(args, device, for_unet=False)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            stego = batch['stego'].to(device)
            clean = batch['clean'].to(device)
            combined_images = torch.cat([clean, stego], dim=0)
            labels = torch.cat([torch.zeros(clean.size(0), 1), torch.ones(stego.size(0), 1)], dim=0).to(device)
            perm = torch.randperm(combined_images.size(0))
            combined_images = combined_images[perm]
            labels = labels[perm]
            optimizer.zero_grad()
            logits = model(combined_images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            probs_sigmoid = torch.sigmoid(logits)
            predicted = (probs_sigmoid >= 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.numel()
            if (batch_idx + 1) % 10 == 0:
                running_acc = 100.0 * train_correct / train_total
                print(f'  [Epoch {epoch + 1}/{start_epoch + args.epochs}] Step {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {running_acc:.1f}%')
        val_loss, val_acc = _validate_stegnet(model, criterion, val_loader, device)
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        elapsed = time.time() - t0
        print(f'\n[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | Time: {elapsed:.1f}s')
        scheduler.step(val_loss)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        _save_checkpoint(model, optimizer, epoch, val_loss, best_val_loss, args.checkpoint_dir, 'stegnet', is_best)
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(model, optimizer, epoch, val_loss, best_val_loss, args.checkpoint_dir, 'stegnet', False, suffix=f'_epoch{epoch + 1}')
    print(f'\n[StegNet] Training complete. Best val loss: {best_val_loss:.4f}')
    print(f'[StegNet] Best weights saved to: {args.checkpoint_dir}/stegnet_best.pth')

def _validate_stegnet(model: StegNet, criterion: nn.Module, val_loader: DataLoader, device: torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            stego = batch['stego'].to(device)
            clean = batch['clean'].to(device)
            combined = torch.cat([clean, stego], dim=0)
            labels = torch.cat([torch.zeros(clean.size(0), 1), torch.ones(stego.size(0), 1)], dim=0).to(device)
            logits = model(combined)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            predicted = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.numel()
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * total_correct / total_samples
    return (avg_loss, accuracy)

def train_unet(args: argparse.Namespace, device: torch.device) -> None:
    print('\n' + '═' * 60)
    print('  Training U-Net Purifier')
    print('═' * 60)
    model = UNetPurifier().to(device)
    criterion = UNetLoss(lambda_recon=args.lambda_recon, lambda_perc=args.lambda_perc, lambda_ssim=args.lambda_ssim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"[U-Net] Resumed from '{args.resume}' (epoch {start_epoch})")
    train_loader, val_loader, _ = build_dataloaders(args, device, for_unet=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        running_loss = {'total': 0.0, 'l1': 0.0, 'perceptual': 0.0, 'ssim': 0.0}
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            stego = batch['stego'].to(device)
            clean = batch['clean'].to(device)
            optimizer.zero_grad()
            purified = model(stego)
            detail = criterion.breakdown(purified, clean)
            loss = torch.tensor(detail['total'], device=device, requires_grad=False)
            loss = criterion(purified, clean)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            detail = criterion.breakdown(purified.detach(), clean.detach())
            for k in running_loss:
                running_loss[k] += detail[k]
            if (batch_idx + 1) % 10 == 0:
                step = batch_idx + 1
                print(f"  [Epoch {epoch + 1}/{start_epoch + args.epochs}] Step {step}/{len(train_loader)} | Loss: {detail['total']:.4f} (L1: {detail['l1']:.4f}, Perc: {detail['perceptual']:.4f}, SSIM: {detail['ssim']:.4f})")
        val_loss, val_detail = _validate_unet(model, criterion, val_loader, device)
        elapsed = time.time() - t0
        n = len(train_loader)
        print(f"\n[Epoch {epoch + 1}] Train: {running_loss['total'] / n:.4f} | Val: {val_loss:.4f} (L1: {val_detail['l1']:.4f}, Perc: {val_detail['perceptual']:.4f}, SSIM: {val_detail['ssim']:.4f}) | Time: {elapsed:.1f}s")
        scheduler.step(val_loss)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        _save_checkpoint(model, optimizer, epoch, val_loss, best_val_loss, args.checkpoint_dir, 'unet', is_best)
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(model, optimizer, epoch, val_loss, best_val_loss, args.checkpoint_dir, 'unet', False, suffix=f'_epoch{epoch + 1}')
    print(f'\n[U-Net] Training complete. Best val loss: {best_val_loss:.4f}')
    print(f'[U-Net] Best weights saved to: {args.checkpoint_dir}/unet_best.pth')

def _validate_unet(model: UNetPurifier, criterion: UNetLoss, val_loader: DataLoader, device: torch.device) -> tuple:
    model.eval()
    total = {'total': 0.0, 'l1': 0.0, 'perceptual': 0.0, 'ssim': 0.0}
    with torch.no_grad():
        for batch in val_loader:
            stego = batch['stego'].to(device)
            clean = batch['clean'].to(device)
            purified = model(stego)
            detail = criterion.breakdown(purified, clean)
            for k in total:
                total[k] += detail[k]
    n = len(val_loader)
    avg = {k: v / n for k, v in total.items()}
    return (avg['total'], avg)

def _save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, val_loss: float, best_loss: float, ckpt_dir: str, model_name: str, is_best: bool, suffix: str='') -> None:
    state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_loss, 'best_val_loss': best_loss}
    latest_path = os.path.join(ckpt_dir, f'{model_name}_latest.pth')
    torch.save(state, latest_path)
    if is_best:
        best_path = os.path.join(ckpt_dir, f'{model_name}_best.pth')
        torch.save(state, best_path)
        print(f'  ✓ New best {model_name} checkpoint saved → {best_path}')
    if suffix:
        periodic_path = os.path.join(ckpt_dir, f'{model_name}{suffix}.pth')
        torch.save(state, periodic_path)
        print(f'  ✓ Periodic checkpoint saved → {periodic_path}')

def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 60}")
    print(f'  Stego-GAN Training Script')
    print(f"{'=' * 60}")
    print(f'  Device     : {device}')
    print(f'  Model(s)   : {args.model}')
    print(f'  Data dir   : {args.data_dir}')
    print(f'  Image size : {args.image_size}×{args.image_size}')
    print(f'  Epochs     : {args.epochs}')
    print(f'  Batch size : {args.batch_size}')
    print(f'  LR         : {args.lr}')
    print(f"{'=' * 60}\n")
    torch.manual_seed(42)
    if args.model in ('stegnet', 'both'):
        train_stegnet(args, device)
    if args.model in ('unet', 'both'):
        if args.model == 'both':
            args.lr = min(args.lr, 0.0005)
            args.resume = None
        train_unet(args, device)
    print('\n[Done] All training complete.')
if __name__ == '__main__':
    main()
