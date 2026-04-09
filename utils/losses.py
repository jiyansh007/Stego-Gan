from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ReconstructionLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target)

class PerceptualLoss(nn.Module):
    _LAYER_INDICES = {'relu1_2': 4, 'relu2_2': 9, 'relu3_3': 16}
    _LAYER_WEIGHTS = {'relu1_2': 1.0, 'relu2_2': 0.75, 'relu3_3': 0.5}

    def __init__(self) -> None:
        super().__init__()
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        except AttributeError:
            vgg = models.vgg16(pretrained=True)
        vgg_features = list(vgg.features.children())
        sorted_layers = sorted(self._LAYER_INDICES.values())
        self.slice_names = sorted(self._LAYER_INDICES.keys())
        self.slices = nn.ModuleList()
        prev_idx = 0
        for layer_idx in sorted_layers:
            self.slices.append(nn.Sequential(*vgg_features[prev_idx:layer_idx + 1]))
            prev_idx = layer_idx + 1
        for param in self.parameters():
            param.requires_grad = False
        self.mse = nn.MSELoss()
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x_01 = (x + 1.0) / 2.0
        x_vgg = (x_01 - self.vgg_mean) / self.vgg_std
        return x_vgg

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_x = self._preprocess(pred)
        target_x = self._preprocess(target)
        total_loss = torch.tensor(0.0, device=pred.device)
        for slice_net, layer_name in zip(self.slices, self.slice_names):
            pred_x = slice_net(pred_x)
            with torch.no_grad():
                target_x = slice_net(target_x)
            weight = self._LAYER_WEIGHTS[layer_name]
            total_loss = total_loss + weight * self.mse(pred_x, target_x)
        return total_loss

class SSIMLoss(nn.Module):

    def __init__(self, window_size: int=11, sigma: float=1.5) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self._window: Optional[torch.Tensor] = None

    def _build_gaussian_window(self, device: torch.device) -> torch.Tensor:
        coords = torch.arange(self.window_size, dtype=torch.float32, device=device)
        coords -= self.window_size // 2
        g1d = torch.exp(-coords ** 2 / (2.0 * self.sigma ** 2))
        g1d /= g1d.sum()
        g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)
        return g2d.unsqueeze(0).unsqueeze(0)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad = self.window_size // 2
        if self._window is None or self._window.device != x.device:
            self._window = self._build_gaussian_window(x.device)
        window = self._window.expand(C, 1, self.window_size, self.window_size)
        mu_x = F.conv2d(x, window, padding=pad, groups=C)
        mu_y = F.conv2d(y, window, padding=pad, groups=C)
        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy
        C1 = (0.01 * 2) ** 2
        C2 = (0.03 * 2) ** 2
        numerator = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
        denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = numerator / (denominator + 1e-08)
        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._ssim(pred, target)

class UNetLoss(nn.Module):

    def __init__(self, lambda_recon: float=1.0, lambda_perc: float=0.1, lambda_ssim: float=0.5) -> None:
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.recon_loss = ReconstructionLoss()
        self.perc_loss = PerceptualLoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_recon = self.recon_loss(pred, target)
        l_perc = self.perc_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        return self.lambda_recon * l_recon + self.lambda_perc * l_perc + self.lambda_ssim * l_ssim

    def breakdown(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        l_recon = self.recon_loss(pred, target)
        l_perc = self.perc_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        total = self.lambda_recon * l_recon + self.lambda_perc * l_perc + self.lambda_ssim * l_ssim
        return {'total': total.item(), 'l1': l_recon.item(), 'perceptual': l_perc.item(), 'ssim': l_ssim.item()}
