from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))

class StegNet(nn.Module):

    def __init__(self, dropout_p: float=0.5) -> None:
        super().__init__()
        self.block1 = _conv_block(3, 32)
        self.block2 = _conv_block(32, 64)
        self.block3 = _conv_block(64, 128)
        self.block4 = _conv_block(128, 256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(p=dropout_p), nn.Linear(128, 1))
        self._gradcam_activations: Optional[torch.Tensor] = None
        self._gradcam_gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:

        def _forward_hook(module, inp, output):
            self._gradcam_activations = output.detach()

        def _backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients = grad_output[0].detach()
        h1 = self.block4.register_forward_hook(_forward_hook)
        h2 = self.block4.register_full_backward_hook(_backward_hook)
        self._hooks.extend([h1, h2])

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        logit = self.classifier(x)
        return logit

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    @torch.enable_grad()
    def generate_gradcam(self, image_tensor: torch.Tensor, target_class: int=1) -> Tuple[float, np.ndarray]:
        self.eval()
        self.zero_grad()
        x = image_tensor
        logit = self.forward(x)
        prob = torch.sigmoid(logit)
        probability = prob.item()
        score = prob if target_class == 1 else 1.0 - prob
        score.backward()
        activations = self._gradcam_activations
        gradients = self._gradcam_gradients
        alpha = gradients.mean(dim=[2, 3], keepdim=True)
        weighted_cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(weighted_cam)
        h_in, w_in = (image_tensor.shape[2], image_tensor.shape[3])
        cam_upscaled = F.interpolate(cam, size=(h_in, w_in), mode='bilinear', align_corners=False)
        heatmap = cam_upscaled.squeeze().cpu().numpy()
        cam_min, cam_max = (float(heatmap.min()), float(heatmap.max()))
        if cam_max - cam_min > 1e-08:
            heatmap = (heatmap - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(heatmap)
        self.zero_grad()
        return (probability, heatmap)
