import os
import random
from typing import Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T

class LSBInjector:

    def inject(self, image: Image.Image, payload_fraction: float=0.5) -> Image.Image:
        image = image.convert('RGB')
        pixel_array = np.array(image, dtype=np.uint8).copy()
        height, width, channels = pixel_array.shape
        total_values = height * width * channels
        num_bits_to_embed = int(total_values * payload_fraction)
        payload_bits = np.random.randint(0, 2, size=num_bits_to_embed, dtype=np.uint8)
        injection_indices = np.random.choice(total_values, size=num_bits_to_embed, replace=False)
        flat = pixel_array.reshape(-1)
        flat[injection_indices] = flat[injection_indices] & np.uint8(254) | payload_bits
        stego_array = flat.reshape(height, width, channels)
        return Image.fromarray(stego_array, mode='RGB')

class StegoDataset(Dataset):
    _VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, root_dir: str, image_size: int=256, payload_fraction: float=0.5, augment: bool=True, apply_srm: bool=True) -> None:
        self.root_dir = root_dir
        self.payload_fraction = payload_fraction
        self.apply_srm = apply_srm
        self.injector = LSBInjector()
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in self._VALID_EXTENSIONS:
                    self.image_paths.append(os.path.join(root, fname))
        self.image_paths.sort()
        if not self.image_paths:
            raise FileNotFoundError(f"No valid images found in '{root_dir}'.\nPlease ensure JPG/PNG images exist in the folder before training.")
        spatial_transforms = [T.Resize((image_size, image_size))]
        if augment:
            spatial_transforms.append(T.RandomHorizontalFlip(p=0.5))
        self.spatial_transform = T.Compose(spatial_transforms)
        self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        srm_kernel = torch.tensor([[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 1.0, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]], dtype=torch.float32)
        self._srm_kernel = srm_kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        self._srm_kernel.requires_grad = False

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        clean_pil = Image.open(img_path).convert('RGB')
        stego_pil = self.injector.inject(clean_pil, payload_fraction=self.payload_fraction)
        seed = random.randint(0, 2 ** 31)
        random.seed(seed)
        torch.manual_seed(seed)
        clean_pil_t = self.spatial_transform(clean_pil)
        random.seed(seed)
        torch.manual_seed(seed)
        stego_pil_t = self.spatial_transform(stego_pil)
        clean_tensor = self.to_tensor(clean_pil_t)
        stego_tensor = self.to_tensor(stego_pil_t)
        if self.apply_srm:
            with torch.no_grad():
                clean_tensor = F.conv2d(clean_tensor.unsqueeze(0), self._srm_kernel, padding=1, groups=3).squeeze(0)
                stego_tensor = F.conv2d(stego_tensor.unsqueeze(0), self._srm_kernel, padding=1, groups=3).squeeze(0)
        return {'clean': clean_tensor, 'stego': stego_tensor, 'label': 1}
