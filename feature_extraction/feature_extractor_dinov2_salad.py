import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data

from typing import List
import numpy as np
from tqdm.auto import tqdm

from .common import get_device
from .feature_extractor import FeatureExtractor


class SALADImageDataset(data.Dataset):
    """Dataset that preprocesses images for DINOv2 SALAD (322x322 input)."""

    def __init__(self, imgs):
        super().__init__()
        self.images = imgs
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((322, 322),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        img = self.images[index]
        img = self.transform(img)
        return img, index

    def __len__(self):
        return len(self.images)


class DINOv2SALADFeatureExtractor(FeatureExtractor):
    """DINOv2 SALAD descriptor (ViT-B/14, 8448-dim)."""

    def __init__(self):
        self.device = get_device()
        self.model = torch.hub.load("serizba/salad", "dinov2_salad")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.dim = 8448

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        img_set = SALADImageDataset(imgs)
        test_data_loader = DataLoader(dataset=img_set, num_workers=4,
                                      batch_size=4, shuffle=False,
                                      pin_memory=torch.cuda.is_available())
        with torch.no_grad():
            global_feats = np.empty((len(img_set), self.dim), dtype=np.float32)
            for input_data, indices in tqdm(test_data_loader):
                indices_np = indices.numpy()
                input_data = input_data.to(self.device)
                image_encoding = self.model(input_data)
                global_feats[indices_np, :] = image_encoding.cpu().numpy()
        return global_feats
