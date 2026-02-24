
from .common import TorchHubFeatureExtractor


class CosPlaceFeatureExtractor(TorchHubFeatureExtractor):
    def __init__(self):
        super().__init__("gmberton/cosplace", backbone="ResNet50", fc_output_dim=2048)
