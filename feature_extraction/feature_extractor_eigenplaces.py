
from .common import TorchHubFeatureExtractor


class EigenPlacesFeatureExtractor(TorchHubFeatureExtractor):
    def __init__(self):
        super().__init__("gmberton/eigenplaces", backbone="ResNet50", fc_output_dim=2048)
