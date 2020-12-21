from unittest import TestCase

import torch

from data_prep.HippocampusDatasetLoader import LoadHippocampusData
from inference.UNetInferenceAgent import UNetInferenceAgent
from networks.RecursiveUNet import UNet
from run_ml_pipeline import Config


class TestUNetInferenceAgent(TestCase):

    def setUp(self) -> None:
        # load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(num_classes=3)
        self.model.to(self.device)

        # load data
        c = Config()
        self.data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)

        self.inference_agent = UNetInferenceAgent(model=self.model, device=self.device)



    def test_single_volume_inference(self):
        volume = self.data[0]['image']
        volume = volume[None]
        pred = self.inference_agent.single_volume_inference(volume[0])
        self.assertEqual(3, len(pred.shape))
