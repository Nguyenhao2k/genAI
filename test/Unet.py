import torch
import torchvision
import torchvision.transforms.functional as F

from torch import nn
from tqdm import tqdm
from torchmultimodal.diffusion_labs.modules.adapters.cfguidance import CFGuidance
from torchmultimodal.diffusion_labs.modules.losses.diffusion_hybrid_loss import DiffusionHybridLoss
from torchmultimodal.diffusion_labs.samplers.ddpm import DDPModule
from torchmultimodal.diffusion_labs.predictors.noise_predictor import NoisePredictor
from torchmultimodal.diffusion_labs.schedules.discrete_gaussian_schedule import linear_beta_schedule, DiscreteGaussianSchedule
from torchmultimodal.diffusion_labs.transforms.diffusion_transform import RandomDiffusionSteps
from torchmultimodal.diffusion_labs.utils.common import DiffusionOutput


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channes, cond_channels):
    super().__init__()
    self.block = nn.Sequential(
        nn.Conv2d(in_channels + cond_channels, out_channes, kernel_size=3, padding=1),
        nn.Relu(),
        nn.Conv2d(out_channes, out_channes, kernel_size=3, padding=1),
        nn.Relu(),
    )
    self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, cond):
        _, _, H, W = x.size()
        c = c 