import torch.nn as nn
import torch

from visualDet3D.networks.backbones.dla import dla102
from visualDet3D.networks.backbones.dlaup import DLAUp
from visualDet3D.networks.detectors.dfe import DepthAwareFE
from visualDet3D.networks.detectors.dpe import DepthAwarePosEnc
from visualDet3D.networks.detectors.dtr import DepthAwareTransformer
from visualDet3D.networks.detectors.denoising_diffusion_pytorch import Unet, GaussianDiffusion
from visualDet3D.networks.detectors.wc import WeatherCodebook
            

class MonoWAD(nn.Module):
    def __init__(self, backbone_arguments=dict()):
        super(MonoWAD, self).__init__()
        self.backbone = dla102(pretrained=True, return_levels=True)
        channels = self.backbone.channels
        self.first_level = 3
        scales = [2**i for i in range(len(channels[self.first_level:]))]
        self.neck = DLAUp(channels[self.first_level:], scales_list=scales)

        self.output_channel_num = 256
        self.dpe = DepthAwarePosEnc(self.output_channel_num)
        self.depth_embed = nn.Embedding(100, self.output_channel_num)
        self.dtr = DepthAwareTransformer(self.output_channel_num)
        self.dfe = DepthAwareFE(self.output_channel_num)
        self.img_conv = nn.Conv2d(self.output_channel_num, self.output_channel_num, kernel_size=3, padding=1)
        
        self.num_timesteps = 15
        self.codebook = WeatherCodebook(4096, self.output_channel_num, 256)
        self.diffusion_init()

    def diffusion_init(self):
        self.unet = Unet(
            dim=64,
            dim_mults=(1, 2, 4),
            full_attn=(False, False, True),
            channels=256,
            flash_attn=True
        )

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=(36, 160),
            timesteps=self.num_timesteps,    # number of steps
        )

    def predict(self, x, t: int, codebook=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.diffusion.p_mean_variance(
            x=x, t=batched_times, codebook=codebook, clip_denoised=True)
        noise = x - x_start if t > 0 else 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def enhancing_feature_representation(self, fog_feat, codebook=None):
        diff_feat = [fog_feat]
        for t in reversed(range(self.num_timesteps)):
            fog_feat, x_start = self.predict(fog_feat, t, codebook)
            diff_feat.append(fog_feat)
        
        diffusion_feat = fog_feat
        return diffusion_feat

    def weather_adaptive_diffusion(self, origin_feat, noised_feat=None, codebook=None):
        if noised_feat is not None:
            loss = self.diffusion(origin_feat, noised_feat, codebook)
            return loss, self.enhancing_feature_representation(origin_feat, codebook)
        else:
            return self.enhancing_feature_representation(origin_feat, codebook)

    def forward(self, x):
        training = x["training"]
        origin_feat = self.backbone(x['image'])
        origin_feat = self.neck(origin_feat[self.first_level:])
        
        if training:
            foggy_feat = self.backbone(x['foggy'])
            foggy_feat = self.neck(foggy_feat[self.first_level:])
            weather_reference_feat, codebook_loss = self.codebook(origin_feat, foggy_feat)
            diff_loss, x = self.weather_adaptive_diffusion(origin_feat, foggy_feat, weather_reference_feat)
            diff_loss = diff_loss + codebook_loss
        else:
            weather_reference_feat = self.codebook(origin_feat)
            x = self.weather_adaptive_diffusion(origin_feat, codebook=weather_reference_feat)
        N, C, H, W = x.shape

        depth, depth_guide, depth_feat = self.dfe(x)
        
        depth_feat = depth_feat.permute(0, 2, 3, 1).view(N, H*W, C)
        
        depth_guide = depth_guide.argmax(1)
        depth_emb = self.depth_embed(depth_guide).view(N, H*W, C)
        depth_emb = self.dpe(depth_emb, (H,W))
        
        img_feat = x + self.img_conv(x)
        img_feat = img_feat.view(N, H*W, C)
        feat = self.dtr(depth_feat, img_feat, depth_emb)
        feat = feat.permute(0, 2, 1).view(N,C,H,W)
        if training:
            return feat, depth, diff_loss
        else:
            return feat, depth
