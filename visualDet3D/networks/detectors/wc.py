import torch
import torch.nn as nn
import torch.nn.functional as F

class WeatherCodebook(nn.Module):
    def __init__(self, codebook_dim, feature_dim, channel):
        super(WeatherCodebook, self).__init__()
        self.codebook_dim = codebook_dim
        self.feature_dim = feature_dim
        self.channel_dim = channel
        self.weather_codebook = nn.Embedding(self.codebook_dim, self.feature_dim)
        self.quant_conv = nn.Conv2d(self.channel_dim, self.channel_dim, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, input, second_input=None):
        if second_input is not None:
            assert self.training, "second_input only used in training"
            clear, adverse = input, second_input
            
            # Element-wise quantization for clear
            clear_q = self.quant_conv(clear)
            clear_feat = clear_q.permute(0, 2, 3, 1).contiguous()
            clear_flatten = F.normalize(clear_feat.view(-1, self.feature_dim), dim=1)
            min_clear_indices = torch.argmin(torch.cdist(clear_flatten, self.weather_codebook.weight, p=2)**2, dim=1)
            weather_reference_feat_clear = self.weather_codebook(min_clear_indices).view(clear_feat.shape).permute(0, 3, 1, 2)
            
            global_clear = self.avgpool(weather_reference_feat_clear).squeeze()
            clear_codebook_softmax = F.log_softmax(global_clear, dim=1)

            clear = F.softmax(self.avgpool(clear).squeeze(), dim=1)
            l_cke = F.kl_div(clear_codebook_softmax, clear, reduction='batchmean')  # CKE loss
            
            # Element-wise quantization for adverse
            adverse_q = self.quant_conv(adverse)
            adverse_feat = adverse_q.permute(0, 2, 3, 1).contiguous()
            adverse_flatten = F.normalize(adverse_feat.view(-1, self.feature_dim), dim=1)
            min_adverse_indices = torch.argmin(torch.cdist(adverse_flatten, self.weather_codebook.weight, p=2)**2, dim=1)
            weather_reference_feat_adverse = self.weather_codebook(min_adverse_indices).view(adverse_feat.shape).permute(0, 3, 1, 2)
            
            l_wig = F.mse_loss(weather_reference_feat_clear, weather_reference_feat_adverse)  # WIG loss
            
            l_ckr = l_cke + l_wig  # CKR loss
            
            return weather_reference_feat_clear, l_ckr
        else:
            # Inference mode
            input_q = self.quant_conv(input)
            input_feat = input_q.permute(0, 2, 3, 1).contiguous()
            input_flatten = F.normalize(input_feat.view(-1, self.feature_dim), dim=1)
            min_indices = torch.argmin(torch.cdist(input_flatten, self.weather_codebook.weight, p=2)**2, dim=1)
            weather_reference_feat = self.weather_codebook(min_indices).view(input_feat.shape).permute(0, 3, 1, 2)
            
            return weather_reference_feat
