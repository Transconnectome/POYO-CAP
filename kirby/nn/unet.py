import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super(UNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1)  # (32x64) → (16x32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)           # → (8x16)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)           # → (4x8)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)          # → (2x4)
        self.conv5 = nn.Conv2d(128, latent_dim, kernel_size=(2, 3), stride=1, padding=0) # → (1x2)


        self.leaky_relu = nn.LeakyReLU(0.2)

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            init.xavier_normal_(layer.weight)

    def forward(self, x):
        skips = []
        x1 = self.leaky_relu(self.conv1(x)); skips.append(x1)
        x2 = self.leaky_relu(self.conv2(x1)); skips.append(x2)
        x3 = self.leaky_relu(self.conv3(x2)); skips.append(x3)
        x4 = self.leaky_relu(self.conv4(x3)); skips.append(x4)
        x5 = self.conv5(x4)
        return x5, skips

class NeuralLatentMapper(nn.Module):
    def __init__(self, neural_dims=64, latent_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(neural_dims, 64)
        self.fc2 = nn.Linear(64, latent_dim * 1 * 2)
        self.relu = nn.LeakyReLU(0.2)
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, self.latent_dim, 1, 2)

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2)  # (1x2) → (2x4)
        self.up2 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1)  # → (4x8)
        self.up3 = nn.ConvTranspose2d(64 + 64, 32, kernel_size=4, stride=2, padding=1)    # → (8x16)
        self.up4 = nn.ConvTranspose2d(32 + 32, 16, kernel_size=4, stride=2, padding=1)    # → (16x32)
        self.up5 = nn.ConvTranspose2d(16 + 16, out_channels, kernel_size=4, stride=2, padding=1)  # → (32x64)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        for layer in [self.up1, self.up2, self.up3, self.up4, self.up5]:
            init.xavier_normal_(layer.weight)

    def forward(self, x, skips):
        x = self.leaky_relu(self.up1(x))
        x = torch.cat([x, F.interpolate(skips[3], x.shape[2:])], dim=1)

        x = self.leaky_relu(self.up2(x))
        x = torch.cat([x, F.interpolate(skips[2], x.shape[2:])], dim=1)

        x = self.leaky_relu(self.up3(x))
        x = torch.cat([x, F.interpolate(skips[1], x.shape[2:])], dim=1)

        x = self.leaky_relu(self.up4(x))
        x = torch.cat([x, F.interpolate(skips[0], x.shape[2:])], dim=1)

        x = self.sigmoid(self.up5(x))
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, latent_dim)
        self.remapper = NeuralLatentMapper(neural_dims=64, latent_dim=latent_dim)
        self.decoder = UNetDecoder(latent_dim, out_channels)

    def forward(self, noise, latent):
        hidden, skips = self.encoder(noise)
        decoder_input = hidden + self.remapper(latent)
        out = self.decoder(decoder_input, skips)
        return out[:, 0, :, :]  # (batch, 32, 64)

class FiLMGenerator(nn.Module):
    def __init__(self, embedding_dim, feature_dims):
        super().__init__()
        self.film_layers = nn.ModuleList([
                                            nn.Sequential(
                                                nn.LayerNorm(embedding_dim),
                                                nn.Linear(embedding_dim, d * 2),
                                                nn.ReLU(),
                                                nn.Linear(d * 2, d * 2)
                                            ) for d in feature_dims
                                        ])

    def forward(self, embedding):
        gammas, betas = [], []
        for film in self.film_layers:
            out = film(embedding)
            gamma, beta = out.chunk(2, dim=1)
            gammas.append(gamma)
            betas.append(beta)
        return gammas, betas

class FiLMLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        # x: (B, C, H, W), gamma/beta: (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class UNetDecoderFiLM(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(64 + 64, 32, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(32 + 32, 16, kernel_size=4, stride=2, padding=1)
        self.up5 = nn.ConvTranspose2d(16 + 16, out_channels, kernel_size=4, stride=2, padding=1)

        self.film = FiLMLayer()
        self.film_generator = FiLMGenerator(embedding_dim, [128, 64, 32, 16, out_channels])

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skips, latent):
        gammas, betas = self.film_generator(latent)

        x = self.film(self.leaky_relu(self.up1(x)), gammas[0], betas[0])
        x = torch.cat([x, F.interpolate(skips[3], x.shape[2:])], dim=1)

        x = self.film(self.leaky_relu(self.up2(x)), gammas[1], betas[1])
        x = torch.cat([x, F.interpolate(skips[2], x.shape[2:])], dim=1)

        x = self.film(self.leaky_relu(self.up3(x)), gammas[2], betas[2])
        x = torch.cat([x, F.interpolate(skips[1], x.shape[2:])], dim=1)

        x = self.film(self.leaky_relu(self.up4(x)), gammas[3], betas[3])
        x = torch.cat([x, F.interpolate(skips[0], x.shape[2:])], dim=1)

        x = self.film(self.sigmoid(self.up5(x)), gammas[4], betas[4])
        return x

class UNetFiLM(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, latent_dim)
        self.remapper = NeuralLatentMapper(neural_dims=64, latent_dim=latent_dim)
        #self.decoder = UNetDecoderFiLM(latent_dim, out_channels, embedding_dim=latent_dim)
        self.decoder = UNetDecoderFiLM(latent_dim, out_channels, embedding_dim=latent_dim * 2)


    def forward(self, noise, latent):
        hidden, skips = self.encoder(noise)
        # print(self.remapper(latent).shape)  # Expect: (B, 128, 1, 2)
        latent_vec = self.remapper(latent).view(latent.shape[0], -1)  # flatten to (B, latent_dim)
        print("latent vec stats", latent_vec.mean().item(), latent_vec.std().item())
        out = self.decoder(hidden, skips, latent_vec)
        return out[:, 0, :, :]  # (B, 32, 64)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Output projection
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate Q, K, V
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # [B, HW, C//8]
        k = self.key(x).view(batch_size, -1, height * width)  # [B, C//8, HW]
        v = self.value(x).view(batch_size, -1, height * width)  # [B, C, HW]
        
        # Attention computation
        attention = torch.bmm(q, k)  # [B, HW, HW]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(batch_size, channels, height, width)
        
        # Apply output projection and residual connection
        out = self.out(out)
        out = self.gamma * out + x
        
        return out



class LargeUNetDecoderOnly(nn.Module):
    def __init__(self, in_channels, out_channels, feat_ch=32):
        super().__init__()

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # 1x1 → 2x2 → 4x4 → 8x8 → 16x16 → 32x32 (same up to here)
        self.up1 = up_block(in_channels, 128)   # 1 → 2
        self.up2 = up_block(128, 64)            # 2 → 4
        self.up3 = up_block(64, 32)             # 4 → 8
        self.up4 = up_block(32, 16)             # 8 → 16

        self.final_up = up_block(16, feat_ch)   # 16 → 32 (keep features)

        # 32x32 → 64x64 → 128x128 → 128x256
        '''
        self.extra_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32→64
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64→128
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False),  # 128x128 → 128x256 (width only 2x)
            nn.Conv2d(feat_ch, out_channels, kernel_size=3, padding=1),  # project to out_channels only at the end
        )
        '''
        self.extra_up = nn.Sequential(
            # 32 → 64
            nn.Conv2d(feat_ch, feat_ch * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),  # sharpness UP!
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),  # Refine
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 → 128
            nn.Conv2d(feat_ch, feat_ch * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),  # Refine
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 → 128×256
            nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_ch, out_channels, kernel_size=3, padding=1),  # Final
        )
        

        

        self.skip1_proj = nn.Linear(in_channels, 128 * 2 * 2)
        self.skip2_proj = nn.Linear(in_channels, 64 * 4 * 4)
        self.skip3_proj = nn.Linear(in_channels, 32 * 8 * 8)
        self.skip4_proj = nn.Linear(in_channels, 16 * 16 * 16)

        self.skip1_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.skip2_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.skip3_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.skip4_conv = nn.Conv2d(32, 16, kernel_size=1)

        self.tanh = nn.Tanh()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    
    def forward(self, latent):
        skip1 = self.skip1_proj(latent).view(latent.size(0), 128, 2, 2)
        skip2 = self.skip2_proj(latent).view(latent.size(0), 64, 4, 4)
        skip3 = self.skip3_proj(latent).view(latent.size(0), 32, 8, 8)
        skip4 = self.skip4_proj(latent).view(latent.size(0), 16, 16, 16)

        x = latent.view(latent.size(0), latent.size(1), 1, 1)

        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.skip1_conv(x)
    
        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.skip2_conv(x)

        x = self.up3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.skip3_conv(x)

        x = self.up4(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.skip4_conv(x)
        
        x = self.final_up(x)
        x = self.extra_up(x)

        x = self.tanh(x)

        return x

import torch
import torch.nn as nn
import torch.nn.init as init

# ---------------------------
# Utils: ICNR init for PixelShuffle
# ---------------------------
def icnr_(w: torch.Tensor, scale: int = 2, init_fn=init.kaiming_normal_):
    """
    w: (out_c, in_c, kH, kW) with out_c = target_channels * scale^2
    """
    oc, ic, kh, kw = w.shape
    subc = oc // (scale ** 2)
    k = torch.empty(subc, ic, kh, kw, device=w.device, dtype=w.dtype)
    init_fn(k, a=0.2, nonlinearity='leaky_relu')
    k = k.repeat_interleave(scale ** 2, dim=0)
    with torch.no_grad():
        w.copy_(k)

# ---------------------------
# Channel Attention (SE-style)
# ---------------------------
class CALayer(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, max(1, c // r), 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(max(1, c // r), c, 1, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

# ---------------------------
# Residual Dense Block (ESRGAN style, 5 convs)
# ---------------------------
class ResidualDenseBlock5C(nn.Module):
    def __init__(self, c, growth=None, scale=0.2):
        super().__init__()
        g = growth if growth is not None else max(8, c // 2)
        self.c = c
        self.scale = scale
        # 5 convs, concatenated features
        self.conv1 = nn.Conv2d(c, g, 3, 1, 1)
        self.conv2 = nn.Conv2d(c + g, g, 3, 1, 1)
        self.conv3 = nn.Conv2d(c + 2 * g, g, 3, 1, 1)
        self.conv4 = nn.Conv2d(c + 3 * g, g, 3, 1, 1)
        self.conv5 = nn.Conv2d(c + 4 * g, c, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * self.scale

# ---------------------------
# RRDB: Residual-in-Residual Dense Block
# ---------------------------
class RRDB(nn.Module):
    def __init__(self, c, growth=None, scale=0.2):
        super().__init__()
        self.rdb1 = ResidualDenseBlock5C(c, growth=growth, scale=scale)
        self.rdb2 = ResidualDenseBlock5C(c, growth=growth, scale=scale)
        self.rdb3 = ResidualDenseBlock5C(c, growth=growth, scale=scale)
        self.ca = CALayer(c)
        self.scale = scale

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = self.ca(out)
        return x + out * self.scale

# ---------------------------
# PixelShuffle up-block (no BN, uses ICNR)
# ---------------------------
class UpShuffle(nn.Module):
    def __init__(self, c_in, c_out, scale=2):
        super().__init__()
        self.upconv = nn.Conv2d(c_in, c_out * (scale ** 2), 3, 1, 1)
        self.ps = nn.PixelShuffle(scale)
        self.refine = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
        )
        # ICNR init for shuffle conv
        icnr_(self.upconv.weight, scale=scale)
        if self.upconv.bias is not None:
            nn.init.zeros_(self.upconv.bias)

    def forward(self, x):
        x = self.ps(self.upconv(x))
        x = self.refine(x)
        return x

# ---------------------------
# Extra Up head (32→64→128→(1×2)→128×256) + edge branch
# ---------------------------
class ExtraUpSharper(nn.Module):
    def __init__(self, feat_ch, out_ch, rrdb_per_stage=1):
        super().__init__()
        blocks1 = [RRDB(feat_ch) for _ in range(rrdb_per_stage)]
        blocks2 = [RRDB(feat_ch) for _ in range(rrdb_per_stage)]

        self.up1 = UpShuffle(feat_ch, feat_ch, 2)   # 32→64
        self.b1 = nn.Sequential(*blocks1)

        self.up2 = UpShuffle(feat_ch, feat_ch, 2)   # 64→128
        self.b2 = nn.Sequential(*blocks2)

        # width only 2x (128×128 → 128×256)
        self.wide_up = nn.Upsample(scale_factor=(1, 2), mode='nearest')
        self.post = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(feat_ch, out_ch, 3, 1, 1),
        )
        # Edge/High-frequency head (shallow correction)
        self.edge_head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, 1, 1),
            RRDB(feat_ch),
            nn.Conv2d(feat_ch, out_ch, 1, 1, 0)
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.b1(x)
        x = self.up2(x)
        x = self.b2(x)
        x = self.wide_up(x)
        x_main = self.post(x)
        x_edge = self.edge_head(x)
        return x_main + x_edge

# ===========================
# Main: ComplexLargeUNetDecoderOnly (refactored)
# ===========================
class ComplexLargeUNetDecoderOnly(nn.Module):
    """
    Input: latent (B, in_channels)  — starting from 1x1
    Output: (B, out_channels, 128, 256)  (same as before)
    Changes:
      - removed bilinear upsampling → unified to Conv+PixelShuffle(+ICNR)
      - removed RRDB(+channel attention) and BatchNorm from intermediate refinement
      - added edge correction branch to ExtraUp
      - default final activation is tanh (compatible with original). for sharper output, use use_tanh=False
    """
    def __init__(self, in_channels, out_channels, feat_ch=32, rrdb_per_stage=1, use_tanh=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_tanh = use_tanh

        act = nn.LeakyReLU(0.2, True)

        # -----------------------
        # 1x1 → 2x2 → 4x4 → 8x8 → 16x16 → 32x32
        # all upsampling done with PixelShuffle
        # -----------------------
        self.up1 = UpShuffle(in_channels, 128, 2)    # 1→2
        self.ref1 = RRDB(128)

        self.up2 = UpShuffle(128, 64, 2)             # 2→4
        self.ref2 = RRDB(64)

        self.up3 = UpShuffle(64, 32, 2)              # 4→8
        self.ref3 = RRDB(32)

        self.up4 = UpShuffle(32, 16, 2)              # 8→16
        self.ref4 = RRDB(16)

        self.final_up = UpShuffle(16, feat_ch, 2)    # 16→32
        self.final_ref = RRDB(feat_ch)

        # -----------------------
        # 32x32 → 64x64 → 128x128 → 128x256 (+ edge head)
        # -----------------------
        self.extra_up = ExtraUpSharper(feat_ch, out_channels, rrdb_per_stage=rrdb_per_stage)

        # -----------------------
        # skip connections (latent → spatial tensor) + 1x1 conv for channel alignment
        # -----------------------
        self.skip1_proj = nn.Linear(in_channels, 128 * 2 * 2)
        self.skip2_proj = nn.Linear(in_channels, 64 * 4 * 4)
        self.skip3_proj = nn.Linear(in_channels, 32 * 8 * 8)
        self.skip4_proj = nn.Linear(in_channels, 16 * 16 * 16)

        self.skip1_conv = nn.Conv2d(128 + 128, 128, kernel_size=1)
        self.skip2_conv = nn.Conv2d(64 + 64, 64, kernel_size=1)
        self.skip3_conv = nn.Conv2d(32 + 32, 32, kernel_size=1)
        self.skip4_conv = nn.Conv2d(16 + 16, 16, kernel_size=1)

        self.tanh = nn.Tanh()

        # -----------------------
        # Init
        # -----------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # conv before PixelShuffle is initialized with ICNR in UpShuffle
                if not hasattr(m, "_is_icnr") and m.kernel_size != (0, 0):
                    init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, latent):
        # latent: (B, C)
        B = latent.size(0)

        # prepare skips
        skip1 = self.skip1_proj(latent).view(B, 128, 2, 2)
        skip2 = self.skip2_proj(latent).view(B, 64, 4, 4)
        skip3 = self.skip3_proj(latent).view(B, 32, 8, 8)
        skip4 = self.skip4_proj(latent).view(B, 16, 16, 16)

        # reshape to 1x1
        x = latent.view(B, self.in_channels, 1, 1)

        # 1→2
        x = self.up1(x)
        x = self.ref1(x)
        x = self.skip1_conv(torch.cat([x, skip1], dim=1))

        # 2→4
        x = self.up2(x)
        x = self.ref2(x)
        x = self.skip2_conv(torch.cat([x, skip2], dim=1))

        # 4→8
        x = self.up3(x)
        x = self.ref3(x)
        x = self.skip3_conv(torch.cat([x, skip3], dim=1))

        # 8→16
        x = self.up4(x)
        x = self.ref4(x)
        x = self.skip4_conv(torch.cat([x, skip4], dim=1))

        # 16→32
        x = self.final_up(x)
        x = self.final_ref(x)

        # 32→64→128→128×256 (+edge)
        x = self.extra_up(x)

        if self.use_tanh:
            x = self.tanh(x)
        return x





class MiddleUNetDecoderOnly(nn.Module):
    def __init__(self, in_channels, out_channels, feat_ch=32):
        super().__init__()

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # 1x1 → 2x2 → 4x4 → 8x8 → 16x16 → 32x32 (same up to here)
        self.up1 = up_block(in_channels, 128)   # 1 → 2
        self.up2 = up_block(128, 64)            # 2 → 4
        self.up3 = up_block(64, 32)             # 4 → 8
        self.up4 = up_block(32, 16)             # 8 → 16

        self.final_up = up_block(16, feat_ch)   # 16 → 32 (keep features)

        # 32x32 → 64x64 → 100x100 → 100x200
        self.extra_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32→64
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(size=(100, 100), mode='bilinear', align_corners=False),  # 64→80
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=(1, 2), mode='bilinear', align_corners=False),  # 100x100 → 100x200 (width only 2x)
            nn.Conv2d(feat_ch, out_channels, kernel_size=3, padding=1),  # project to out_channels only at the end
        )
        

        self.skip1_proj = nn.Linear(in_channels, 128 * 2 * 2)
        self.skip2_proj = nn.Linear(in_channels, 64 * 4 * 4)
        self.skip3_proj = nn.Linear(in_channels, 32 * 8 * 8)
        self.skip4_proj = nn.Linear(in_channels, 16 * 16 * 16)

        self.skip1_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.skip2_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.skip3_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.skip4_conv = nn.Conv2d(32, 16, kernel_size=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    
    def forward(self, latent):
        skip1 = self.skip1_proj(latent).view(latent.size(0), 128, 2, 2)
        skip2 = self.skip2_proj(latent).view(latent.size(0), 64, 4, 4)
        skip3 = self.skip3_proj(latent).view(latent.size(0), 32, 8, 8)
        skip4 = self.skip4_proj(latent).view(latent.size(0), 16, 16, 16)

        x = latent.view(latent.size(0), latent.size(1), 1, 1)

        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.skip1_conv(x)
    
        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.skip2_conv(x)

        x = self.up3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.skip3_conv(x)

        x = self.up4(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.skip4_conv(x)
        
        x = self.final_up(x)
        x = self.extra_up(x)

        return x

class UNetDecoderOnly(nn.Module):
    def __init__(self, in_channels, out_channels, feat_ch=32):
        super().__init__()

        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2)
            )
        

        # Upsampling blocks
        self.up1 = up_block(in_channels, 128)  # 1x1 → 2x2
        self.up2 = up_block(128, 64)           # 2x2 → 4x4
        self.up3 = up_block(64, 32)            # 4x4 → 8x8
        self.up4 = up_block(32, 16)            # 8x8 → 16x16
        self.final_up = up_block(16, feat_ch) #out_channels) # feat_ch # 16x16 → 32x32
        '''
        # Final resolution upsampling (32x32 → 64x64 → 64x128)
        self.extra_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),      # 32x32 → 64x64
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Upsample(size=(64, 128), mode='bilinear', align_corners=True),      # 64x64 → 64x128
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        '''
        # first Conv in/out of extra_up is also feat_ch
        
        self.extra_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32→64
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(size=(64, 128), mode='bilinear', align_corners=False),  # 64→64x128
            nn.Conv2d(feat_ch, out_channels, kernel_size=3, padding=1),        # project to out_channels only at the final layer
        )
        
        

        # Skip connections (project latent vector)
        self.skip1_proj = nn.Linear(in_channels, 128 * 2 * 2)
        self.skip2_proj = nn.Linear(in_channels, 64 * 4 * 4)
        self.skip3_proj = nn.Linear(in_channels, 32 * 8 * 8)
        self.skip4_proj = nn.Linear(in_channels, 16 * 16 * 16)

        self.skip1_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.skip2_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.skip3_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.skip4_conv = nn.Conv2d(32, 16, kernel_size=1)


        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, latent):
        skip1 = self.skip1_proj(latent).view(latent.size(0), 128, 2, 2)
        skip2 = self.skip2_proj(latent).view(latent.size(0), 64, 4, 4)
        skip3 = self.skip3_proj(latent).view(latent.size(0), 32, 8, 8)
        skip4 = self.skip4_proj(latent).view(latent.size(0), 16, 16, 16)

        x = latent.view(latent.size(0), latent.size(1), 1, 1)

        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.skip1_conv(x)
    
        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.skip2_conv(x)

        x = self.up3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.skip3_conv(x)

        x = self.up4(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.skip4_conv(x)
        
        x = self.final_up(x)
        x = self.extra_up(x)
        return x


class HalfUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_dim=256):
        super().__init__()
        if latent_dim >= 1024:
            # self.decoder = ComplexLargeUNetDecoderOnly(latent_dim, out_channels)
            self.decoder = LargeUNetDecoderOnly(latent_dim, out_channels)
        else:
            self.decoder = UNetDecoderOnly(latent_dim, out_channels)

    def forward(self, latent):
        out = self.decoder(latent)
        
        # only if single channel output is needed
        if out.size(1) == 1:
            return out.squeeze(1)
        return out



class ProgressiveMLPDecoder(nn.Module):
    """
    Lightweight version of the original ProgressiveMLPDecoder.
    - Reduced number of intermediate layers from 8 to 4.
    - Number of channels (dimensions) per layer is adjustable via the base_channels parameter.
    """
    def __init__(self, latent_dim, out_channels=1, base_channels=64, p_drop=0.1, use_layernorm=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.use_layernorm = use_layernorm

        # target flat dimension per stage
        def n(c, h, w): return c * h * w

        # reduced to 4 expansion stages with redesigned intermediate dimensions
        dims = [
            n(base_channels * 4, 8, 8),      # -> 8x8
            n(base_channels * 2, 16, 32),     # -> 16x32
            n(base_channels * 1, 64, 128),    # -> 64x128
            n(out_channels, 128, 256),        # -> 128x256 (final)
        ]

        blocks = []
        in_dim = latent_dim
        for i, out_dim in enumerate(dims):
            blocks += [
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(p_drop),
            ]
            in_dim = out_dim
            
        # remove activation function and dropout from the last layer
        blocks = blocks[:-2]
        self.mlp = nn.Sequential(*blocks)

        self.ln = nn.LayerNorm(latent_dim) if use_layernorm else nn.Identity()

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, latent):
        """
        latent: (B, D) or (B, 1, D) or (B, S, D)
        - if a sequence is given, only the first vector is used.
        """
        if latent.dim() == 3:
            latent = latent[:, 0, :]
        elif latent.dim() != 2:
            raise ValueError("Input latent must have 2 or 3 dimensions")

        z = self.ln(latent)
        x = self.mlp(z)
        B = x.size(0)
        x = x.view(B, self.out_channels, 128, 256)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_dim=256):
        super().__init__()
        self.decoder = ProgressiveMLPDecoder(latent_dim)
    def forward(self, latent):
        out = self.decoder(latent)
        return out