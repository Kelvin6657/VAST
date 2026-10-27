# SwinUNETR backbone with LoRA and segmentation/classification decoders.
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.swin_unetr import SwinTransformer, WindowAttention
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
import math

class _LoRA_qkv(nn.Module):
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v, r: int, alpha: float = 1.0):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.r = r
        self.alpha = alpha
        self.scale = alpha / float(r)

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += self.scale * new_q
        qkv[:, :, -self.dim:] += self.scale * new_v
        return qkv


class SwinUNETR_Seg(nn.Module):
    """
    EchoCare-style SwinUNETR (2D) segmentation backbone:
    - encoder: SwinTransformer (in_chans=3)
    - decoder: UNETR-style blocks
    """
    def __init__(
        self,
        seg_num_classes: int,
        ssl_checkpoint: str = None,
        in_chans: int = 3,
        r: int = 5,
        alpha: float = 5.0,
        train_lora_only: bool = True,
    ):
        super().__init__()

        self.Swin_encoder = SwinTransformer(
            in_chans=in_chans,
            embed_dim=128,
            window_size=[8, 8],
            patch_size=[2, 2],
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            use_checkpoint=True,
            spatial_dims=2,
            use_v2=True,
        )

     
        if ssl_checkpoint is not None:
            model_dict = torch.load(ssl_checkpoint, map_location="cpu")
            if isinstance(model_dict, dict) and "state_dict" in model_dict:
                model_dict = model_dict["state_dict"]
            # Remove mask_token as mentioned in EchoCare
            if isinstance(model_dict, dict) and "mask_token" in model_dict:
                model_dict.pop("mask_token")
            msg = self.Swin_encoder.load_state_dict(model_dict, strict=False)
            print("missing:", len(msg.missing_keys))
            print("unexpected:", len(msg.unexpected_keys))
            print("example missing:", msg.missing_keys[:20])
            print("example unexpected:", msg.unexpected_keys[:20])

            print("Using pretrained self-supervised Swin backbone weights!")
            
        for name, module in self.Swin_encoder.named_modules():
            if isinstance(module, WindowAttention):
                old_qkv = module.qkv
                dim = old_qkv.in_features

                # Initialize LoRA 
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                
                module.qkv = _LoRA_qkv(
                    module.qkv,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                    r=r,
                    alpha=alpha,   # Scale the LoRA query/value residuals.
                )

        
        self.reset_parameters()

        for name, p in self.Swin_encoder.named_parameters():
            if train_lora_only:
                p.requires_grad = "linear_a" in name or "linear_b" in name
            else:
                p.requires_grad = True

        # UNETR decoder blocks (consistent with EchoCare code)
        spatial_dims = 2
        encode_feature_size = 128
        decode_feature_size = 64
        norm_name = "instance"

        # Note: encoder1 input channels is in_chans (=3)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=encode_feature_size,
            out_channels=decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * encode_feature_size,
            out_channels=2 * decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * encode_feature_size,
            out_channels=4 * decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * encode_feature_size,
            out_channels=8 * decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # Bottleneck adaptation (hidden_states_out[4] has 16*encode_feature_size=2048 channels)
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * encode_feature_size,
            out_channels=16 * decode_feature_size,  # 16*64 = 1024
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * decode_feature_size,
            out_channels=8 * decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * decode_feature_size,
            out_channels=4 * decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * decode_feature_size,
            out_channels=2 * decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * decode_feature_size,
            out_channels=decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=decode_feature_size,
            out_channels=decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=decode_feature_size,
            out_channels=seg_num_classes
        )

        # Bottleneck dimension for UniMatch head
        self.bottleneck_dim = 16 * decode_feature_size  # 1024

    def encode(self, x3):
        """
        Return skip features + bottleneck feature needed for UNETR decoder:
        enc0..enc4, dec4
        """
        hidden_states_out = self.Swin_encoder(x3)  # list: [B,C,H,W]...
        enc0 = self.encoder1(x3)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        enc4 = self.encoder5(hidden_states_out[3])
        dec4 = self.encoder10(hidden_states_out[4])  # bottleneck (B,1024,h,w)
        return enc0, enc1, enc2, enc3, enc4, dec4

    def decode(self, enc0, enc1, enc2, enc3, enc4, dec4):
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, _LoRA_qkv):
                nn.init.kaiming_uniform_(m.linear_a_q.weight, a=math.sqrt(5))
                nn.init.zeros_(m.linear_b_q.weight)
                nn.init.kaiming_uniform_(m.linear_a_v.weight, a=math.sqrt(5))
                nn.init.zeros_(m.linear_b_v.weight)


class Echocare_UniMatch(nn.Module):
    """
    Joint segmentation and multi-label classification using a SwinUNETR backbone.

    The forward method returns segmentation and classification logits. With
    feature perturbation enabled, each output is a pair of ordinary and
    feature-perturbed logits.
    """
    def __init__(
        self,
        in_chns: int,
        seg_class_num: int,
        cls_class_num: int,
        view_num_classes: int = 4,
        ssl_checkpoint: str = None,
        train_lora_only: bool = True,
    ):
        super().__init__()

        if in_chns == 3:
            self.in_adapter = nn.Identity()
            in_chans = 3
        else:
            self.in_adapter = nn.Conv2d(1, 3, 1, bias=False)
            with torch.no_grad():
                self.in_adapter.weight.zero_()
                self.in_adapter.weight[:, 0, 0, 0] = 1.0   # Copy to 3 channels
            in_chans = 3

        self.seg_net = SwinUNETR_Seg(
            seg_num_classes=seg_class_num,
            ssl_checkpoint=ssl_checkpoint,
            in_chans=in_chans,
            train_lora_only=train_lora_only,
        )

        bottleneck_dim = self.seg_net.bottleneck_dim  # 1024
        self.cls_head_input_dim = bottleneck_dim * 2
        hidden_dim = 1024  # Classification head hidden dimension.

        # Multi-label classification head (outputs logits -> sigmoid)
        self.cls_decoder = nn.Sequential(
            nn.Linear(self.cls_head_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Classification feature dropout.
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, cls_class_num),
        )

        self.fp_dropout = nn.Dropout2d(0.5)

    def _pool_embed(self, feat_bchw: torch.Tensor) -> torch.Tensor:
        # feat: (B,C,H,W) -> (B,C)
        return F.adaptive_avg_pool2d(feat_bchw, 1).flatten(1)

    def forward(self, x, need_fp: bool = False):
        """
        Returns:
          - need_fp=False: seg_logits (B,C,H,W), cls_logits (B,K)
          - need_fp=True: (seg_logits, seg_logits_fp), (cls_logits, cls_logits_fp)
        """
        x3 = self.in_adapter(x)

        enc0, enc1, enc2, enc3, enc4, dec4 = self.seg_net.encode(x3)

        if need_fp:
            # Concatenate ordinary and dropout-perturbed features along the batch axis.
            p_enc0 = torch.cat([enc0, self.fp_dropout(enc0)], dim=0)
            p_enc1 = torch.cat([enc1, self.fp_dropout(enc1)], dim=0)
            p_enc2 = torch.cat([enc2, self.fp_dropout(enc2)], dim=0)
            p_enc3 = torch.cat([enc3, self.fp_dropout(enc3)], dim=0)
            p_enc4 = torch.cat([enc4, self.fp_dropout(enc4)], dim=0)
            p_dec4 = torch.cat([dec4, self.fp_dropout(dec4)], dim=0)

            seg_logits = self.seg_net.decode(p_enc0, p_enc1, p_enc2, p_enc3, p_enc4, p_dec4)

            feat_avg = F.adaptive_avg_pool2d(p_dec4, 1).flatten(1)
            feat_max = F.adaptive_max_pool2d(p_dec4, 1).flatten(1)

            embed = torch.cat([feat_avg, feat_max], dim=1)
            cls_logits = self.cls_decoder(embed)
            cls_prob = cls_logits

            return seg_logits.chunk(2), cls_prob.chunk(2)

        # normal
        seg_logits = self.seg_net.decode(enc0, enc1, enc2, enc3, enc4, dec4)

        feat_avg = F.adaptive_avg_pool2d(dec4, 1).flatten(1)
        feat_max = F.adaptive_max_pool2d(dec4, 1).flatten(1)

        embed = torch.cat([feat_avg, feat_max], dim=1)
        cls_logits = self.cls_decoder(embed)
        cls_prob = cls_logits

        return seg_logits, cls_prob
