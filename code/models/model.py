from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .layers import ResidualAttention

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        x_, attention_score = self.attention(self.ln_1(x))
        x = x + x_
        x = x + self.mlp(self.ln_2(x))
        return x, attention_score

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        attention_score = []
        for i in range(self.layers):
            x, score = self.resblocks[i](x)
            attention_score.append(score)
        return x, attention_score

class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, att_score = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        return x, att_score
    
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class MGABase(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 
                 weight_sharing: bool = False,
                 feature_fusion: str = 'avg',
                 num_class: int = 90
                 ):
        super().__init__()
        #set default to weight sharing
        if weight_sharing is None:
            weight_sharing = False
            
        self.weight_sharing = weight_sharing
        self.feature_fusion = feature_fusion
        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )
        if weight_sharing:
            self.visual2 = self.visual
        else:
            self.visual2 = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )
        
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.img_k_ratio = 0.1
        self.text_k_ratio = 0.2
        self.sketch_k_ratio = 0.1
        self.kw =  3
        self.kp = 3
        self.scale_cs = 0.07
        self.scale_cd = 0.01 
        self.block = ResidualAttention(num_layers=1,
                                       d_model=512,
                                       n_head=8,
                                       att_type='cross')

        self.bottleneck_image = nn.BatchNorm1d(512)
        self.bottleneck_image.bias.requires_grad_(False)
        self.bottleneck_image.apply(weights_init_kaiming)
        self.bottleneck_text = nn.BatchNorm1d(512)
        self.bottleneck_text.bias.requires_grad_(False)
        self.bottleneck_text.apply(weights_init_kaiming)
        self.bottleneck_sketch = nn.BatchNorm1d(512)
        self.bottleneck_sketch.bias.requires_grad_(False)
        self.bottleneck_sketch.apply(weights_init_kaiming)
        self.similarityNorm = nn.Softmax(dim=2)
        
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))
    
    def encode_sketch(self, image):
        return self.visual2(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, att_scores = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection
        return x, att_scores
    
    def feature_fuse(self, text_features, sketch_features):
        #mode = avg|max
        if self.feature_fusion == 'avg':
            fused_features = (text_features + sketch_features)/2
        else:
            raise Exception(f'Mode {self.feature_fusion} not yet supported')
        return fused_features

    def get_select(self, scores, feats, k_ratio):
        score = scores[-1][:, 0, :]
        temp = torch.zeros(score.size(0), 1)
        score_mask = (torch.cat((temp, torch.ones(score.size(0), score.size(1)-1)), dim=1)).to(score.device)
        score = score * score_mask
        feats_selected = []
        top_K = int(feats.size(1) * k_ratio)
        for b in range(feats.size(0)):
            _, idx = score[b].topk(score.size(1), largest=True, sorted=True)
            feats_selected.append(feats[b][idx[:top_K], :])
        feats_selected = torch.stack(feats_selected, dim=0)
        part = self.block(feats_selected, feats)
        return feats_selected, part

    def forward(self, image, text, sketch, cates, tokens, masks):
        image_feats, image_scores = self.encode_image(image)
        sketch_feats, sketch_scores = self.encode_sketch(sketch)
        text_feats, text_scores = self.encode_text(text)
        # print(image_feats.shape, sketch_feats.shape, text_feats.shape)

        image_feat_global = image_feats[:, 0, :]
        sketch_feat_global = sketch_feats[:, 0, :]
        # text_feat_global = text_feats[:, 0, :]
        text_feat_global = text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)]

        image_feat_global = image_feat_global / image_feat_global.norm(dim=-1, keepdim=True)#self.bottleneck_image(image_feats[:, 0, :])
        sketch_feat_global = sketch_feat_global / sketch_feat_global.norm(dim=-1, keepdim=True)#self.bottleneck_sketch(sketch_feats[:, 0, :])
        text_feat_global = text_feat_global / text_feat_global.norm(dim=-1, keepdim=True)#self.bottleneck_text(text_feats[:, 0, :])

        text_masks = torch.zeros_like(tokens).masked_fill_(tokens == 0, 1).bool()

        image_feats_selected, image_part = self.get_select(image_scores, image_feats, self.img_k_ratio)
        sketch_feats_selected, sketch_part = self.get_select(sketch_scores, sketch_feats, self.sketch_k_ratio)
        text_feats_selected, text_part = self.get_select(text_scores, text_feats, self.text_k_ratio)
        # print(image_part.shape, text_part.shape, sketch_part)

        if self.training:
            # ---------------------------
            # Cross-Similarity
            # ---------------------------
            G_img_token = image_feats[:, 0, :].unsqueeze(1)
            L_img_token = image_feats_selected
            B = L_img_token.size(0)
            G_sketch_token = sketch_feats[:, 0, :].unsqueeze(1)
            L_sketch_token = sketch_feats_selected
            G_text_token = text_feats[:, 0, :].unsqueeze(1)
            L_text_token = text_feats_selected

            G_img_token_norm = G_img_token / G_img_token.norm(dim=-1, keepdim=True)
            L_img_token_norm = L_img_token / L_img_token.norm(dim=-1, keepdim=True)
            G_sketch_token_norm = G_sketch_token / G_sketch_token.norm(dim=-1, keepdim=True)
            L_sketch_token_norm = L_sketch_token / L_sketch_token.norm(dim=-1, keepdim=True)
            G_text_token_norm = G_text_token / G_text_token.norm(dim=-1, keepdim=True)
            L_text_token_norm = L_text_token / L_text_token.norm(dim=-1, keepdim=True)

            # # image-word sim
            # G_img_token_norm_l = G_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)
            # L_text_token_norm_r = L_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)

            # sim_iw = torch.matmul(G_img_token_norm_l, L_text_token_norm_r.transpose(-2, -1)) / self.scale_cs
            # weight_iw = F.softmax(sim_iw, dim=-1)
            # sim_iw = torch.mul(sim_iw, weight_iw)
            # sim_iw = torch.sum(sim_iw, dim=-1).squeeze()

            # # patch-text sim
            # L_img_token_norm_l = L_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)
            # G_text_token_norm_r = G_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)

            # sim_pt = torch.matmul(L_img_token_norm_l, G_text_token_norm_r.transpose(-2, -1)) / self.scale_cs
            # weight_pt = F.softmax(sim_pt, dim=2)
            # sim_pt = torch.mul(sim_pt, weight_pt)
            # sim_pt = torch.sum(sim_pt, dim=2).squeeze()

            # sim_cs = (sim_iw + sim_pt) / 2

            # ---------------------------
            # Correspondence Discovery
            # ---------------------------
            L_img_token1 = image_feats_selected[:, :, :]
            L_text_token1 = text_feats_selected[:, :, :]
            L_img_token_norm1 = L_img_token1 / L_img_token1.norm(dim=-1, keepdim=True)
            L_text_token_norm1 = L_text_token1 / L_text_token1.norm(dim=-1, keepdim=True)
            _b, _n, _c = L_text_token1.shape
            vidwordSim = torch.bmm(L_img_token_norm1, L_text_token_norm1.permute(0, 2, 1))
            vidwordSim = self.similarityNorm(vidwordSim)
            # ---------- posWord -------
            _, idxWord = vidwordSim.topk(self.kw, dim=2, largest=True, sorted=True)
            posWord = []
            for _batch in range(idxWord.shape[0]):
                posWord.append(L_text_token1[_batch, idxWord[_batch], :])
            posWord = torch.stack(posWord)
            posWord = torch.mean(posWord, dim=2)
            posWord_norm = posWord / posWord.norm(dim=-1, keepdim=True)
            # ---------- posClip -------
            _, idxVid = vidwordSim.topk(self.kp, dim=1, largest=True, sorted=True)
            posClip = []
            for _batch in range(idxVid.shape[0]):
                posClip.append(L_img_token1[_batch, idxVid[_batch], :])
            posClip = torch.stack(posClip)
            posClip = torch.mean(posClip, dim=1)
            posClip_norm = posClip / posClip.norm(dim=-1, keepdim=True)

            # piexl_word sim B*B*Select_num*emb_dim
            L_img_token_norm_l1 = L_img_token_norm1.unsqueeze(1).repeat(1, _b, 1, 1)
            posWord_norm_r = posWord_norm.unsqueeze(0).repeat(_b, 1, 1, 1)
            posClip_norm_l = posClip_norm.unsqueeze(1).repeat(1, _b, 1, 1)
            L_text_token_norm_r1 = L_text_token_norm1.unsqueeze(0).repeat(_b, 1, 1, 1)

            #B*B*select_num*select_num
            sim_pw0 = torch.matmul(L_img_token_norm_l1, posWord_norm_r.transpose(-2, -1)) / self.scale_cd
            # print("sim_pw0", sim_pw0.shape)
            sim_pw0 = torch.diagonal(sim_pw0, dim1=-2, dim2=-1)
            #B*B
            sim_pw0 = torch.mean(sim_pw0, dim=2)

            sim_pw1 = torch.matmul(posClip_norm_l, L_text_token_norm_r1.transpose(-2, -1)) / self.scale_cd
            sim_pw1 = torch.diagonal(sim_pw1, dim1=-2, dim2=-1)
            sim_pw1 = torch.mean(sim_pw1, dim=2)

            sim_cd = (sim_pw0 + sim_pw1) / 2
            # print("sim_cd.shape:::",sim_cd.shape)        
        return image_feat_global, text_feat_global, sketch_feat_global

    
    