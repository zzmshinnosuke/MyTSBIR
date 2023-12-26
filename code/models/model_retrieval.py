import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MGABase
from .AsymmetricLoss import AsymmetricLossOptimized
from .gpt import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Config

class ClassModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).cuda()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim).cuda()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        # self.sigmoid = nn.Softmax()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)
        out = self.relu(out)
        return out

class MGA(MGABase):
    def __init__(self, **model_info):
        super(MGA, self).__init__(**model_info)
        self.classmodel = ClassModel(512, 256, 40)
        self.ASL_Loss = AsymmetricLossOptimized()

        config = GPT2Config(
        vocab_sizea = 49408,
        n_layer=6,
        n_head=8,
        n_ctx=77,
        )
        self.gptmodel = GPT2LMHeadModel(config).cuda()

    def forward(self, image, text, sketch, cates, tokens, masks):
        image_feat, image_embeds = self.encode_image(image)
        sketch_feat, sketch_embeds = self.encode_sketch(sketch)
        text_feat, text_embeds = self.encode_text(text)

        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        sketch_feat = sketch_feat / sketch_feat.norm(dim=-1, keepdim=True)

        fused_feat = self.feature_fuse(text_feat, sketch_feat)

        loss_contra = self.get_contrastive_loss(image_feat, fused_feat)
        # loss_h = self.get_hard_loss(image_feat,fused_feat, image_embeds, text_embeds, sketch_embeds)
        loss_class = self.get_class_loss(image_feat, text_feat, sketch_feat, cates)
        loss_gpt = self.get_gpt_loss(tokens, fused_feat, masks)

        return loss_contra, loss_class, loss_gpt #, loss_h 
    
    def get_contrastive_loss(self, image_feat, fused_feat):
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feat @ fused_feat.t()
        bsz = image_feat.shape[0]
        labels = torch.arange(bsz, device=image_feat.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2
        
    def get_hard_loss(self, image_feat, fused_feat, image_embeds, text_embeds, sketch_embeds):
        pass

    def get_class_loss(self, image_feat, text_feat, sketch_feat, cates):
        logit_txt = self.classmodel(text_feat.float())
        logit_img = self.classmodel(image_feat.float())
        logit_sketch = self.classmodel(sketch_feat.float())
        bsz = image_feat.shape[0]
        Lc_loss_txt = self.ASL_Loss(logit_txt, cates)
        Lc_loss_img = self.ASL_Loss(logit_img, cates)
        Lc_loss_sketch = self.ASL_Loss(logit_sketch, cates)
        Lc_loss = (Lc_loss_txt + Lc_loss_img + Lc_loss_sketch) / (3 * bsz)
        return Lc_loss
    
    def get_gpt_loss(self, tokens, fused_feats, masks):
        Ld_loss, outputs, _ = self.gptmodel(tokens, fused_feats, labels=tokens, attention_mask=masks)
        return Ld_loss