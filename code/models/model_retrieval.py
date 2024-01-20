import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MGABase
from .AsymmetricLoss import AsymmetricLossOptimized
from .gpt import GPT2LMHeadModel
from .layers import ResidualAttention
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

class MGA(MGABase):
    def __init__(self, args, **model_info):
        super(MGA, self).__init__(**model_info)
        print(args)
        self.classmodel = ClassModel(512, 256, args.output_dim)
        self.ASL_Loss = AsymmetricLossOptimized()

        config = GPT2Config(
        vocab_sizea = 49408,
        n_layer=6,
        n_head=8,
        n_ctx=77,
        )
        self.gptmodel = GPT2LMHeadModel(config).cuda()

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
    
    def get_feat_norm(self, feat):
        return feat / feat.norm(dim=-1, keepdim=True)
    
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
        # print("feats_selected1",feats_selected.shape)
        # feats_selected = torch.cat((feats[:, 0, :].unsqueeze(1), feats_selected), dim=1)
        # print("feats_selected2",feats_selected.shape)
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

        fused_feat = self.feature_fuse(text_feat_global, sketch_feat_global)
        loss_contra = self.get_contrastive_loss(image_feat_global, fused_feat)
        # loss_h = self.get_hard_loss(image_feat,fused_feat, image_embeds, text_embeds, sketch_embeds)
        loss_class = self.get_class_loss(image_feat_global, text_feat_global, sketch_feat_global, cates)
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