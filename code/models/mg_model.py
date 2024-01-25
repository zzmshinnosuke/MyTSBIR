
from .model import MGABase
from .AsymmetricLoss import AsymmetricLossOptimized
from .gpt import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Config

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import pytorch_lightning as pl
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors

class ClassModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).cuda()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim).cuda()
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out

class MultiGrainModel(pl.LightningModule):
    def __init__(self, config):
        super(MultiGrainModel, self).__init__()
        self.config = config
        self.get_model()   
        self.save_hyperparameters()

    def get_model(self):
        self.use_cuda = True       
        with open(self.config.model_config, 'r') as f:
            model_info = json.load(f)
        self.model = MGABase(**model_info)
        if self.config.pre_model:
            checkpoints = torch.load(self.config.pre_model, map_location='cpu')
            sd = checkpoints["state_dict"]
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=False)
        self.gptmodel = GPT2LMHeadModel(GPT2Config(vocab_sizea = 49408,n_layer=6,n_head=8,n_ctx=77,)).cuda()
        self.classmodel = ClassModel(512, 256, self.config.output_dim)
        self.ASL_Loss = AsymmetricLossOptimized()
    
    def configure_optimizers(self):
        # optimizer =  torch.optim.Adam(self.parameters(), lr = self.config.learning_rate, weight_decay = 0.0005)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, 
                                                factor=self.config.learning_rate_factor,
                                                patience=self.config.learning_rate_decay_frequency),
                "monitor": "val/loss",
                "frequency": self.config.learning_rate_decay_frequency
            },
        }

    def forward(self, image, text, sketch, cates, tokens, masks):
        image_feat, text_feat, sketch_feat = self.model(image, text, sketch, cates, tokens, masks)
        if self.config.input_type == "T":
            fused_feat = text_feat
        elif self.config.input_type == "S":
            fused_feat = sketch_feat
        else:
            fused_feat = self.model.feature_fuse(text_feat, sketch_feat)
        loss_contra = self.get_contrastive_loss(image_feat, fused_feat)
        # loss_h = self.get_hard_loss(image_feat,fused_feat, image_embeds, text_embeds, sketch_embeds)
        loss_class = self.get_class_loss(image_feat, text_feat, sketch_feat, cates)
        loss_gpt = self.get_gpt_loss(tokens, fused_feat, masks)
        return image_feat, fused_feat, loss_contra, loss_class, loss_gpt

    def fetch_batch(self, batch):
        sketch_id = batch[0]
        image = batch[1]
        sketch = batch[2]
        text = batch[3]
        cates = batch[4]
        tokens = batch[5]
        masks = batch[6]
        return sketch_id, image, text, sketch, cates, tokens, masks
    
    def training_step(self, train_batch, batch_idx):
        sketch_id, image, text, sketch, cates, tokens, masks = self.fetch_batch(train_batch)
        _, _, loss_contra, loss_class, loss_gpt = self.forward(image, text, sketch, cates, tokens, masks)
        loss = (10 * loss_class + loss_gpt + 100 * loss_contra) / 111
        # loss = loss_contra
        self.log('train/loss_contra', loss_contra, on_step=True, on_epoch=True,batch_size=self.config.batch_size)
        self.log('train/loss_class', loss_class, on_step=True, on_epoch=True,batch_size=self.config.batch_size)
        self.log('train/loss_gpt', loss_gpt, on_step=True, on_epoch=True,batch_size=self.config.batch_size)
        self.log('train/loss', loss, on_step=True, on_epoch=True,batch_size=self.config.batch_size)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        sketch_id, image, text, sketch, cates, tokens, masks = self.fetch_batch(val_batch)
        image_feat, fused_feat, loss_contra, loss_class, loss_gpt = self.forward(image, text, sketch, cates, tokens, masks)
        loss = (10 * loss_class + loss_gpt + 100 * loss_contra) / 111
        self.log('val/loss_contra', loss_contra, on_epoch=True,batch_size=self.config.batch_size)
        self.log('val/loss_class', loss_class, on_epoch=True,batch_size=self.config.batch_size)
        self.log('val/loss_gpt', loss_gpt, on_epoch=True,batch_size=self.config.batch_size)
        self.log('val/loss', loss, on_epoch=True,batch_size=self.config.batch_size)
        return image_feat, fused_feat, sketch_id

    def test_step(self, test_batch, batch_idx):
        sketch_id, image, text, sketch, cates, tokens, masks = self.fetch_batch(test_batch)
        image_feat, fused_feat, _, _, _ = self.forward(image, text, sketch, cates, tokens, masks)
        return image_feat, fused_feat, sketch_id

    def test_epoch_end(self, outs):
        Len = len(outs)
        image_feature_all = torch.cat([outs[i][0] for i in range(Len)]) # shape: B x dim x 7 x 7
        fused_feature_all = torch.cat([outs[i][1] for i in range(Len)]) # shape: B x dim x 7 x 7
          
        image_feature_all = image_feature_all.cpu().detach().numpy()
        fused_feature_all = fused_feature_all.cpu().detach().numpy()
        
        img_feats = np.stack(image_feature_all)
        fused_feats = np.stack(fused_feature_all)

        nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine').fit(img_feats)
        distances, indices = nbrs.kneighbors(fused_feats)
        recall1, recall5, recall10 = 0,0,0
        for index,indice in enumerate(indices):
            if index in indice:
                recall10 += 1
            if index in indice[:5]:
                recall5 += 1
            if index in indice[:1]:
                recall1 += 1
        print("top1",round(recall1 / len(img_feats), 4))
        print("top5",round(recall5 / len(img_feats), 4))
        print("top10",round(recall10 / len(img_feats), 4))
        
    def training_epoch_end(self, outs):
        self.log('learning-rate', self.optimizers().param_groups[0]['lr'], batch_size=self.config.batch_size)
        
    def validation_epoch_end(self, outs):
        Len = len(outs)
        image_feature_all = torch.cat([outs[i][0] for i in range(Len)]) # shape: B x dim x 7 x 7
        fused_feature_all = torch.cat([outs[i][1] for i in range(Len)]) # shape: B x dim x 7 x 7
            
        image_feature_all = image_feature_all.cpu().detach().numpy()
        fused_feature_all = fused_feature_all.cpu().detach().numpy()

        img_feats = np.stack(image_feature_all)
        fused_feats = np.stack(fused_feature_all)
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine').fit(img_feats)
        distances, indices = nbrs.kneighbors(fused_feats)
        recall1, recall5, recall10 = 0,0,0
        for index,indice in enumerate(indices):
            if index in indice:
                recall10 += 1
            if index in indice[:5]:
                recall5 += 1
            if index in indice[:1]:
                recall1 += 1
        top1 = round(recall1 / len(img_feats), 4)
        top5 = round(recall5 / len(img_feats), 4)
        top10 = round(recall10 / len(img_feats), 4)
        self.log('val/top1', top1, on_epoch=True,batch_size=self.config.batch_size)
        self.log('val/top5', top5, on_epoch=True,batch_size=self.config.batch_size)
        self.log('val/top10', top10, on_epoch=True,batch_size=self.config.batch_size)

    def get_contrastive_loss(self, image_feat, fused_feat):
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_feat @ fused_feat.t()
        bsz = image_feat.shape[0]
        labels = torch.arange(bsz, device=image_feat.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2

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