#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2023-09-11 10:50:39
# @Author: zzm

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from transformers.modeling_utils import CrossEntropyLoss
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP, GPT2PreTrainedModel

MAX_LENGTH = 77
EFFNET_OUT = 512

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False, layer_idx=None):
        super().__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)#torch.nn.MultiheadAttention(embed_dim = nx, num_heads = 1, dropout = 0.5) 
        # self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(4 * nx, config)
        
    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False):
        output_attn = self.attn(
            self.ln_1(x),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)

        x = x + a
        # print(self.ln_2(x).shape)
        m = self.mlp(self.ln_2(x))
        x = x + m
        
        # print("[x]:", type([x]))
        # print("output_attn[1:]:", type(output_attn[1:]))
        outputs = [x] + list(output_attn[1:])
        return outputs[0], outputs[1]

class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte


    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings


    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)
    
    def run_block(self, block, layer_past, attention_mask, head_mask, use_cache):
        def custom_forward(*inputs):
            x, present = block(
                inputs[0],
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
            )
            return x, present
        return custom_forward

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=True,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)
            
            hidden_states, present = checkpoint.checkpoint(
                self.run_block(block, layer_past, attention_mask, head_mask[i], use_cache),
                hidden_states
            )
            
            
            '''
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
            )
            hidden_states, present = outputs[:2]
            '''
            
            if use_cache is True:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)
    
class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config, hidden1=384, hidden2=256):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()
        
        #self.efficient = EfficientNet.from_pretrained(efficient, advprop=False)
        # self.efficient = EfficientNetCheck.from_pretrained(efficient, advprop=True)
        # self.efficient = clipmodel
        
        # MLP
        self.encoder1 = nn.Linear(config.n_embd + EFFNET_OUT, hidden1)
        self.decoder1 = nn.Linear(hidden1, config.n_embd)
        self.encoder2 = nn.Linear(config.n_embd, hidden2)
        self.decoder2 = nn.Linear(hidden2, config.n_embd)
        
        self.relu = nn.ReLU()
        
        nn.init.xavier_normal_(self.encoder1.weight, gain=0.1)
        nn.init.xavier_normal_(self.decoder1.weight, gain=0.1)
        nn.init.xavier_normal_(self.encoder2.weight, gain=0.1)
        nn.init.xavier_normal_(self.decoder2.weight, gain=0.1)
        
    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {"input_ids": input_ids, "image": kwargs["image"], "past": past, "use_cache": kwargs["use_cache"]}

    def forward(self, input_ids=None, fused_feature=None, past=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, use_cache=True):
        
        transformer_outputs = self.transformer(input_ids, past=past, attention_mask=attention_mask, 
                                               token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, 
                                               inputs_embeds=inputs_embeds, use_cache=use_cache)
        # print("transformer_outputs:", transformer_outputs[0].shape)
        transformer = transformer_outputs[0]
        # with torch.no_grad():
        #     image_feature = clipmodel.encode_image(image)
        #     sketch_feature = clipmodel.encode_sketch(sketch)
        #     image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        #     sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)
        #     fused_feature = clipmodel.feature_fuse(image_feature, sketch_feature)
        if MAX_LENGTH:
            efficient = fused_feature.unsqueeze(1).repeat(1, MAX_LENGTH, 1)
        else:
            efficient = fused_feature.unsqueeze(1).repeat(transformer.shape[0], transformer.shape[1], 1)
        
        # print("efficient:", efficient.shape)
        # print("transformer:", transformer.shape)
        latent = torch.cat((efficient, transformer), 2)
        
        encoded = self.relu(self.encoder1(latent))
        hidden_states = self.decoder1(encoded) + transformer
        
        encoded = self.relu(self.encoder2(hidden_states))
        hidden_states = self.decoder2(encoded) + hidden_states
        
        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
