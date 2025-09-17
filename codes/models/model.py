import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AdamW
# from allennlp.nn.util import batched_index_select
# from allennlp.modules import FeedForward
from tqdm import tqdm
import os
import torch.nn.functional as F
from utils import get_devices
from d2l import torch as d2l
from collections import defaultdict
import numpy as np
    
class ERModel(nn.Module):
    def __init__(self, encoder_class, args):
        super(ERModel, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        in_features = self.encoder.config.hidden_size
        self.sub_startlayer = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=in_features, out_features=1))
        self.sub_endlayer = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=in_features, out_features=1))
        self.obj_startlayer = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=in_features, out_features=1))
        self.obj_endlayer = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=in_features, out_features=1))

    def forward(self, input_ids, token_type_ids, attention_mask, input_ngram_ids=None, ngram_position_matrix=None,
                ngram_token_type_ids=None, ngram_attention_mask=None):
        
        outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]   # batch, seq, hidden
#         print(last_hidden_state.shape, self.sub_startlayer)
        sub_start_idx = self.sub_startlayer(last_hidden_state).sigmoid()
        sub_end_idx = self.sub_endlayer(last_hidden_state).sigmoid()
        obj_start_idx = self.obj_startlayer(last_hidden_state).sigmoid()
        obj_end_idx = self.obj_endlayer(last_hidden_state).sigmoid()

        return sub_start_idx.squeeze(-1), sub_end_idx.squeeze(-1), \
               obj_start_idx.squeeze(-1), obj_end_idx.squeeze(-1)

class REModel(nn.Module):
    def __init__(self, tokenizer, encoder_class, num_labels, args):
        super(REModel, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        self.encoder.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(in_features=self.encoder.config.hidden_size*2, out_features=num_labels))
        self.args = args

    def forward(self, input_ids, token_type_ids, attention_mask, flag, labels=None, input_ngram_ids=None, ngram_position_matrix=None,
                ngram_token_type_ids=None, ngram_attention_mask=None, mode='train'):
        device = input_ids.device

        
        outputs = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]   # batch, seq, hidden
        batch_size, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(batch_size, 2*hidden_size) # batch, 2*hidden
        # flag: batch, 2
        for i in range(batch_size):
            sub_start_idx, obj_start_idx = flag[i, 0], flag[i, 1]
            start_entity = last_hidden_state[i, sub_start_idx, :].view(hidden_size, )   # s_start: hidden,
            end_entity = last_hidden_state[i, obj_start_idx, :].view(hidden_size, )   # o_start: hidden,
            entity_hidden_state[i] = torch.cat([start_entity, end_entity], dim=-1)
        entity_hidden_state = entity_hidden_state.to(device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            cal_loss = self.cal_rdrop_loss if self.args.do_rdrop and mode=='train' else self.cal_loss
            return cal_loss(logits, labels), logits
        return logits
    
    def cal_loss(self, logits, labels):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, labels.view(-1))
    
    def cal_rdrop_loss(self, logits, labels):
        loss_ce = self.cal_loss(logits, labels)
        loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='mean') + \
                  F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='mean')
        return loss_ce + loss_kl / 4 * self.args.rdrop_alpha
    
# BIBM论文方法
class GPNER2Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER2Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 2
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    

    
class RawGlobalPointer(nn.Module):
    def __init__(self, hiddensize, ent_type_size, inner_dim, RoPE=True, tril_mask=True, do_rdrop=False, dropout=0):
        '''
        :param encoder: BERT
        :param ent_type_size: 实体数目
        :param inner_dim: 64
        '''
        super().__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hiddensize
        if do_rdrop:
            self.dense = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2))
        else:
            self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE
        self.trail_mask = tril_mask

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, last_hidden_state,  attention_mask):
        self.device = attention_mask.device
#         last_hidden_state = context_outputs[0]
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        # 排除下三角
        if self.trail_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

class EfficientGlobalPointer(nn.Module):
    def __init__(self, hiddensize, ent_type_size, inner_dim, RoPE=True, tril_mask=True):
        '''
        :param encoder: BERT
        :param ent_type_size: 实体数目
        :param inner_dim: 64
        '''
        super().__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hiddensize
        self.p_dense = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.q_dense = nn.Linear(self.inner_dim * 2, self.ent_type_size * 2)

        self.RoPE = RoPE
        self.trail_mask = tril_mask

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, last_hidden_state,  attention_mask):
        self.device = attention_mask.device
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]
        
        inputs = self.p_dense(last_hidden_state)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim ** 0.5
        bias = torch.einsum('bnh->bhn', self.q_dense(inputs)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        # 排除下三角
        if self.trail_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12

        return logits 
    
class GPLinkerModel(nn.Module):
    def __init__(self, encoder_class, args, schema):
        super(GPLinkerModel, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        if args.use_efficient_global_pointer == True:
            GlobalPointer = EfficientGlobalPointer
        else:
            GlobalPointer = RawGlobalPointer
        
        self.mention_detect = GlobalPointer(hiddensize=hiddensize, ent_type_size=2, inner_dim=args.inner_dim).to(args.device)#实体关系抽取任务默认不提取实体类型
        self.s_o_head = GlobalPointer(hiddensize=hiddensize, ent_type_size=len(schema), inner_dim=args.inner_dim, RoPE=False, tril_mask=False).to(args.device)
        self.s_o_tail = GlobalPointer(hiddensize=hiddensize, ent_type_size=len(schema), inner_dim=args.inner_dim, RoPE=False, tril_mask=False).to(args.device)

        
    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return mention_outputs, so_head_outputs, so_tail_outputs
    
class GPFilterModel(nn.Module):
    def __init__(self, encoder_class, args, schema):
        super(GPFilterModel, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        if args.use_efficient_global_pointer == True:
            GlobalPointer = EfficientGlobalPointer
        else:
            GlobalPointer = RawGlobalPointer
        
        self.s_o_head = GlobalPointer(hiddensize=hiddensize, ent_type_size=len(schema), inner_dim=args.inner_dim,
                                      RoPE=False, tril_mask=False, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        self.s_o_tail = GlobalPointer(hiddensize=hiddensize, ent_type_size=len(schema), inner_dim=args.inner_dim,
                                      RoPE=False, tril_mask=False, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

        
    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return so_head_outputs, so_tail_outputs
    
class GPFilterace05Model(nn.Module):
    def __init__(self, encoder_class, args, schema):
        super(GPFilterace05Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        if args.use_efficient_global_pointer == True:
            GlobalPointer = EfficientGlobalPointer
        else:
            GlobalPointer = RawGlobalPointer
        
        self.s_o_head = GlobalPointer(hiddensize=hiddensize, ent_type_size=len(schema), inner_dim=args.inner_dim,
                                      RoPE=False, tril_mask=False, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        self.s_o_tail = GlobalPointer(hiddensize=hiddensize, ent_type_size=len(schema), inner_dim=args.inner_dim,
                                      RoPE=False, tril_mask=False, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

        
    def forward(self, batch_token_ids, batch_mask_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids)[0]

        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return so_head_outputs, so_tail_outputs
    


    
class GPNERModel(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNERModel, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        if self.args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs

def masked_activate(X, valid_lens, mode='sigmoid'):
    """通过在最后一个轴上遮盖元素来执行 softmax 操作"""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    assert mode in ['softmax', 'sigmoid']
    
    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
#             print(valid_lens)
    else:
        # 只有在Transformer的decoder中的masked attention才会用到二维的valid_lens
        valid_lens = valid_lens.reshape(-1)
    # 在最后的轴上，被遮盖的元素使用一个非常大的负值替换，从而其 softmax (指数)输出为 0
    
    if mode == 'softmax':
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    elif mode == 'sigmoid':
#         print('attention计算方式：', mode)
        X = nn.functional.sigmoid(X)
        return d2l.sequence_mask(X.clone().reshape(-1, shape[-1]), valid_lens,value=0).reshape(shape)
    
class HTAttention(nn.Module):
    """可加性注意力
       用于编码头和尾的注意力
    """
    def __init__(self, key_size, query_size, num_hiddens, dropout, atten_mode, **kwargs):
        super(HTAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.atten_mode = atten_mode
        
    def forward(self, queries, keys, values, valid_lens):
        batch = keys.shape[0]
        seq = keys.shape[1]
        queries, keys = self.W_q(queries), self.W_k(keys)
#         print(queries.shape, keys.shape, values.shape)
        # 在维度扩展后，
        # `queries` 的形状：(`batch_size`, 查询的个数, 1, `num_hidden`)
        # `key` 的形状：(`batch_size`, 1, “键－值”对的个数, `num_hiddens`)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # `self.w_v` 仅有一个输出，因此从形状中移除最后那个维度。
        # `scores` 的形状：(`batch_size`, 查询的个数, “键-值”对的个数)
        scores = self.W_v(features).squeeze(-1)
        attention_weights = masked_activate(scores, valid_lens, self.atten_mode).reshape(batch, seq, 1)
#         print('attention_weights.shape:', attention_weights.shape)
        return values * attention_weights

#@save
class AdditiveAttention(nn.Module):
    """可加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # `queries` 的形状：(`batch_size`, 查询的个数, 1, `num_hidden`)
        # `key` 的形状：(`batch_size`, 1, “键－值”对的个数, `num_hiddens`)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # `self.w_v` 仅有一个输出，因此从形状中移除最后那个维度。
        # `scores` 的形状：(`batch_size`, 查询的个数, “键-值”对的个数)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_activate(scores, valid_lens, mode='softmax')
        # `values` 的形状：(`batch_size`, “键－值”对的个数, 值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)

class HTRawGlobalPointer(nn.Module):
    def __init__(self, hiddensize, ent_type_size, inner_dim, RoPE=True, tril_mask=True, do_rdrop=False, dropout=0):
        '''
        :param encoder: BERT
        :param ent_type_size: 实体数目
        :param inner_dim: 64
        '''
        super().__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hiddensize
        if do_rdrop:
            self.dense = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2))
        else:
            self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE
        self.trail_mask = tril_mask

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, last_hidden_state_head, last_hidden_state_tail,  attention_mask):
        self.device = attention_mask.device
        batch_size = last_hidden_state_head.size()[0]
        seq_len = last_hidden_state_head.size()[1]
        
        outputs = self.dense(last_hidden_state_head)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw = outputs[..., :self.inner_dim]
        
        outputs = self.dense(last_hidden_state_tail)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        kw = outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        # 排除下三角
        if self.trail_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5
    
# GPNERModel基础上加入HTAtten
class GPNER3Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER3Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 11
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        else:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
#         print('head_query.shape, tail_query.shape:', head_query.shape, tail_query.shape)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
            
            
        return mention_outputs
    
# GPNER9Model基础上加入HTAtten
class GPNER4Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER4Model, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        else:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        return mention_outputs
    
    
class GPNERACE05Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNERACE05Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        if args.with_type:
            entity_class_num = 7
        else:
            entity_class_num = 1
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids)[0]
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    

    
class GPNER9Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER9Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    

class GPNER2Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER2Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 2
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER5SubModel(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER5SubModel, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        if self.args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER5Sp2oModel(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER5Sp2oModel, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        if self.args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER6ObjModel(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER6ObjModel, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER6Op2sModel(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER6Op2sModel, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER7Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER7Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten', 'add_aver_atten', 'add_atten', 'atten_add', 'aver_atten_add', 'aver_atten']:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
#         print(batch_head_positions, batch_head_positions.shape)
#         print(batch_tail_positions, batch_tail_positions.shape)
#         print(outputs.shape)
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten3':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query + tail_query) / 2
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten4':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1]
                avery_embedding = entity_embedding.mean(dim=0).unsqueeze(0)
                avery_query.append(avery_embedding)
            avery_query = torch.cat(avery_query, dim=0).unsqueeze(1)
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten5':
            valid_lens = batch_mask_ids.sum(-1)
            avery_output = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1].unsqueeze(0)
                output = outputs[i].unsqueeze(0)
                valid_len = valid_lens[i].unsqueeze(0)
                entity_output = self.atten2(entity_embedding, output, output, valid_len)
                entity_output = entity_output.mean(dim=1)
                avery_output.append(entity_output)
            avery_output = torch.cat(avery_output, dim=0).unsqueeze(1)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add_atten2':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output + avery_query
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten_add':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            avery_query = (head_query + tail_query) / 2
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query+tail_query) / 2
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_aver_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten_add':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER8Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER8Model, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten', 'add_aver_atten', 'add_atten', 'atten_add', 'aver_atten_add', 'aver_atten']:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
#         print('head_query.shape:', head_query.shape)
#         print('args.prefix_merge_mode:', args.prefix_merge_mode)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
#             print('head_output.shape:',head_output.shape)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten3':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query + tail_query) / 2
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten4':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1]
                avery_embedding = entity_embedding.mean(dim=0).unsqueeze(0)
                avery_query.append(avery_embedding)
            avery_query = torch.cat(avery_query, dim=0).unsqueeze(1)
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten5':
            valid_lens = batch_mask_ids.sum(-1)
            avery_output = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1].unsqueeze(0)
                output = outputs[i].unsqueeze(0)
                valid_len = valid_lens[i].unsqueeze(0)
                entity_output = self.atten2(entity_embedding, output, output, valid_len)
                entity_output = entity_output.mean(dim=1)
                avery_output.append(entity_output)
            avery_output = torch.cat(avery_output, dim=0).unsqueeze(1)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add_atten2':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output + avery_query
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten_add':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            avery_query = (head_query + tail_query) / 2
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'aver_atten':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query+tail_query) / 2
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add_aver_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten_add':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPFilter78Model(nn.Module):
    def __init__(self, encoder_class, args, schema):
        super(GPFilter78Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        if args.use_efficient_global_pointer == True:
            GlobalPointer = EfficientGlobalPointer
        else:
            GlobalPointer = RawGlobalPointer
        
        self.s_o_head = GlobalPointer(hiddensize=hiddensize, ent_type_size=len(schema), inner_dim=args.inner_dim,
                                      RoPE=False, tril_mask=False, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        self.s_o_tail = GlobalPointer(hiddensize=hiddensize, ent_type_size=len(schema), inner_dim=args.inner_dim,
                                      RoPE=False, tril_mask=False, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

        
    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return so_head_outputs, so_tail_outputs
    
class GPNER17Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER17Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 44
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        if self.args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER18Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER18Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 44
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class CasRelModel(nn.Module):
    def __init__(self, encoder_class, args, tokenizer):
        super(CasRelModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hidden_size = self.encoder.config.hidden_size
        class_num = 44
        self.dense_sh = nn.Linear(hidden_size, 1)
        self.dense_st = nn.Linear(hidden_size, 1)
        self.dense_oh = nn.Linear(hidden_size, class_num)
        self.dense_ot = nn.Linear(hidden_size, class_num)
        
        

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, sub_head_in=None, sub_tail_in=None, tokens=None, id2class=None, text=None):
        args = self.args
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        
        
        if sub_head_in != None:
            shs_feature = self.dense_sh(outputs).sigmoid()
            sts_feature = self.dense_st(outputs).sigmoid()
            sh_feature = outputs[range(batch_size), sub_head_in] # batch, hidden
            st_feature = outputs[range(batch_size), sub_tail_in] # batch, hidden
            sub_feature = (sh_feature + st_feature) / 2
            outputs = outputs + sub_feature.unsqueeze(1)
            ohs_feature = self.dense_oh(outputs).sigmoid()
            ots_feature = self.dense_ot(outputs).sigmoid()
            return shs_feature, sts_feature, ohs_feature, ots_feature
        else:
            token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=args.max_length, truncation=True)["offset_mapping"]
            new_span, entities = [], []
            for i in token2char_span_mapping:
                if i[0] == i[1]:
                    new_span.append([])
                else:
                    if i[0] + 1 == i[1]:
                        new_span.append([i[0]])
                    else:
                        new_span.append([i[0], i[-1] - 1])
            
            # 预测时一次只有一条数据
            h_bar = args.h_bar
            t_bar = args.t_bar
            shs_feature = self.dense_sh(outputs).sigmoid().data.cpu().numpy()
            sts_feature = self.dense_st(outputs).sigmoid().data.cpu().numpy()
            sub_heads, sub_tails = np.where(shs_feature[0] > h_bar)[0], np.where(sts_feature[0] > t_bar)[0]
            subjects = []
            
            for sub_head in sub_heads:
                sub_tail = sub_tails[sub_tails >= sub_head]
                if len(sub_tail) > 0:
                    sub_tail = sub_tail[0]
#                     subject = tokens[sub_head: sub_tail+1]
                    subject = text[new_span[sub_head][0]:new_span[sub_tail][-1] + 1]
                    subjects.append((subject, sub_head, sub_tail))
#             print('tokens:', tokens)
#             print('subjects:', subjects)
            if subjects:
                triple_list = []
                
                batch_token_ids = batch_token_ids.repeat(len(subjects), 1)
                batch_mask_ids = batch_mask_ids.repeat(len(subjects), 1)
                batch_token_type_ids = batch_token_type_ids.repeat(len(subjects), 1)
                 
                sub_head_in, sub_tail_in = torch.tensor([sub[1:] for sub in subjects]).T.reshape((2, -1))
                sh_feature = outputs[range(batch_size), sub_head_in] # batch, hidden
                st_feature = outputs[range(batch_size), sub_tail_in] # batch, hidden
                sub_feature = (sh_feature + st_feature) / 2
                outputs = outputs + sub_feature.unsqueeze(1)
                ohs_feature = self.dense_oh(outputs).sigmoid().data.cpu().numpy()
                ots_feature = self.dense_ot(outputs).sigmoid().data.cpu().numpy()
                for i, subject in enumerate(subjects):
                    sub = subject[0]
                    obj_heads, obj_tails = np.where(ohs_feature[i] > h_bar), np.where(ots_feature[i] > t_bar)
                    for obj_head, rel_head in zip(*obj_heads):
                        for obj_tail, rel_tail in zip(*obj_tails):
                            if obj_head <= obj_tail and rel_head == rel_tail:
                                rel = id2class[rel_head]
#                                 obj = tokens[obj_head: obj_tail+1]
                                obj = text[new_span[obj_head][0]:new_span[obj_tail][-1] + 1]
                                triple_list.append((sub, rel, obj))
                                if args.do_so_1v1:
                                    break
                triple_set = set()
                for s, r, o in triple_list:
                    s = ''.join(s)
                    o = ''.join(o)
                    triple_set.add((s, r, o))
                return list(triple_set)
            else:
                return []
            
class GPNER11Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER11Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        else:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
#         print(batch_head_positions, batch_head_positions.shape)
#         print(batch_tail_positions, batch_tail_positions.shape)
#         print(outputs.shape)
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten3':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query + tail_query) / 2
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten4':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1]
                avery_embedding = entity_embedding.mean(dim=0).unsqueeze(0)
                avery_query.append(avery_embedding)
            avery_query = torch.cat(avery_query, dim=0).unsqueeze(1)
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten5':
            valid_lens = batch_mask_ids.sum(-1)
            avery_output = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1].unsqueeze(0)
                output = outputs[i].unsqueeze(0)
                valid_len = valid_lens[i].unsqueeze(0)
                entity_output = self.atten2(entity_embedding, output, output, valid_len)
                entity_output = entity_output.mean(dim=1)
                avery_output.append(entity_output)
            avery_output = torch.cat(avery_output, dim=0).unsqueeze(1)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add_atten2':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output + avery_query
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten_add':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            avery_query = (head_query + tail_query) / 2
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query+tail_query) / 2
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_aver_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten_add':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        
        
        return mention_outputs
    
class GPNER12Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER12Model, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        else:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
#         print('head_query.shape:', head_query.shape)
#         print('args.prefix_merge_mode:', args.prefix_merge_mode)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
#             print('head_output.shape:',head_output.shape)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten3':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query + tail_query) / 2
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten4':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1]
                avery_embedding = entity_embedding.mean(dim=0).unsqueeze(0)
                avery_query.append(avery_embedding)
            avery_query = torch.cat(avery_query, dim=0).unsqueeze(1)
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten5':
            valid_lens = batch_mask_ids.sum(-1)
            avery_output = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1].unsqueeze(0)
                output = outputs[i].unsqueeze(0)
                valid_len = valid_lens[i].unsqueeze(0)
                entity_output = self.atten2(entity_embedding, output, output, valid_len)
                entity_output = entity_output.mean(dim=1)
                avery_output.append(entity_output)
            avery_output = torch.cat(avery_output, dim=0).unsqueeze(1)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add_atten2':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output + avery_query
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten_add':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            avery_query = (head_query + tail_query) / 2
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'aver_atten':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query+tail_query) / 2
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add_aver_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten_add':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        return mention_outputs
    
class GPNER13Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER13Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        else:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
#         print(batch_head_positions, batch_head_positions.shape)
#         print(batch_tail_positions, batch_tail_positions.shape)
#         print(outputs.shape)
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten3':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query + tail_query) / 2
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten4':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1]
                avery_embedding = entity_embedding.mean(dim=0).unsqueeze(0)
                avery_query.append(avery_embedding)
            avery_query = torch.cat(avery_query, dim=0).unsqueeze(1)
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten5':
            valid_lens = batch_mask_ids.sum(-1)
            avery_output = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1].unsqueeze(0)
                output = outputs[i].unsqueeze(0)
                valid_len = valid_lens[i].unsqueeze(0)
                entity_output = self.atten2(entity_embedding, output, output, valid_len)
                entity_output = entity_output.mean(dim=1)
                avery_output.append(entity_output)
            avery_output = torch.cat(avery_output, dim=0).unsqueeze(1)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add_atten2':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output + avery_query
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten_add':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            avery_query = (head_query + tail_query) / 2
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query+tail_query) / 2
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_aver_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten_add':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        
        
        return mention_outputs
    
class GPNER14Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER14Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        else:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
#         print(batch_head_positions, batch_head_positions.shape)
#         print(batch_tail_positions, batch_tail_positions.shape)
#         print(outputs.shape)
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten3':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query + tail_query) / 2
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten4':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1]
                avery_embedding = entity_embedding.mean(dim=0).unsqueeze(0)
                avery_query.append(avery_embedding)
            avery_query = torch.cat(avery_query, dim=0).unsqueeze(1)
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten5':
            valid_lens = batch_mask_ids.sum(-1)
            avery_output = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1].unsqueeze(0)
                output = outputs[i].unsqueeze(0)
                valid_len = valid_lens[i].unsqueeze(0)
                entity_output = self.atten2(entity_embedding, output, output, valid_len)
                entity_output = entity_output.mean(dim=1)
                avery_output.append(entity_output)
            avery_output = torch.cat(avery_output, dim=0).unsqueeze(1)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add_atten2':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output + avery_query
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten_add':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            avery_query = (head_query + tail_query) / 2
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query+tail_query) / 2
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_aver_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten_add':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        
        
        return mention_outputs
    
class GPNER15Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER15Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        else:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
#         print(batch_head_positions, batch_head_positions.shape)
#         print(batch_tail_positions, batch_tail_positions.shape)
#         print(outputs.shape)
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten3':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query + tail_query) / 2
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten4':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1]
                avery_embedding = entity_embedding.mean(dim=0).unsqueeze(0)
                avery_query.append(avery_embedding)
            avery_query = torch.cat(avery_query, dim=0).unsqueeze(1)
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten5':
            valid_lens = batch_mask_ids.sum(-1)
            avery_output = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1].unsqueeze(0)
                output = outputs[i].unsqueeze(0)
                valid_len = valid_lens[i].unsqueeze(0)
                entity_output = self.atten2(entity_embedding, output, output, valid_len)
                entity_output = entity_output.mean(dim=1)
                avery_output.append(entity_output)
            avery_output = torch.cat(avery_output, dim=0).unsqueeze(1)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add_atten2':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output + avery_query
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten_add':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            avery_query = (head_query + tail_query) / 2
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query+tail_query) / 2
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_aver_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten_add':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        
        
        return mention_outputs
    
class GPNER21Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER21Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        else:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
#         print(batch_head_positions, batch_head_positions.shape)
#         print(batch_tail_positions, batch_tail_positions.shape)
#         print(outputs.shape)
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten3':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query + tail_query) / 2
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten4':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1]
                avery_embedding = entity_embedding.mean(dim=0).unsqueeze(0)
                avery_query.append(avery_embedding)
            avery_query = torch.cat(avery_query, dim=0).unsqueeze(1)
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten5':
            valid_lens = batch_mask_ids.sum(-1)
            avery_output = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1].unsqueeze(0)
                output = outputs[i].unsqueeze(0)
                valid_len = valid_lens[i].unsqueeze(0)
                entity_output = self.atten2(entity_embedding, output, output, valid_len)
                entity_output = entity_output.mean(dim=1)
                avery_output.append(entity_output)
            avery_output = torch.cat(avery_output, dim=0).unsqueeze(1)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add_atten2':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output + avery_query
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten_add':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            avery_query = (head_query + tail_query) / 2
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query+tail_query) / 2
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_aver_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten_add':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        return mention_outputs
    
class GPNER23Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER23Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        else:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
#         print(batch_head_positions, batch_head_positions.shape)
#         print(batch_tail_positions, batch_tail_positions.shape)
#         print(outputs.shape)
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten3':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query + tail_query) / 2
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten4':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1]
                avery_embedding = entity_embedding.mean(dim=0).unsqueeze(0)
                avery_query.append(avery_embedding)
            avery_query = torch.cat(avery_query, dim=0).unsqueeze(1)
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten5':
            valid_lens = batch_mask_ids.sum(-1)
            avery_output = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1].unsqueeze(0)
                output = outputs[i].unsqueeze(0)
                valid_len = valid_lens[i].unsqueeze(0)
                entity_output = self.atten2(entity_embedding, output, output, valid_len)
                entity_output = entity_output.mean(dim=1)
                avery_output.append(entity_output)
            avery_output = torch.cat(avery_output, dim=0).unsqueeze(1)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add_atten2':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output + avery_query
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten_add':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            avery_query = (head_query + tail_query) / 2
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query+tail_query) / 2
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_aver_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten_add':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        return mention_outputs
    
class GPNER22Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER22Model, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        else:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
#         print('head_query.shape:', head_query.shape)
#         print('args.prefix_merge_mode:', args.prefix_merge_mode)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
#             print('head_output.shape:',head_output.shape)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten3':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query + tail_query) / 2
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten4':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1]
                avery_embedding = entity_embedding.mean(dim=0).unsqueeze(0)
                avery_query.append(avery_embedding)
            avery_query = torch.cat(avery_query, dim=0).unsqueeze(1)
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten5':
            valid_lens = batch_mask_ids.sum(-1)
            avery_output = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1].unsqueeze(0)
                output = outputs[i].unsqueeze(0)
                valid_len = valid_lens[i].unsqueeze(0)
                entity_output = self.atten2(entity_embedding, output, output, valid_len)
                entity_output = entity_output.mean(dim=1)
                avery_output.append(entity_output)
            avery_output = torch.cat(avery_output, dim=0).unsqueeze(1)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add_atten2':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output + avery_query
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten_add':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            avery_query = (head_query + tail_query) / 2
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'aver_atten':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query+tail_query) / 2
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add_aver_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten_add':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        return mention_outputs
    
class GPNER24Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER24Model, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        else:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
#         print('head_query.shape:', head_query.shape)
#         print('args.prefix_merge_mode:', args.prefix_merge_mode)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
#             print('head_output.shape:',head_output.shape)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten3':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query + tail_query) / 2
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten4':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1]
                avery_embedding = entity_embedding.mean(dim=0).unsqueeze(0)
                avery_query.append(avery_embedding)
            avery_query = torch.cat(avery_query, dim=0).unsqueeze(1)
            avery_output = self.atten2(avery_query, outputs, outputs, valid_lens)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten5':
            valid_lens = batch_mask_ids.sum(-1)
            avery_output = []
            for i in range(batch_size):
                entity_embedding = outputs[i, batch_head_positions[i] : batch_tail_positions[i]+1].unsqueeze(0)
                output = outputs[i].unsqueeze(0)
                valid_len = valid_lens[i].unsqueeze(0)
                entity_output = self.atten2(entity_embedding, output, output, valid_len)
                entity_output = entity_output.mean(dim=1)
                avery_output.append(entity_output)
            avery_output = torch.cat(avery_output, dim=0).unsqueeze(1)
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add_atten2':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output + avery_query
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'atten_add':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            avery_query = (head_query + tail_query) / 2
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'aver_atten':
            valid_lens = batch_mask_ids.sum(-1)
            avery_query = (head_query+tail_query) / 2
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)

            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'add_aver_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        elif args.prefix_merge_mode == 'aver_atten_add':
            avery_query = (head_query + tail_query) / 2
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(avery_query, outputs, outputs, valid_lens)
            head_output = head_output + avery_query
            tail_output = tail_output + avery_query
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
        
        return mention_outputs
    
class GPNER25Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER25Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        self.mention_detect1 = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=1, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        self.mention_detect2 = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

        if args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            
    def forward_atten2(self, outputs, batch_mask_ids, batch_head_positions, batch_tail_positions):
        batch_size = len(batch_mask_ids)
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        valid_lens = batch_mask_ids.sum(-1)
        head_output = self.atten2(head_query, outputs, outputs, valid_lens)
        tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
        avery_output = (head_output+tail_output) / 2
        outputs = outputs + avery_output
        mention_outputs2 = self.mention_detect2(outputs, batch_mask_ids)
        return mention_outputs2
    
    def forward_add(self, outputs, batch_mask_ids, batch_head_positions, batch_tail_positions):
        batch_size = len(batch_mask_ids)
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        avery_query = (head_query + tail_query) / 2
        outputs = outputs + avery_query 
        mention_outputs2 = self.mention_detect2(outputs, batch_mask_ids)
        return mention_outputs2
        
        
    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions=None, batch_tail_positions=None, text=None, id2class=None, new_span=None):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        if batch_head_positions != None:
            # 训练
            if args.prefix_merge_mode == 'atten2':
                mention_outputs1 = self.mention_detect1(outputs, batch_mask_ids)
                mention_outputs2 = self.forward_atten2(outputs, batch_mask_ids, batch_head_positions, batch_tail_positions)
            elif args.prefix_merge_mode == 'add':
                mention_outputs1 = self.mention_detect1(outputs, batch_mask_ids)
                mention_outputs2 = self.forward_add(outputs, batch_mask_ids, batch_head_positions, batch_tail_positions)
            return mention_outputs1, mention_outputs2
        else:
            threshold = 0
            batch_head_positions = []
            batch_tail_positions = []
            device = args.device
            spo_list = []
            # 预测
            
            mention_outputs1 = self.mention_detect1(outputs, batch_mask_ids)
            entity1_outputs = mention_outputs1[0].data.cpu().numpy()
            entities = set()
            entity1_outputs[:, [0, -1]] -= np.inf
            entity1_outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(entity1_outputs > threshold)):
                batch_head_positions.append(h)
                batch_tail_positions.append(t)
                entities.add((entity_type ,h, t))
            entity1_list = []
            for entity_type, sh, st in entities:
                entity1_list.append(text[new_span[sh][0]:new_span[st][-1] + 1])
            if batch_head_positions == []:
                return spo_list
            outputs = outputs.repeat(len(batch_head_positions), 1, 1)
            batch_mask_ids = batch_mask_ids.repeat(len(batch_head_positions), 1)
            batch_head_positions = torch.tensor(batch_head_positions).to(device)
            batch_tail_positions = torch.tensor(batch_tail_positions).to(device)
            if args.prefix_merge_mode == 'atten2':
                mention_outputs2 = self.forward_atten2(outputs, batch_mask_ids, batch_head_positions, batch_tail_positions)
            elif args.prefix_merge_mode == 'add':
                mention_outputs2 = self.forward_add(outputs, batch_mask_ids, batch_head_positions, batch_tail_positions)
            for i in range(len(mention_outputs2)):
                entity1 = entity1_list[i]
                entities = set()
                entity2_outputs = mention_outputs2[i].data.cpu().numpy()
                entities = set()
                entity2_outputs[:, [0, -1]] -= np.inf
                entity2_outputs[:, :, [0, -1]] -= np.inf
                for entity_type, h, t in zip(*np.where(entity2_outputs > threshold)):
                    entities.add((entity_type ,h, t))
                for entity_type, sh, st in entities:
                    entity2 = text[new_span[sh][0]:new_span[st][-1] + 1]
                    predicate = id2class[entity_type]
                    spo_list.append({'predicate': predicate, 'subject': entity1, 'object': {'@value': entity2}})
            return spo_list 

class GPNER26Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER26Model, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 44
        self.mention_detect1 = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=1, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        self.mention_detect2 = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        if args.prefix_merge_mode =='atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            
    def forward_atten2(self, outputs, batch_mask_ids, batch_head_positions, batch_tail_positions):
        batch_size = len(batch_mask_ids)
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        valid_lens = batch_mask_ids.sum(-1)
        head_output = self.atten2(head_query, outputs, outputs, valid_lens)
        tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
        avery_output = (head_output+tail_output) / 2
        outputs = outputs + avery_output
        mention_outputs2 = self.mention_detect2(outputs, batch_mask_ids)
        return mention_outputs2
    
    def forward_add(self, outputs, batch_mask_ids, batch_head_positions, batch_tail_positions):
        batch_size = len(batch_mask_ids)
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        avery_query = (head_query + tail_query) / 2
        outputs = outputs + avery_query 
        mention_outputs2 = self.mention_detect2(outputs, batch_mask_ids)
        return mention_outputs2
            
    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions=None, batch_tail_positions=None, text=None, id2class=None, new_span=None):
        args = self.args
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        if batch_head_positions != None:
            if args.prefix_merge_mode == 'atten2':
                mention_outputs1 = self.mention_detect1(outputs, batch_mask_ids)
                mention_outputs2 = self.forward_atten2(outputs, batch_mask_ids, batch_head_positions, batch_tail_positions)

            elif args.prefix_merge_mode == 'add':
                mention_outputs1 = self.mention_detect1(outputs, batch_mask_ids)
                mention_outputs2 = self.forward_add(outputs, batch_mask_ids, batch_head_positions, batch_tail_positions)

            return mention_outputs1, mention_outputs2
        else:
            threshold = 0
            batch_head_positions = []
            batch_tail_positions = []
            device = args.device
            spo_list = []
            # 预测
            
            mention_outputs1 = self.mention_detect1(outputs, batch_mask_ids)
            entity1_outputs = mention_outputs1[0].data.cpu().numpy()
            entities = set()
            entity1_outputs[:, [0, -1]] -= np.inf
            entity1_outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(entity1_outputs > threshold)):
                batch_head_positions.append(h)
                batch_tail_positions.append(t)
                entities.add((entity_type ,h, t))
            entity1_list = []
            for entity_type, sh, st in entities:
                entity1_list.append(text[new_span[sh][0]:new_span[st][-1] + 1])
            if batch_head_positions == []:
                return spo_list
            outputs = outputs.repeat(len(batch_head_positions), 1, 1)
            batch_mask_ids = batch_mask_ids.repeat(len(batch_head_positions), 1)
            batch_head_positions = torch.tensor(batch_head_positions).to(device)
            batch_tail_positions = torch.tensor(batch_tail_positions).to(device)
            if args.prefix_merge_mode == 'atten2':
                mention_outputs2 = self.forward_atten2(outputs, batch_mask_ids, batch_head_positions, batch_tail_positions)
            elif args.prefix_merge_mode == 'add':
                mention_outputs2 = self.forward_add(outputs, batch_mask_ids, batch_head_positions, batch_tail_positions)
            for i in range(len(mention_outputs2)):
                entity1 = entity1_list[i]
                entities = set()
                entity2_outputs = mention_outputs2[i].data.cpu().numpy()
                entities = set()
                entity2_outputs[:, [0, -1]] -= np.inf
                entity2_outputs[:, :, [0, -1]] -= np.inf
                for entity_type, h, t in zip(*np.where(entity2_outputs > threshold)):
                    entities.add((entity_type ,h, t))
                for entity_type, sh, st in entities:
                    entity2 = text[new_span[sh][0]:new_span[st][-1] + 1]
                    predicate = id2class[entity_type]
                    spo_list.append({'predicate': predicate, 'subject': entity2, 'object': {'@value': entity1}})
            return spo_list 

class GPNER31Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER31Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
        
class GPNER32Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER32Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        if self.args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER33Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER33Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER34Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER34Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER35Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER35Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER36Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER36Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        if self.args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER38Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER38Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        if self.args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER37Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER37Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs
    
class GPNER39Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER39Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 11
        self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        return mention_outputs

#R7M
class GPNER41Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER41Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        # entity_class_num = 45   #！
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
        # #！
        # # 打印 batch_size
        # print("Batch size:", batch_size)
        # # 打印 batch_head_positions
        # print("Batch head positions:", batch_head_positions)
        # print("Batch tail positions:", batch_tail_positions)
        # print("Shape of batch_head_positions:", batch_head_positions.shape)
        # print("Shape of batch_tail_positions:", batch_tail_positions.shape)
        # # 打印 head_query
        # print("Head query:", head_query)
        # print("Shape of head_query:", head_query.shape)

        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        # #！
        # # 打印 mention_outputs 的值和形状
        # print("Mention outputs:", mention_outputs)
        # print("Shape of mention outputs:", mention_outputs.shape)
            
        return mention_outputs
    
# R8M
class GPNER42Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER42Model, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)

        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        return mention_outputs
    
# R41M
class GPNER51Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER51Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
            
        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        return mention_outputs
    
# R41M
class GPNER52Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER52Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
            
        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        return mention_outputs
    
# R41M
class GPNER53Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER53Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
            
        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        return mention_outputs
    
# R41M
class GPNER54Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER54Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
            
        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        return mention_outputs
    
# R41M
class GPNER55Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER55Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
            
        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        return mention_outputs
    
# R41M
class GPNER56Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER56Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
            
        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        return mention_outputs
    
# R41M
class GPNER58Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER58Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
            
        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        return mention_outputs
    
# R42M
class GPNER57Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER57Model, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)

        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        return mention_outputs
    
# R42M
class GPNER59Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER59Model, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)

        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        return mention_outputs
    
# R41M
class GPNER61Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER61Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 1
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
            
        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        return mention_outputs
    
# R41M
class GPNER62Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER62Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
            
        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        return mention_outputs
    
# R42M
class GPNER63Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER63Model, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 1
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)

        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        return mention_outputs
    
# R42M
class GPNER64Model(nn.Module):
    def __init__(self, encoder_class, args):
        super(GPNER64Model, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 44
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)

        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        return mention_outputs
    
# R41M
class GPNER41ACE05Model(nn.Module):
    def __init__(self, encoder_class, args, tokenizer):
        self.args = args
        super(GPNER41ACE05Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        if args.prefix_mode in ['entity', 'entity-marker']:
            self.encoder.resize_token_embeddings(len(tokenizer))
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 65
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roberta':
#             print('训练数据')
#             print('batch_token_ids', batch_token_ids)
#             print('batch_mask_ids', batch_mask_ids)
            outputs = self.encoder(batch_token_ids, batch_mask_ids)[0]
#             print('roberta模型model不使用token_type_ids')
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
            
        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        return mention_outputs
    
# R42M
class GPNER42ACE05Model(nn.Module):
    def __init__(self, encoder_class, args, tokenizer):
        super(GPNER42ACE05Model, self).__init__()
        self.args = args
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        if args.prefix_mode in ['entity', 'entity-marker']:
            self.encoder.resize_token_embeddings(len(tokenizer))
        hiddensize = self.encoder.config.hidden_size
        entity_class_num = 65
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'atten2':
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode == 'none':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roberta':
            outputs = self.encoder(batch_token_ids, batch_mask_ids)[0]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)

        if args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'none':
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        
        return mention_outputs
    
# GPFilter78M
class GPFilter78ACE05Model(nn.Module):
    def __init__(self, encoder_class, args, schema):
        super(GPFilter78ACE05Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.args = args
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
        if args.use_efficient_global_pointer == True:
            GlobalPointer = EfficientGlobalPointer
        else:
            GlobalPointer = RawGlobalPointer
        
        self.s_o_head = GlobalPointer(hiddensize=hiddensize, ent_type_size=len(schema), inner_dim=args.inner_dim,
                                      RoPE=False, tril_mask=False, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        self.s_o_tail = GlobalPointer(hiddensize=hiddensize, ent_type_size=len(schema), inner_dim=args.inner_dim,
                                      RoPE=False, tril_mask=False, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)

        
    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        if self.args.model_type == 'roberta':
            outputs = self.encoder(batch_token_ids, batch_mask_ids)[0]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]

        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return so_head_outputs, so_tail_outputs
    
# R3M
class GPNER75Model(nn.Module):
    def __init__(self, encoder_class, args):
        self.args = args
        super(GPNER75Model, self).__init__()
        encoder_path = os.path.join(args.model_dir, args.pretrained_model_name)
        self.encoder = encoder_class.from_pretrained(encoder_path)
        hiddensize = self.encoder.config.hidden_size
#         print('hiddensize:',hiddensize)
        entity_class_num = 11
        if args.prefix_merge_mode == 'add':
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        elif args.prefix_merge_mode in ['atten2', 'add_atten2', 'atten3', 'atten4', 'atten5']:
            self.atten2 = AdditiveAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0)
            self.mention_detect = RawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
        else:
            self.mention_detect = HTRawGlobalPointer(hiddensize=hiddensize, ent_type_size=entity_class_num, inner_dim=args.inner_dim, do_rdrop=args.do_rdrop, dropout=args.dropout).to(args.device)
            self.ht_atten = HTAttention(key_size=hiddensize, query_size=hiddensize ,num_hiddens=args.atten_dim, dropout=0, atten_mode=args.atten_mode)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions):
        args = self.args
        if args.model_type == 'roformer':
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids, output_hidden_states=True).hidden_states[-1]
        else:
            outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)[0]
        batch_size = batch_token_ids.shape[0]
        head_query = outputs[range(batch_size), batch_head_positions].unsqueeze(1)
        tail_query = outputs[range(batch_size), batch_tail_positions].unsqueeze(1)
#         print('head_query.shape, tail_query.shape:', head_query.shape, tail_query.shape)
        if args.prefix_merge_mode == 'atten':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'atten2':
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.atten2(head_query, outputs, outputs, valid_lens)
            tail_output = self.atten2(tail_query, outputs, outputs, valid_lens)
            avery_output = (head_output+tail_output) / 2
            outputs = outputs + avery_output
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
            
        elif args.prefix_merge_mode == 'add':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query 
            mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        elif args.prefix_merge_mode == 'add_atten':
            avery_query = (head_query + tail_query) / 2
            outputs = outputs + avery_query
            
            valid_lens = batch_mask_ids.sum(-1)
            head_output = self.ht_atten(head_query, outputs, outputs, valid_lens)
            tail_output = self.ht_atten(tail_query, outputs, outputs, valid_lens)
            mention_outputs = self.mention_detect(head_output, tail_output, batch_mask_ids)
            
            
        return mention_outputs