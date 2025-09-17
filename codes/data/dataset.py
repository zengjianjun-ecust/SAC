import torch
import numpy as np
from random import choice
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from constant import text_start, left_bracket, right_bracket
    
def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)
    
    
class ERDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128,
            model_type='bert',
    ):
        super(ERDataset, self).__init__()

        self.texts = samples['text']
        if mode != 'test':
            self.spo_lists = samples['spo_list']

        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.model_type = model_type

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        inputs = self.tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True)
#         print("text, inputs['input_ids']:", text, inputs['input_ids'])
        if self.mode != "test":
            spo_list = self.spo_lists[idx]

            sub_start_label = np.zeros((self.max_length,), dtype=int)
            sub_end_label = np.zeros((self.max_length,), dtype=int)
            obj_start_label = np.zeros((self.max_length,), dtype=int)
            obj_end_label = np.zeros((self.max_length,), dtype=int)
            for spo in spo_list:
                sub_encode = self.tokenizer.encode(spo[0])
                sub_start_idx = self.data_processor.search(inputs['input_ids'], sub_encode[1:-1])  # 去掉CLS SEP
                sub_end_idx = sub_start_idx + len(sub_encode[1:-1]) - 1
                obj_encode = self.tokenizer.encode(spo[2])
                obj_start_idx = self.data_processor.search(inputs['input_ids'], obj_encode[1:-1])
                obj_end_idx = obj_start_idx + len(obj_encode[1:-1]) - 1

                sub_start_label[sub_start_idx] = 1
                sub_end_label[sub_end_idx] = 1
                obj_start_label[obj_start_idx] = 1
                obj_end_label[obj_end_idx] = 1
            
            return torch.tensor(inputs['input_ids']), \
                       torch.tensor(inputs['token_type_ids']), \
                       torch.tensor(inputs['attention_mask']), \
                       torch.tensor(sub_start_label).long(), \
                       torch.tensor(sub_end_label).long(), \
                       torch.tensor(obj_start_label).long(), \
                       torch.tensor(obj_end_label).long()
        else:
            
            return torch.tensor(inputs['input_ids']).long(), \
                       torch.tensor(inputs['token_type_ids']).long(), \
                       torch.tensor(inputs['attention_mask']).long()

    def __len__(self):
        return len(self.texts)
        
class REDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128,
    ):
        super(REDataset, self).__init__()

        self.texts = samples['text']
        self.flags = samples['flag']

        if mode != "test":
            self.labels = samples['label']

        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __getitem__(self, idx):
        text, flag = self.texts[idx], self.flags[idx]
        
        inputs = self.tokenizer.encode_plus(text, max_length=self.max_length, padding='max_length', truncation=True)

        s_encode = self.tokenizer.encode(flag[0])
        s_start_idx = self.data_processor.search(inputs['input_ids'], s_encode[1:-1])

        o_encode = self.tokenizer.encode(flag[1])
        o_start_idx = self.data_processor.search(inputs['input_ids'], o_encode[1:-1])
        if self.mode != "test":
            label = self.labels[idx]
            
            return torch.tensor(inputs['input_ids']), \
                       torch.tensor(inputs['token_type_ids']), \
                       torch.tensor(inputs['attention_mask']), \
                       torch.tensor([s_start_idx, o_start_idx]).long(), \
                       torch.tensor(label).long()
        else:
            
            return torch.tensor(inputs['input_ids']), \
                       torch.tensor(inputs['token_type_ids']).long(), \
                       torch.tensor(inputs['attention_mask']).float(), \
                       torch.tensor([s_start_idx, o_start_idx]).long()

    def __len__(self):
        return len(self.texts)
    
class GPNER2Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, obj in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            obj = self.tokenizer.encode(obj, add_special_tokens=False)
            oh = self.data_processor.search(obj, input_ids)
            if sh != -1 and oh != -1:
                spoes.add((sh, sh+len(sub)-1, oh, oh+len(obj)-1))
        entity_labels = [set() for i in range(2)]
        for sh, st, oh, ot in spoes:
            entity_labels[0].add((sh, st)) 
            entity_labels[1].add((oh, ot)) 
        for label in entity_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        return entity_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        for item in examples:
            head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    


class GPLinkerDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.schema = data_processor.schema #spo
        self.args = args
    def __len__(self):
        return len(self.samples)

    def encoder(self, item):
        args = self.args
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o, s_t, o_t in item["spo_list"]:
            s = self.tokenizer.encode(s, add_special_tokens=False)
            p = self.schema[s_t + "_" + p + "_" +o_t]
            o = self.tokenizer.encode(o, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(s, input_ids)
                ohs = self.data_processor.search_all(o, input_ids)
                for sh in shs:
                    for oh in ohs:
                        spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
            else:
                sh = self.data_processor.search(s, input_ids)
                oh = self.data_processor.search(o, input_ids)
                if sh != -1 and oh != -1:
                    spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        for sh, st, p, oh, ot in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        return text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP

        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels
    
class GPFilterDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train', 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.schema = data_processor.schema #spo
        self.args = args
        
    def __len__(self):
        return len(self.samples)

    def encoder(self, item):
        args = self.args
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o, s_t, o_t in item["spo_list"]:
            s = self.tokenizer.encode(s, add_special_tokens=False)
            p = self.schema[s_t + "_" + p + "_" +o_t]
            o = self.tokenizer.encode(o, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(s, input_ids)
                ohs = self.data_processor.search_all(o, input_ids)
                for sh in shs:
                    for oh in ohs:
                        spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
            else:
                sh = self.data_processor.search(s, input_ids)
                oh = self.data_processor.search(o, input_ids)
                if sh != -1 and oh != -1:
                    spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))

        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        for sh, st, p, oh, ot in spoes:
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        return head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels, batch_tail_labels = [], []
        for item in examples:
            head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP

        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_tail_labels
    
class GPFilterace05Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train'
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.predicate2id = data_processor.predicate2id #spo
        self.schema = data_processor.schema #spo
        self.args = args
        
    def __len__(self):
        return len(self.samples)

    def encoder(self, item):
        args = self.args
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
#         token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o, s_t, o_t in item["spo_list"]:
            sub_tokens = self.tokenizer.encode(s, add_special_tokens=False)
            key = s_t + "_" + p + "_" +o_t if args.with_type else p
            p = self.predicate2id[key]
            obj_tokens = self.tokenizer.encode(o, add_special_tokens=False)
            sh = self.data_processor.search(sub_tokens, input_ids)
            oh = self.data_processor.search(obj_tokens, input_ids)
            
            if sh == -1:
                sub_tokens = self.tokenizer.encode(' '+s, add_special_tokens=False)
                sh = self.data_processor.search(sub_tokens, input_ids)
            if oh == -1:
                obj_tokens = self.tokenizer.encode(' '+o, add_special_tokens=False)
                oh = self.data_processor.search(obj_tokens, input_ids)
            if sh != -1 and oh != -1:
                spoes.add((sh, sh+len(sub_tokens)-1, p, oh, oh+len(obj_tokens)-1))

        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        for sh, st, p, oh, ot in spoes:
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        return head_labels, tail_labels, input_ids, attention_mask

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids = [], []
        batch_head_labels, batch_tail_labels = [], []
        for item in examples:
            head_labels, tail_labels, input_ids, attention_mask = item
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()

        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\

        return batch_token_ids, batch_mask_ids, batch_head_labels, batch_tail_labels
    

    
class GPNERDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNERACE05Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train'
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = 7 if args.with_type else 1
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
#         token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        if args.with_type:
            for sub, sub_type in item["entity_list"]:
                sub_tokens = self.tokenizer.encode(sub, add_special_tokens=False)
                sh = self.data_processor.search(sub_tokens, input_ids)

                if sh != -1:
                    spoes.add((sh, sh+len(sub_tokens)-1, class2id[sub_type]))
                else:
                    sub_tokens = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                    sh = self.data_processor.search(sub_tokens, input_ids)
                    if sh != -1:
                        spoes.add((sh, sh+len(sub_tokens)-1, class2id[sub_type]))
            head_labels = [set() for i in range(num_labels)]
            for sh, st, sub_type in spoes:
                head_labels[sub_type].add((sh, st)) 
        else:
            for sub in item["entity_list"]:
                sub_tokens = self.tokenizer.encode(sub, add_special_tokens=False)
                sh = self.data_processor.search(sub_tokens, input_ids)

                if sh != -1:
                    spoes.add((sh, sh+len(sub_tokens)-1))
                else:
                    sub_tokens = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                    sh = self.data_processor.search(sub_tokens, input_ids)
                    if sh != -1:
                        spoes.add((sh, sh+len(sub_tokens)-1))
            head_labels = [set() for i in range(num_labels)]
            for sh, st in spoes:
                head_labels[0].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return head_labels, input_ids, attention_mask

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids = [], []
        batch_head_labels = []
        for item in examples:
            head_labels, input_ids, attention_mask = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return batch_token_ids, batch_mask_ids, batch_head_labels
    
class GPNER3Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    

class GPNER4Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
class GPNER9Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNER5SubDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNER5Sp2oDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNER6ObjDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNER6Op2sDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
    
class GPNER7Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions

class GPNER8Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
class GPFilter78Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.schema = data_processor.schema #spo
        self.predicate2id = data_processor.predicate2id
        self.args = args
        
    def __len__(self):
        return len(self.samples)

    def encoder(self, item):
        args = self.args
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o in item["spo_list"]:
            s = self.tokenizer.encode(s, add_special_tokens=False)
            p = self.predicate2id[p]
            o = self.tokenizer.encode(o, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(s, input_ids)
                ohs = self.data_processor.search_all(o, input_ids)
                for sh in shs:
                    for oh in ohs:
                        spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
            else:
                sh = self.data_processor.search(s, input_ids)
                oh = self.data_processor.search(o, input_ids)
                if sh != -1 and oh != -1:
                    spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))

        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        for sh, st, p, oh, ot in spoes:
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        return head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels, batch_tail_labels = [], []
        for item in examples:
            head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP

        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_tail_labels
    
class GPNER17Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels

class GPNER18Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class CasRelDataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        text_len = len(input_ids)
        
        s2ro_map = {}
        for sub, predicate, obj in item["spo_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            obj = self.tokenizer.encode(obj, add_special_tokens=False)
            oh = self.data_processor.search(obj, input_ids)
            if sh != -1 and oh != -1:
                st = sh+len(sub)-1
                ot = oh+len(obj)-1
                if (sh, st) not in s2ro_map:
                    s2ro_map[(sh, st)] = []
                s2ro_map[(sh, st)].append((oh, ot, class2id[predicate]))
#         if not s2ro_map:
#             print(text)
        sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
        sub_head, sub_tail = 0, 0
        obj_heads, obj_tails = np.zeros((text_len, num_labels)), np.zeros((text_len, num_labels))
        if s2ro_map:
            for s in s2ro_map:
                sub_heads[s[0]] = 1     
                sub_tails[s[1]] = 1
            sub_head, sub_tail = choice(list(s2ro_map.keys()))
            for ro in s2ro_map.get((sub_head, sub_tail), []): 
                obj_heads[ro[0]][ro[2]] = 1
                obj_tails[ro[1]][ro[2]] = 1            
        
        return input_ids, attention_mask, token_type_ids, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_sub_heads, batch_sub_tails = [], []
        batch_sub_head, batch_sub_tail, batch_obj_heads, batch_obj_tails = [], [], [], []
        batch_head_labels = []
        for item in examples:
            input_ids, attention_mask, token_type_ids, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails = item
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_sub_heads.append(sub_heads)
            batch_sub_tails.append(sub_tails)
            batch_sub_head.append(sub_head)
            batch_sub_tail.append(sub_tail)
            batch_obj_heads.append(obj_heads)
            batch_obj_tails.append(obj_tails)



        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_sub_heads = torch.tensor(sequence_padding(batch_sub_heads)).long()
        batch_sub_tails = torch.tensor(sequence_padding(batch_sub_tails)).long()
        batch_sub_head = torch.tensor(batch_sub_head).long()
        batch_sub_tail = torch.tensor(batch_sub_tail).long()
        batch_obj_heads = torch.tensor(sequence_padding(batch_obj_heads)).long()
        batch_obj_tails = torch.tensor(sequence_padding(batch_obj_tails)).long()
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_sub_heads, batch_sub_tails, batch_sub_head, batch_sub_tail, batch_obj_heads, batch_obj_tails
    
class GPNER11Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions

class GPNER12Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
class GPNER13Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
class GPNER14Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
class GPNER15Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
    #       print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#       print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
class GPNER21Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
class GPNER23Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
class GPNER22Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
class GPNER24Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
class GPNER25Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub in item["entity1_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, 0))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, 0))
        head_labels = [set() for i in range(1)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        spoes = set()
        for sub, sub_type in item["entity2_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        tail_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            tail_labels[sub_type].add((sh, st)) 
        for label in tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        head_position, tail_position = 0, 0

        prefix_entity = item['entity1']
#             print(prefix_entity)
        sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
        sh = self.data_processor.search(sub, input_ids)
        if sh != -1:
            head_position = sh
            tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, tail_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels, batch_tail_labels = [], []
        text_list = []
        for item in examples:
            text, head_labels, tail_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_tail_labels, batch_head_positions, batch_tail_positions
    
class GPNER26Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub in item["entity1_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, 0))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, 0))
        head_labels = [set() for i in range(1)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        spoes = set()
        for sub, sub_type in item["entity2_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        tail_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            tail_labels[sub_type].add((sh, st)) 
        for label in tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        head_position, tail_position = 0, 0

        prefix_entity = item['entity1']
#             print(prefix_entity)
        sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
        sh = self.data_processor.search(sub, input_ids)
        if sh != -1:
            head_position = sh
            tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, tail_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels, batch_tail_labels = [], []
        text_list = []
        for item in examples:
            text, head_labels, tail_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_tail_labels, batch_head_positions, batch_tail_positions

class GPNER31Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNER32Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNER33Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNER34Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNER35Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNER36Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNER38Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNER37Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
class GPNER39Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        return text, head_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        
        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels
    
#R7D
class GPNER41Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions

# R8D
class GPNER42Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
# R41D    
class GPNER51Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
# R41D
class GPNER52Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
# R41D
class GPNER53Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
# R41D
class GPNER54Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
# R41D
class GPNER55Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
# R41D
class GPNER56Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
# R41D
class GPNER58Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions

# R42D
class GPNER57Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
# R42D
class GPNER59Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
# R41D
class GPNER61Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
# R41D
class GPNER62Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
# R42D
class GPNER63Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions
    
# R42D
class GPNER64Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions

# R41D
class GPNER41ACE05Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        if args.model_type == 'roberta':
            token_type_ids = [0] * len(input_ids)
#             print('roberta模型dataset构造token_type_ids')
        else:
            token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub_tokens = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub_tokens, input_ids)
                if shs == []:
                    sub_tokens = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                    shs = self.data_processor.search_all(sub_tokens, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub_tokens)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub_tokens, input_ids)
                if sh == -1:
                    sub_tokens = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                    sh = self.data_processor.search(sub_tokens, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub_tokens)-1, class2id[sub_type]))
                    
                    
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions

# R42D
class GPNER42ACE05Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.args = args
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        args = self.args
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
#         print(text)
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        if args.model_type == 'roberta':
            token_type_ids = [0] * len(input_ids)
        else:
            token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub_tokens = self.tokenizer.encode(sub, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub_tokens, input_ids)
                if shs == []:
                    sub_tokens = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                    shs = self.data_processor.search_all(sub_tokens, input_ids)
                for sh in shs:
                    spoes.add((sh, sh+len(sub_tokens)-1, class2id[sub_type]))
            else:
                sh = self.data_processor.search(sub_tokens, input_ids)
                if sh == -1:
                    sub_tokens = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                    sh = self.data_processor.search(sub_tokens, input_ids)
                if sh != -1:
                    spoes.add((sh, sh+len(sub_tokens)-1, class2id[sub_type]))
                    
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()
        
        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions

# GPFilter78D
class GPFilter78ACE05Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            args,
            mode='train',
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = args.max_length
        self.schema = data_processor.schema #spo
        self.predicate2id = data_processor.predicate2id
        self.args = args
        
    def __len__(self):
        return len(self.samples)

    def encoder(self, item):
        args = self.args
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        if args.model_type == 'roberta':
            token_type_ids = [0] * len(input_ids)
        else:
            token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o in item["spo_list"]:
            sub_tokens = self.tokenizer.encode(s, add_special_tokens=False)
            p = self.predicate2id[p]
            obj_tokens = self.tokenizer.encode(o, add_special_tokens=False)
            if args.do_search_all:
                shs = self.data_processor.search_all(sub_tokens, input_ids)
                ohs = self.data_processor.search_all(obj_tokens, input_ids)
                if shs == []:
                    sub_tokens = self.tokenizer.encode(' '+s, add_special_tokens=False)
                    shs = self.data_processor.search_all(sub_tokens, input_ids)
                if ohs == []:
                    obj_tokens = self.tokenizer.encode(' '+o, add_special_tokens=False)
                    ohs = self.data_processor.search_all(obj_tokens, input_ids)
                for sh in shs:
                    for oh in ohs:
                        spoes.add((sh, sh+len(sub_tokens)-1, p, oh, oh+len(obj_tokens)-1))
            else:
                sh = self.data_processor.search(sub_tokens, input_ids)
                oh = self.data_processor.search(obj_tokens, input_ids)
                if sh == -1:
                    sub_tokens = self.tokenizer.encode(' '+s, add_special_tokens=False)
                    sh = self.data_processor.search(sub_tokens, input_ids)
                if oh == -1:
                    obj_tokens = self.tokenizer.encode(' '+o, add_special_tokens=False)
                    oh = self.data_processor.search(obj_tokens, input_ids)
                if sh != -1 and oh != -1:
                    spoes.add((sh, sh+len(sub_tokens)-1, p, oh, oh+len(obj_tokens)-1))

        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        for sh, st, p, oh, ot in spoes:
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        return head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_labels, batch_tail_labels = [], []
        for item in examples:
            head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP

        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_tail_labels

# R3D
class GPNER75Dataset(Dataset):
    def __init__(
            self,
            samples,
            data_processor,
            tokenizer,
            mode='train',
            max_length=128, 
    ):
        self.samples = samples
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
#         self.schema = data_processor.schema #spo
        
    def __len__(self):
        return len(self.samples)
    
    def encoder(self, item):
        num_labels = self.data_processor.num_labels
        class2id = self.data_processor.class2id
        text = item["text"]
        
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for sub, sub_type in item["entity_list"]:
            sub = self.tokenizer.encode(sub, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                spoes.add((sh, sh+len(sub)-1, class2id[sub_type]))
        head_labels = [set() for i in range(num_labels)]
        for sh, st, sub_type in spoes:
            head_labels[sub_type].add((sh, st)) 
        for label in head_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        head_labels = sequence_padding([list(l) for l in head_labels])
        head_position, tail_position = 0, 0
        if item['type'] != 0:
            prefix_entity = item['prefix_entity']
#             print(prefix_entity)
            sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
            sh = self.data_processor.search(sub, input_ids)
            if sh != -1:
                head_position = sh
                tail_position = sh+len(sub)-1
#             print(head_position, tail_position)
        return text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position

    def __getitem__(self, idx):
        item = self.samples[idx]
        return self.encoder(item)

    @staticmethod
    def collate_fn(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_head_positions, batch_tail_positions = [], []
        batch_head_labels = []
        text_list = []
        for item in examples:
            text, head_labels, input_ids, attention_mask, token_type_ids, head_position, tail_position = item
            batch_head_labels.append(head_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_head_positions.append(head_position)
            batch_tail_positions.append(tail_position)

            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_head_positions = torch.tensor(batch_head_positions).long()
        batch_tail_positions = torch.tensor(batch_tail_positions).long()

        return batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_head_positions, batch_tail_positions