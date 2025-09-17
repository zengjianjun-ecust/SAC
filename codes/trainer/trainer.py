import torch
import torch.nn as nn
import os
import json
import jsonlines
import shutil
import math
import time
import numpy as np
import torch.nn.functional as F
from d2l import torch as d2l
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import ProgressBar, TokenRematch, get_time, save_args, SPO, ACESPO, SPO_No_Type, save_cur_epoch, cal_ace_prf
from metrics import er_metric, re_metric, gen_metric, rc_metric, p2so_metric
from loss import multilabel_categorical_crossentropy, sparse_multilabel_categorical_crossentropy
from optimizer import GPLinkerOptimizer

class Trainer(object):
    def __init__(
            self,
            args,
            data_processor,
            logger,
            model=None,
            tokenizer=None,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):

        self.args = args
        self.model = model
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        

        if train_dataset is not None and isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset

        if eval_dataset is not None and isinstance(eval_dataset, Dataset):
            self.eval_dataset = eval_dataset

        self.logger = logger
        self.ngram_dict = ngram_dict

    def train(self):
        args = self.args
        logger = self.logger
        model = self.model
        self.output_dir = os.path.join(args.output_dir, args.time)

        
        if args.distributed == True:
            model = nn.DataParallel(model, device_ids=args.devices).to(args.device)
        else:
            model.to(args.device)
            
        
        
        train_dataloader = self.get_train_dataloader()

        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = num_training_steps * args.warmup_proportion
        num_examples = len(train_dataloader.dataset)
        # skip_evaluate_models：妥协做法，以后应该把skip_evaluate单独设置在主函数的args中
        skip_evaluate_models = ['baseline', 'gplinker', 'gpner', 'gpner2', 'gpner3', 'gpner4', 'gpner5', 'gpner6', 'gpner7', 'gpner8', 'gpner9', 'ace05', 'gpfilter', 'gpfilter78', 'biaffinefilter', 'gpner11', 'gpner12', 'gpner13', 'gpner14', 'bibm_dual', 'gpner14', 'gpner15', 'gpner17', 'gpner18', 'gpnersub', 'gpnerobj', 'gpner16', 'gpner17ace05', 'gplinkerace05', 'gpner19ace05', 'casrel', 'gpner21', 'gpner22', 'gpner23', 'gpner24', 'gpner25', 'gpner26', 'gpner31', 'gpner32', 'gpner33', 'gpner34', 'gpner35', 'gpner36', 'gpner37', 'gpner38', 'gpner39', 'gpner40', 'gpner41', 'gpner42', 'gpner51', 'gpner52', 'gpner53', 'gpner54', 'gpner55', 'gpner56', 'gpner57', 'gpner58', 'gpner59', 'gpner61', 'gpner62', 'gpner63', 'gpner64', 'gpner41ace05', 'gpner42ace05', 'gpfilter78ace05', 'gpner75']
        if args.method_name in skip_evaluate_models:
            args.skip_evaluate = True
#         if args.method_name in skip_evaluate_models:
        if args.skip_evaluate:
            optimizer = GPLinkerOptimizer(model, args, train_steps= len(train_dataloader)  * args.epochs)
        else:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_training_steps)

        logger.info("***** Running training *****")
        logger.info("Num samples %d", num_examples)
        logger.info("Num epochs %d", args.epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_warmup_steps)

        global_step = 0
        best_step = None
        best_score = 0
        cnt_patience = 0
        
        animator = d2l.Animator(xlabel='epoch', xlim=[0, args.epochs], ylim=[0, 1], fmts=('k-', 'r--', 'y-.', 'm:', 'g--', 'b-.', 'c:'),
                                legend=[f'train loss/{args.loss_show_rate}', 'train_p', 'train_r', 'train_f1', 'test_p', 'test_r', 'test_f1'])
        # 统计指标
        metric = d2l.Accumulator(6)
        num_batches = len(train_dataloader)
        
        total_data = len(train_dataloader.dataset)
        total_batches = len(train_dataloader)

        print(f"总数据条数: {total_data}")
        print(f"总批次数: {total_batches}")
        
        start_time = time.time()
        for epoch in range(args.cur_epoch, args.epochs):
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, item in enumerate(train_dataloader):
                loss, train_p, train_r, train_f1 = self.training_step(model, item)
                loss = loss.item()
                metric.add(loss, train_p, train_r, train_f1, item[0].shape[0], 1)
                pbar(step, {'loss': loss})

                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if not args.skip_evaluate:
                    scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.skip_evaluate:
                        animator.add(
                                global_step / num_batches, 
                                (loss / args.loss_show_rate, train_p, train_r, train_f1, None, None, None))
                        if not os.path.exists(self.output_dir):
                            os.makedirs(self.output_dir)
                        d2l.plt.savefig(os.path.join(self.output_dir, '训练过程.jpg'), dpi=300)
#                     else:
            if not args.skip_evaluate:
                val_p, val_r, val_f1 = self.evaluate(model)
                animator.add(
                    global_step / num_batches, 
                    (# metric[0] / metric[-1] / args.loss_show_rate, # loss太大，除以loss_show_rate才能在[0,1]范围内看到
                     loss / args.loss_show_rate,
                     train_p,  # metric[1] / metric[-1],
                     train_r,  # metric[2] / metric[-1],
                     train_f1, # metric[3] / metric[-1],
                     val_p,
                     val_r,
                     val_f1))
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                d2l.plt.savefig(os.path.join(self.output_dir, '训练过程.jpg'), dpi=300)

                if args.save_metric == 'step':
                    save_metric = global_step
                elif args.save_metric == 'epoch':
                    save_metric = epoch
                elif args.save_metric == 'loss':
                    # e的700次方刚好大于0，不存在数值问题
                    # 除以10，避免loss太大，exp(-loss)次方由于数值问题会小于0，导致存不上，最大可以处理7000的loss
                    save_metric = math.exp(- loss / 10) # math.exp(- metric[0] / metric[-1] / 10)
                elif args.save_metric == 'p':
                    save_metric = val_p
                elif args.save_metric == 'r':
                    save_metric = val_r
                elif args.save_metric == 'f1':
                    save_metric = val_f1

                if save_metric > best_score:
                    best_score = save_metric
                    best_step = global_step
                    cnt_patience = 0
                    self.args.loss = loss # metric[0] / metric[-1]
                    self.args.train_p, self.args.train_r, self.args.train_f1 = train_p, train_r, train_f1
                                     #  metric[1] / metric[-1], metric[2] / metric[-1], metric[3] / metric[-1]
                    self.args.val_p, self.args.var_r, self.args.val_f1 = val_p, val_r, val_f1
                    self._save_checkpoint(model, epoch+1)
#                 else:
#                     cnt_patience += 1
#                     self.logger.info("Earlystopper counter: %s out of %s", cnt_patience, args.earlystop_patience)
#                     if cnt_patience >= self.args.earlystop_patience:
#                         break
#             if cnt_patience >= args.earlystop_patience:
#                 break
#             if args.method_name in skip_evaluate_models:
            if args.skip_evaluate:
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.args.total_examples = metric[4]
                self.args.time_per_example = elapsed_time / metric[4]
                self.args.num_batches = num_batches
                self.args.loss = loss
                self.args.train_p, self.args.train_r, self.args.train_f1 = train_p, train_r, train_f1
                self.args.cur_epoch = epoch + 1
                self._save_checkpoint(model, epoch + 1)
                if 'ace05' in args.method_name:
                    if args.method_name in ['gpner41ace05', 'gpner42ace05']:
                        test_f1, test_p, test_r = self.predict(self.output_dir, epoch+1)
                    elif args.method_name == 'gpfilter78ace05':
                        test_f1, test_p, test_r = self.predict_filter(output_dir=self.output_dir, epoch=epoch+1)
                    self._save_checkpoint(model, epoch+1, name=f'pytorch_model_epoch{epoch+1}.pt')
                    animator.add(epoch+1, (None, None, None, None, test_p/100, test_r/100, test_f1/100))
                elif args.method_name in ['gpner41', 'gpner42']:
                    self.predict(epoch+1)
                    animator.add(epoch+1, (None, None, None, None, 0, 0, 0))
                    self._save_checkpoint(model, epoch+1, name=f'pytorch_model_epoch{epoch+1}.pt')
                elif args.method_name == 'gpfilter78':
                    self.predict_filter(epoch=epoch+1)
                    animator.add(epoch+1, (None, None, None, None, 0, 0, 0))
                    self._save_checkpoint(model, epoch+1, name=f'pytorch_model_epoch{epoch+1}.pt')
        
        logger.info(f"\n***** {args.finetuned_model_name} model training stop *****" )
        logger.info(f'finished time: {get_time()}')
        logger.info(f"best val_{args.save_metric}: {best_score}, best step: {best_step}\n" )
        # 这句会报oom的错误
#         if args.device.__str__() != 'cpu':
#             torch.cuda.empty_cache()

        return global_step, best_step

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model, epoch, name=''):
        args = self.args
        
        if args.distributed:
            model=model.module
        # 防止91存到3卡，但是82没有3卡的情况
        model = model.to(torch.device('cpu'))
        if name == '':
            name = 'pytorch_model.pt'
        torch.save(model.state_dict(), os.path.join(self.output_dir, name))
        self.logger.info('Saving models checkpoint to %s', self.output_dir)
        self.tokenizer.save_vocabulary(save_directory=self.output_dir)
        model = model.to(args.device)
        save_args(args, self.output_dir)
        save_cur_epoch(epoch, self.output_dir)
        shutil.copyfile(os.path.join(args.model_dir, args.pretrained_model_name, 'config.json'),
                        os.path.join(self.output_dir, 'config.json'))
    
    
    def load_checkpoint(self, fix_state_dict=False, model_name='pytorch_model.pt'):
        args = self.args
        load_dir = os.path.join(args.output_dir, args.model_version)
        self.logger.info(f'load model from {load_dir}')
        # 每次加载到cpu中，防止爆显存
        checkpoint = torch.load(os.path.join(load_dir, model_name), map_location=torch.device('cpu'))
        
        if fix_state_dict:
            checkpoint['mention_detect.dense.1.weight'] = checkpoint['mention_detect.dense.weight']
            checkpoint['mention_detect.dense.1.bias'] = checkpoint['mention_detect.dense.bias']
            del checkpoint['mention_detect.dense.weight']
            del checkpoint['mention_detect.dense.bias']
        # 打印状态字典
#         for key, value in checkpoint.items():
#             print(key, value.shape)
            
            
        if 'module' in list(checkpoint.keys())[0].split('.'):
            self.model = nn.DataParallel(self.model, device_ids=args.devices).to(args.device)
        self.model.load_state_dict(checkpoint)
    
    def training_step(self, model, item):
        raise NotImplementedError

    def get_train_dataloader(self):
        collate_fn = self.train_dataset.collate_fn if hasattr(self.train_dataset, 'collate_fn') else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False if self.args.do_rdrop else True,
            collate_fn=collate_fn
        )

    def get_eval_dataloader(self):
        collate_fn = self.eval_dataset.collate_fn if hasattr(self.eval_dataset, 'collate_fn') else None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

    def get_test_dataloader(self, test_dataset, batch_size=None):
        collate_fn = test_dataset.collate_fn if hasattr(test_dataset, 'collate_fn') else None
        if not batch_size:
            batch_size = self.args.eval_batch_size

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )


class ERTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(ERTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            ngram_dict=ngram_dict
        )

        self.loss_fn = nn.BCELoss()

    def training_step(self, model, item):
        model.train()

        item = [i.to(self.args.device) for i in item]
        input_ids, token_type_ids, attention_mask, sub_start_label, sub_end_label, obj_start_label, obj_end_label = item
        
        
        
        sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits = model(input_ids,
                                                                                       token_type_ids,
                                                                                       attention_mask)
        active_index = attention_mask.view(-1) == 1
        cal_loss = self.cal_rdrop_loss if self.args.do_rdrop else self.cal_loss
        
        sub_start_loss = cal_loss(sub_start_logits, sub_start_label, active_index)
        sub_end_loss = cal_loss(sub_end_logits, sub_end_label, active_index)
        obj_start_loss = cal_loss(obj_start_logits, obj_start_label, active_index)
        obj_end_loss = cal_loss(obj_end_logits, obj_end_label, active_index)
        loss = sub_start_loss + sub_end_loss + obj_start_loss + obj_end_loss
        
        sub_start_p, sub_start_r, sub_start_f1 = self.cal_prf1(sub_start_logits, sub_start_label, active_index)
        sub_end_p, sub_end_r, sub_end_f1 = self.cal_prf1(sub_end_logits, sub_start_label, active_index)
        obj_start_p, obj_start_r, obj_start_f1 = self.cal_prf1(obj_start_logits, obj_start_label, active_index)
        obj_end_p, obj_end_r, obj_end_f1 = self.cal_prf1(obj_end_logits, obj_end_label, active_index)
        
        p = (sub_start_p + sub_end_p + obj_start_p + obj_end_p) / 4
        r = (sub_start_r + sub_end_r + obj_start_r + obj_end_r) / 4
        f1 = (sub_start_f1 + sub_end_f1 + obj_start_f1 + obj_end_f1) / 4
        loss.backward()

        return loss.detach(), p, r, f1
    
    def cal_prf1(self, logits, labels, active_index):
        active_index = active_index.cpu()
        threshold = self.args.train_threshold
        active_logits = (logits.detach().view(-1) >= threshold).cpu().long()[active_index]
        active_labels = labels.detach().cpu().view(-1)[active_index]
        p, r, f1, _ = er_metric(active_logits, active_labels)
        return p, r, f1
    
    def cal_loss(self, logits, labels, active_index):
        active_labels = labels.view(-1)[active_index]
        active_logits = logits.view(-1)[active_index]
        return self.loss_fn(active_logits.float()[1:-1], active_labels.float()[1:-1])
    
    def cal_rdrop_loss(self, logits, labels, active_index):
        loss_bce = self.cal_loss(logits, labels, active_index)
        active_logits = logits.view(-1)[active_index].float()[1:-1]
        loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='mean') + \
                  F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='mean')
        return loss_bce + loss_kl / 4 * self.args.rdrop_alpha

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        sub_start_preds = []
        sub_end_preds = []
        obj_start_preds = []
        obj_end_preds = []

        sub_start_trues = []
        sub_end_trues = []
        obj_start_trues = []
        obj_end_trues = []

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()

            item = [i.to(args.device) for i in item]
            
            input_ids, token_type_ids, attention_mask, sub_start_label, sub_end_label, obj_start_label, obj_end_label = item
            

            with torch.no_grad():
                
                sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits = model(input_ids,
                                                                                               token_type_ids,
                                                                                               attention_mask)

            active_index = attention_mask.view(-1) == 1
            sub_start_preds.extend((sub_start_logits.detach().view(-1) >= args.train_threshold).cpu().long()[active_index])
            sub_end_preds.extend((sub_end_logits.detach().view(-1) >= args.train_threshold).cpu().long()[active_index])
            obj_start_preds.extend((obj_start_logits.detach().view(-1) >= args.train_threshold).cpu().long()[active_index])
            obj_end_preds.extend((obj_end_logits.detach().view(-1) >= args.train_threshold).cpu()[active_index])

            sub_start_trues.extend(sub_start_label.detach().cpu().view(-1)[active_index].tolist())
            sub_end_trues.extend(sub_end_label.detach().cpu().view(-1)[active_index].tolist())
            obj_start_trues.extend(obj_start_label.detach().cpu().view(-1)[active_index].tolist())
            obj_end_trues.extend(obj_end_label.detach().cpu().view(-1)[active_index].tolist())

        s_start_p, s_start_r, s_start_f1, _ = er_metric(sub_start_preds, sub_start_trues)
        s_end_p, s_end_r, s_end_f1, _ = er_metric(sub_end_preds, sub_end_trues)
        o_start_p, o_start_r, o_start_f1, _ = er_metric(obj_start_preds, obj_start_trues)
        o_end_p, o_end_r, o_end_f1, _ = er_metric(obj_end_preds, obj_end_trues)
        p = (s_start_p + s_end_p + o_end_p + o_start_p) / 4
        r = (s_start_r + s_end_r + o_end_r + o_start_r) / 4
        f1 = (s_start_f1 + s_end_f1 + o_end_f1 + o_start_f1) / 4
        
        logger.info("%s f1 score: %s", args.finetuned_model_name, f1)
        return p, r, f1

    def predict(self, test_dataset):
        args = self.args
        logger = self.logger
        model = self.model
        test_dataloader = self.get_test_dataloader(test_dataset, batch_size=1)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.output_dir, 'CMeIE_test.json')
        logger.info(f"***** write predict file to {output_dir} *****")
        with open(output_dir, 'w', encoding='utf-8') as f:
            pbar = ProgressBar(n_total=len(test_dataloader), desc='Predicting')
            start_time = time.time()
            for step, item in enumerate(test_dataloader):
                pbar(step)
                model.eval()

                item = [i.to(args.device) for i in item]
                
                input_ids, token_type_ids, attention_mask = item
                
                with torch.no_grad():
                    
                    sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits = model(input_ids,
                                                                                                   token_type_ids,
                                                                                                   attention_mask)

                    text = test_dataset.texts[step]
                    text_start_id, text_end_id = 1, attention_mask.sum().int().item() - 1  # end+1
#                     print(text, self.tokenizer.tokenize(text))
#                     if args.pretrained_model_name == 'albert-xxlarge-v1':
#                         tokenize_list = [i if i != ',' else '，' for i in self.tokenizer.tokenize(text)[1:]]
#                         text_mapping = TokenRematch().rematch(text, tokenize_list)
#                     else:
                    text_mapping = TokenRematch().rematch(text, self.tokenizer.tokenize(text))

                    sub_arg_list = self.data_processor.extract_arg(sub_start_logits.view(-1), sub_end_logits.view(-1), text_start_id, text_end_id,
                                                                   text, text_mapping)
                    obj_arg_list = self.data_processor.extract_arg(obj_start_logits.view(-1), obj_end_logits.view(-1), text_start_id, text_end_id,
                                                                   text, text_mapping)
                    result = {'text': text, 'sub_list': sub_arg_list, 'obj_list': obj_arg_list}
                    json_data = json.dumps(result, ensure_ascii=False)
                    f.write(json_data + '\n')
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_per_example = elapsed_time / num_examples
            print('总时间：', elapsed_time)
            print('样本数：', num_examples)
            print('每个样本所花时间：', time_per_example)
            
            
class RETrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
    ):
        super(RETrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )

    def training_step(self, model, item):
        model.train()

        
        input_ids, token_type_ids, attention_mask, flag, label = item

        input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                 token_type_ids.to(self.args.device), \
                                                                 attention_mask.to(self.args.device), \
                                                                 flag.to(self.args.device), label.to(self.args.device)
        
        loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label)
        loss = loss.mean()
        loss.backward()
        
        preds = logits.argmax(axis=1).detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        p, r, f1, _ = re_metric(preds, label)
        return loss.detach(), p, r, f1

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()

            input_ids, token_type_ids, attention_mask, flag, label = item

            input_ids, token_type_ids, attention_mask, flag, label = input_ids.to(self.args.device), \
                                                                     token_type_ids.to(self.args.device), \
                                                                     attention_mask.to(self.args.device), \
                                                                     flag.to(self.args.device), label.to(self.args.device)

            with torch.no_grad():
                loss, logits = model(input_ids, token_type_ids, attention_mask, flag, label, mode='dev')

            if preds is None:
                preds = logits.detach().cpu().numpy()
                eval_labels = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, label.detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        p, r, f1, _ = re_metric(preds, eval_labels)
        logger.info("%s precision: %s - recall: %s - f1 score: %s", args.finetuned_model_name, p, r, f1)
        return p, r, f1

    def predict(self, test_samples, model, re_dataset_class):
        args = self.args
        logger = self.logger
        model.to(args.device)
        model.eval()
        
        logger.info("***** Running prediction *****")
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            pbar = ProgressBar(n_total=len(test_samples), desc='Predicting')
            start_time = time.time()
            for step, data in enumerate(test_samples):
                pbar(step)
                results, outputs = self.data_processor.build_text(data)
                spo_list = [re['spo_list'] for re in results]
                temp_re_dataset = re_dataset_class(outputs, data_processor=self.data_processor,
                                                   tokenizer=self.tokenizer, max_length=args.max_length, mode="test")
                logits = []
                probs = []
                with torch.no_grad():
                    for item in temp_re_dataset:
                        input_ids, token_type_ids, attention_mask, flag = item
                        input_ids, token_type_ids, attention_mask, flag = input_ids.to(args.device), \
                                                                          token_type_ids.to(args.device), \
                                                                          attention_mask.to(args.device), \
                                                                          flag.to(args.device)
                        logit = model(input_ids=input_ids.view(1, -1), token_type_ids=token_type_ids.view(1, -1),
                                          attention_mask=attention_mask.view(1, -1),
                                          flag=flag.view(1, -1))  # batch, labels
                        
                        prob = round(nn.functional.softmax(logit).max().item(),5)
                        probs.append(prob)
                        logit = logit.argmax(dim=-1).squeeze(-1)  # batch,
                        logits.append(logit.detach().cpu().item())
                for i in range(len(temp_re_dataset)):
                    if logits[i] > 0:
                        spo_list[i]['predicate'] = self.data_processor.id2predicate[logits[i]]
                        spo_list[i]['prob'] = probs[i]

                new_spo_list = []
                for spo in spo_list:
                    if 'predicate' in spo.keys():
                        combined = True
                        for text in data['text'].split("。"):
                            if spo['object'] in text and spo['subject'] in text:
                                combined = False
                                break
                        tmp = {}
                        tmp['prob'] = spo['prob']
                        tmp['Combined'] = combined
                        tmp['predicate'] = spo['predicate'].split('|')[0]
                        tmp['subject'] = spo['subject']
                        tmp['subject_type'] = self.data_processor.pre_sub_obj[spo['predicate']][0]
                        tmp['object'] = {'@value': spo['object']}
                        tmp['object_type'] = {'@value': self.data_processor.pre_sub_obj[spo['predicate']][1]}
                        new_spo_list.append(tmp)

                new_spo_list2 = []  # 去重
                for s in new_spo_list:
                    if s not in new_spo_list2:
                        new_spo_list2.append(s)

                for i in range(len(new_spo_list2)):
                    if 'object' not in new_spo_list2[i].keys():
                        del new_spo_list2[i]

                tmp_result = dict()
                tmp_result['text'] = data['text']
                tmp_result['spo_list'] = new_spo_list2
                f.write(tmp_result)
        
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_per_example = elapsed_time / len(test_samples)
            print('总时间：', elapsed_time)
            print('样本数：', len(test_samples))
            print('每个样本所花时间：', time_per_example)
    
class GPLinkerTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPLinkerTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )

    def training_step(self, model, item):
        model.train()
#         print(type(item))
#         print(item)
        device = self.args.device
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device),\
                batch_entity_labels.to(device), batch_head_labels.to(device), batch_tail_labels.to(device)
        logits1, logits2, logits3 = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
#         print('pred形状：', logits1.shape, 'true形状：', batch_entity_labels.shape)
#         print('\n实体：', batch_entity_labels.shape, logits1.shape)
#         print('关系头：', batch_head_labels.shape, logits2.shape)
#         print('关系尾：', batch_tail_labels.shape, logits3.shape)
        loss1 = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1, mask_zero=True)
        loss2 = self.sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2, mask_zero=True)
        loss3 = self.sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3, mask_zero=True)
        loss = sum([loss1, loss2, loss3]) / 3
        loss.backward()

        p1, r1, f11 = self.cal_prf1(logits1, batch_entity_labels)
        p2, r2, f12 = self.cal_prf1(logits2, batch_head_labels)
        p3, r3, f13 = self.cal_prf1(logits3, batch_tail_labels)
#         print('训练准确率：', p1,p2,p3)
        p = (p1 + p2 + p3) / 3 
        r = (r1 + r2 + r3) / 3
        f1 = (f11 + f12 + f13) / 3
        return loss.detach(), p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def sparse_multilabel_categorical_crossentropy(self, y_true=None, y_pred=None, mask_zero=False):
        '''
        稀疏多标签交叉熵损失的torch实现
        '''
        shape = y_pred.shape
        y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
        y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
        zeros = torch.zeros_like(y_pred[...,:1])
        y_pred = torch.cat([y_pred, zeros], dim=-1)
        if mask_zero:
            infs = zeros + 1e12
            y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
        y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
        if mask_zero:
            y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
            y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
        pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
        all_loss = torch.logsumexp(y_pred, dim=-1)
        aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
        aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
        neg_loss = all_loss + torch.log(aux_loss)
        loss = torch.mean(torch.sum(pos_loss + neg_loss))
        return loss

    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device),\
                    batch_entity_labels.to(device), batch_head_labels.to(device), batch_tail_labels.to(device)
            logits1, logits2, logits3 = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss1 = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1, mask_zero=True)
            loss2 = self.sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2, mask_zero=True)
            loss3 = self.sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3, mask_zero=True)
            loss = sum([loss1, loss2, loss3]) / 3

        p1, r1, f11 = self.cal_prf1(logits1, batch_entity_labels)
        p2, r2, f12 = self.cal_prf1(logits2, batch_head_labels)
        p3, r3, f13 = self.cal_prf1(logits3, batch_tail_labels)
        p = (p1 + p2 + p3) / 3 
        r = (r1 + r2 + r3) / 3
        f1 = (f11 + f12 + f13) / 3
        return p, r, f1
        

    def predict(self, test_dataset):
        args = self.args
        logger = self.logger
        model = self.model
        device = args.device
        num_examples = len(test_dataset)
        id2predicate = self.data_processor.id2predicate
        model.to(device)

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
            start_time = time.time()
            for step, text in enumerate(test_dataset):
                pbar(step)
                model.eval()
                token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=args.max_length)["offset_mapping"]
                new_span, entities = [], []
                for i in token2char_span_mapping:
                    if i[0] == i[1]:
                        new_span.append([])
                    else:
                        if i[0] + 1 == i[1]:
                            new_span.append([i[0]])
                        else:
                            new_span.append([i[0], i[-1] - 1])
                threshold = 0.0
                encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length)
                input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
                token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
                attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
                scores = model(input_ids, attention_mask, token_type_ids)
                outputs = [o[0].data.cpu().numpy() for o in scores]
                subjects, objects = set(), set()
                outputs[0][:, [0, -1]] -= np.inf
                outputs[0][:, :, [0, -1]] -= np.inf
#                 outputs[0][0]=1
                for l, h, t in zip(*np.where(outputs[0] > 0)):
                    if l == 0:
                        subjects.add((h, t))
                    else:
                        objects.add((h, t))
                spoes = set()
                for sh, st in subjects:
                    for oh, ot in objects:
                        p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
                        p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
                        ps = set(p1s) & set(p2s)
                        for p in ps:
                            prob = (torch.sigmoid(torch.tensor(outputs[1][p][sh][oh])).item() + \
                                    torch.sigmoid(torch.tensor(outputs[2][p][st][ot])).item()) / 2
                            spoes.add((
                                text[new_span[sh][0]:new_span[st][-1] + 1], id2predicate[p],
                                text[new_span[oh][0]:new_span[ot][-1] + 1], prob
                            ))
                spo_list = []
                for spo in list(spoes):
                    spo_list.append({"predicate":spo[1].split("_")[1], "object":{"@value":spo[2]}, "object_type": {"@value": spo[1].split("_")[2]},
                                     "subject":spo[0], "subject_type":spo[1].split("_")[0], 'prob':spo[3]
                                     })
                f.write({"text":text, "spo_list":spo_list})
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_per_example = elapsed_time / num_examples
            print('总时间：', elapsed_time)
            print('样本数：', num_examples)
            print('每个样本所花时间：', time_per_example)
            
    def predict_filter(self):
        args = self.args
        logger = self.logger
        model = self.model
        data_processor = self.data_processor
        schema = data_processor.schema
        tokenizer = self.tokenizer
        device = args.device
        num_examples = 4482
        id2predicate = data_processor.id2predicate
        model.to(device)
        model.eval()
        
        logger.info("***** Running prediction filter *****")
        logger.info("Num samples %d", num_examples)
        
        output_dir = os.path.join('./result_output', 'filter')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir = output_dir + '/CMeIE-V2_test.jsonl'
        read_dir = os.path.join('./result_output', 'merge', 'CMeIE-V2_test.jsonl')
        
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f, jsonlines.open(read_dir, mode='r') as test_samples:
            pbar = ProgressBar(n_total=num_examples, desc='Filtering')
            for step, data in enumerate(test_samples):
                pbar(step)
                text = data['text']
                token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=args.max_length)["offset_mapping"]
                new_span, entities = [], []
                for i in token2char_span_mapping:
                    if i[0] == i[1]:
                        new_span.append([])
                    else:
                        if i[0] + 1 == i[1]:
                            new_span.append([i[0]])
                        else:
                            new_span.append([i[0], i[-1] - 1])
                threshold = 0.0
                encoder_txt = tokenizer.encode_plus(text, max_length=args.max_length)
                input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
                token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
                attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
                scores = model(input_ids, attention_mask, token_type_ids)
                outputs = [o[0].data.cpu().numpy() for o in scores]

                dic = {'text': text, 'spo_list': []}

                for spo in data['spo_list']:
                    sub = spo['subject']
                    obj = spo['object']['@value']
                    relation_key = spo['subject_type'] + "_" + spo['predicate'] + '_' + spo['object_type']['@value']
                    if relation_key not in schema:
                        continue
                    p = schema[relation_key]
                    s = tokenizer.encode(sub, add_special_tokens=False)
                    o = tokenizer.encode(obj, add_special_tokens=False)
                    sh = data_processor.search(s, encoder_txt["input_ids"])
                    oh = data_processor.search(o, encoder_txt["input_ids"])
                    st = sh + len(s) - 1
                    ot = oh + len(o) - 1

                    if sh != -1 and oh != -1:
                        if outputs[1][p, sh, oh] > args.filter_head_threshold and outputs[2][p, st, ot] > args.filter_tail_threshold:
                            dic['spo_list'].append(spo)
                # 去重
                filter_set = set(SPO(spo) for spo in dic['spo_list'])
                dic['spo_list'] = []
                for spo in filter_set:
                    dic['spo_list'].append(spo.spo)
                f.write(dic)
                
class GPFilterTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPFilterTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )

    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_tail_labels = item
        logits1, logits2 = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)

        loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits1, mask_zero=True)
        loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits2, mask_zero=True)
        loss = sum([loss1, loss2]) / 2
        
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits1[::2],dim=-1), F.softmax(logits1[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits1[1::2],dim=-1), F.softmax(logits1[::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits2[::2],dim=-1), F.softmax(logits2[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits2[1::2],dim=-1), F.softmax(logits2[::2],dim=-1), reduction='sum')
            # ’/ 4 * self.args.rdrop_alpha‘三是公式里带的, '/ 2'是为了头尾求平均
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits1.shape[0] / 2
        
        loss.backward()

        p1, r1, f11 = self.cal_prf1(logits1, batch_head_labels)
        p2, r2, f12 = self.cal_prf1(logits2, batch_tail_labels)
        p = (p1 + p2) / 2 
        r = (r1 + r2) / 2
        f1 = (f11 + f12) / 2
        return loss.detach(), p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
                
    def predict_filter(self, read_dir):
        args = self.args
        logger = self.logger
        model = self.model
        data_processor = self.data_processor
        schema = data_processor.schema
        tokenizer = self.tokenizer
        device = args.device
        num_examples = 4482
        id2predicate = data_processor.id2predicate
        model.to(device)
        model.eval()
        
        logger.info("***** Running prediction filter *****")
        logger.info("Num samples %d", num_examples)
        
        output_dir = os.path.join('./result_output', 'filter')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir = output_dir + '/CMeIE-V2_test.jsonl'
#         read_dir = os.path.join('./result_output', 'merge', 'CMeIE-V2_test.jsonl')
        
        logger.info(f"***** write predict file to {output_dir} *****")
        print('read_dir:', read_dir)
        with jsonlines.open(output_dir, mode='w') as f, jsonlines.open(read_dir, mode='r') as test_samples:
            pbar = ProgressBar(n_total=num_examples, desc='Filtering')
            for step, data in enumerate(test_samples):
                pbar(step)
                text = data['text']
                token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=args.max_length)["offset_mapping"]
                new_span, entities = [], []
                for i in token2char_span_mapping:
                    if i[0] == i[1]:
                        new_span.append([])
                    else:
                        if i[0] + 1 == i[1]:
                            new_span.append([i[0]])
                        else:
                            new_span.append([i[0], i[-1] - 1])
                threshold = 0.0
                encoder_txt = tokenizer.encode_plus(text, max_length=args.max_length)
                input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
                token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
                attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
                scores = model(input_ids, attention_mask, token_type_ids)
                outputs = [o[0].data.cpu().numpy() for o in scores]

                dic = {'text': text, 'spo_list': []}

                for spo in data['spo_list']:
                    sub = spo['subject']
                    obj = spo['object']['@value']
                    relation_key = spo['subject_type'] + "_" + spo['predicate'] + '_' + spo['object_type']['@value']
                    if relation_key not in schema:
                        continue
                    p = schema[relation_key]
                    s = tokenizer.encode(sub, add_special_tokens=False)
                    o = tokenizer.encode(obj, add_special_tokens=False)
                    sh = data_processor.search(s, encoder_txt["input_ids"])
                    oh = data_processor.search(o, encoder_txt["input_ids"])
                    st = sh + len(s) - 1
                    ot = oh + len(o) - 1

                    if sh != -1 and oh != -1:
                        # 之前的预测结果不带 prob 字段，因此代码需要兼容
                        # and self.data_processor.regular(spo):
                        if (outputs[0][p, sh, oh] > args.filter_head_threshold and outputs[1][p, st, ot] > args.filter_tail_threshold):
                            if 'prob' in spo.keys():
                                del spo['prob']
                            dic['spo_list'].append(spo)
                        if 'prob' in spo.keys() and spo['prob'] > args.predict_threshold:
                            dic['spo_list'].append(spo)
                # 去重
                filter_set = set(SPO(spo) for spo in dic['spo_list'])
                dic['spo_list'] = []
                for spo in filter_set:
                    dic['spo_list'].append(spo.spo)
                f.write(dic)
                
    def predict_gpner2_filter(self, mode='gpner2'):
        args = self.args
        logger = self.logger
        model = self.model
        data_processor = self.data_processor
        schema = data_processor.schema
        tokenizer = self.tokenizer
        device = args.device
        num_examples = 4482
        id2predicate = data_processor.id2predicate
        predicate2id = data_processor.predicate2id

        model.to(device)
        model.eval()
        
        logger.info("***** Running prediction filter *****")
        logger.info("Num samples %d", num_examples)
        if mode == 'gpner2':
            output_dir = os.path.join('./result_output', mode)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        else:
            output_dir = os.path.join('./result_output', mode, 'entity_filter')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        output_dir = output_dir + '/CMeIE-V2_test.jsonl'
        read_dir = os.path.join('./result_output', mode, 'entity_list.jsonl')
        
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f, jsonlines.open(read_dir, mode='r') as test_samples:
            pbar = ProgressBar(n_total=num_examples, desc='Filtering')
            for step, data in enumerate(test_samples):
                pbar(step)
                text = data['text']
                token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=args.max_length)["offset_mapping"]
                new_span, entities = [], []
                for i in token2char_span_mapping:
                    if i[0] == i[1]:
                        new_span.append([])
                    else:
                        if i[0] + 1 == i[1]:
                            new_span.append([i[0]])
                        else:
                            new_span.append([i[0], i[-1] - 1])
                threshold = 0.0
                encoder_txt = tokenizer.encode_plus(text, max_length=args.max_length)
                input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
                token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
                attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
                scores = model(input_ids, attention_mask, token_type_ids)
#                 print(len(scores))
#                 print([s.shape for s in scores])

                outputs = [o[0].data.cpu().numpy() for o in scores]

                dic = {'text': text, 'spo_list': []}
                for sub in data['subject_list']:
                    for obj in data['object_list']:
                        s = tokenizer.encode(sub, add_special_tokens=False)
                        o = tokenizer.encode(obj, add_special_tokens=False)
                        sh = data_processor.search(s, encoder_txt["input_ids"])
                        oh = data_processor.search(o, encoder_txt["input_ids"])
                        st = sh + len(s) - 1
                        ot = oh + len(o) - 1
                        if sh != -1 and oh != -1:
                            # 之前的预测结果不带 prob 字段，因此代码需要兼容
                            # and self.data_processor.regular(spo):
#                             for entity_type, h, t in zip(*np.where(outputs > threshold)):
#                                 entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
                            relation_head_set = set(np.where(outputs[0][:,sh,oh]>args.filter_head_threshold)[0])
                            relation_tail_set = set(np.where(outputs[0][:,sh,oh]>args.filter_tail_threshold)[0])
                            relations = list(relation_head_set & relation_tail_set)
                            for relation in relations:
                                relation = id2predicate[relation]
                                subject_type = relation.split('_')[0]
                                predicate = relation.split('_')[1]
                                object_type = relation.split('_')[2]
                                dic['spo_list'].append({
                                    'predicate': predicate,
                                    'subject': sub,
                                    'subject_type': subject_type,
                                    'object': {'@value': obj},
                                    'object_type': {'@value': object_type}
                                })


#                             if 'prob' in spo.keys() and spo['prob'] > args.predict_threshold:
#                                 dic['spo_list'].append(spo)
                # 去重
                filter_set = set(SPO(spo) for spo in dic['spo_list'])
                dic['spo_list'] = []
                for spo in filter_set:
                    dic['spo_list'].append(spo.spo)
                f.write(dic)
                
class GPNERTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNERTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            if args.is_reina:
                for sample in test_samples:
                    sample["text"] = self.data_processor.add_reina_prefix_0(sample["text"], sample["bm25_list"])
#                     print('sub:', sample["text"])
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_sp2o(self, mode='test'):
        args = self.args
        logger = self.logger
        with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
            samples = []
            for line in lines:
                samples.append(line)
        num_examples = len(samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, f'CMeIE-V2_{mode}.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        pbar = ProgressBar(n_total=num_examples, desc='Predicting')
                
        with jsonlines.open(output_dir, mode='w') as f:
            for step, sample in enumerate(samples):
                pbar(step)
                text = sample['text']
                test_samples = []
                for entity_dic in sample['entity_list']:
                    entity = entity_dic['entity']
                    entity_type = entity_dic['entity_type']
                    for predicate in self.data_processor.subject_predicate_dic[entity_type]:
                        test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False) 
                # 最终预测的结果
                dic = {'text': text, 'spo_list': []}
                for data in predict_data1:
                    subject = data['entity']
                    subject_type = data['entity_type']
                    predicate = data['predicate']
                    for entity_dic in data['entity_list']:
                        entity = entity_dic['entity']
                        entity_type = entity_dic['entity_type']
                        dic['spo_list'].append({'predicate': predicate, 'subject': subject, 'subject_type': subject_type,
                                                'object': {'@value': entity} , 'object_type': {'@value': entity_type}})
                f.write(dic)
    
    def predict(self):
        args = self.args
        logger = self.logger
        self.predict_entity()
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
    
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='dev')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_op2s_sp2o_dual(self):
        args = self.args
        logger = self.logger
#         with jsonlines.open(os.path.join('./result_output', 'gpner9', 'CMeIE_test-dual-1.jsonl'), mode='r') as lines:
        with jsonlines.open(os.path.join('./result_output', 'dualgpner', 'CMeIE_test-dual-1.jsonl'), mode='r') as lines:
            op2s_data = [line for line in lines]
        sp2o_dual_data = self.predict_xp2x(is_sp2o=True, op2s_result=op2s_data)
#         output_dir = os.path.join('./result_output', 'gpner9', 'CMeIE_test-dual.jsonl')
        output_dir = os.path.join('./result_output', 'dualgpner', 'CMeIE_test-dual-2.jsonl')
        num_examples = 4482
        logger.info("***** Running Dual op2s-sp20 *****")
        logger.info("Num samples %d", num_examples)
        logger.info(f"***** write predict file to {output_dir} *****")
        pbar = ProgressBar(n_total=num_examples, desc='Dual')
        with jsonlines.open(output_dir, mode='w') as f:
            for step, (sp2o, op2s) in enumerate(zip(op2s_data, sp2o_dual_data)):
                pbar(step)
                dic = {'text': sp2o['text'], 'spo_list': []}
#                 for spo in (set(SPO(spo) for spo in sp2o['spo_list']) & set(SPO(spo) for spo in op2s['spo_list'])):
                for spo in (set(SPO(spo) for spo in sp2o['spo_list']) | set(SPO(spo) for spo in op2s['spo_list'])):
                    dic['spo_list'].append(spo.spo)
                f.write(dic)

    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if op2s_result == None:
                    for predicate in predicate_dic[entity_type]:
                        test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
#                         print('sp2o:', self.data_processor.add_reina_prefix_12(prefix_text, line['bm25_list'], entity, predicate))
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result
    
#     def predict_op2s_dual(self):
#         args = self.args
#         logger = self.logger
#         logger.info("***** Running op2s_dual *****")
#         logger.info("Num samples %d", 4482)
#         output_dir = os.path.join(args.result_output_dir, 'op2s_sp20.jsonl')
#         logger.info(f"***** write predict file to {output_dir} *****")
#         op2s_file = os.path.join('./result_output', 'gpner9', 'CMeIE-V2_test.jsonl')
#         with jsonlines.open(output_dir, mode='w') as f, jsonlines.open(op2s_file, mode='r') as r:

class GPNER3Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER3Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            if args.is_reina:
                for sample in test_samples:
                    sample["text"] = self.data_processor.add_reina_prefix_0(sample["text"], sample["bm25_list"])
#                     print('sub:', sample["text"])
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_sp2o(self, mode='test'):
        args = self.args
        logger = self.logger
        with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
            samples = []
            for line in lines:
                samples.append(line)
        num_examples = len(samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, f'CMeIE-V2_{mode}.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        pbar = ProgressBar(n_total=num_examples, desc='Predicting')
                
        with jsonlines.open(output_dir, mode='w') as f:
            for step, sample in enumerate(samples):
                pbar(step)
                text = sample['text']
                test_samples = []
                for entity_dic in sample['entity_list']:
                    entity = entity_dic['entity']
                    entity_type = entity_dic['entity_type']
                    for predicate in self.data_processor.subject_predicate_dic[entity_type]:
                        test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False) 
                # 最终预测的结果
                dic = {'text': text, 'spo_list': []}
                for data in predict_data1:
                    subject = data['entity']
                    subject_type = data['entity_type']
                    predicate = data['predicate']
                    for entity_dic in data['entity_list']:
                        entity = entity_dic['entity']
                        entity_type = entity_dic['entity_type']
                        dic['spo_list'].append({'predicate': predicate, 'subject': subject, 'subject_type': subject_type,
                                                'object': {'@value': entity} , 'object_type': {'@value': entity_type}})
                f.write(dic)
    
    def predict(self):
        args = self.args
        logger = self.logger
        self.predict_entity()
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
    
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='dev')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_op2s_sp2o_dual(self):
        args = self.args
        logger = self.logger
#         with jsonlines.open(os.path.join('./result_output', 'gpner9', 'CMeIE_test-dual-1.jsonl'), mode='r') as lines:
        with jsonlines.open(os.path.join('./result_output', 'dualgpner', 'CMeIE_test-dual-1.jsonl'), mode='r') as lines:
            op2s_data = [line for line in lines]
        sp2o_dual_data = self.predict_xp2x(is_sp2o=True, op2s_result=op2s_data)
#         output_dir = os.path.join('./result_output', 'gpner9', 'CMeIE_test-dual.jsonl')
        output_dir = os.path.join('./result_output', 'dualgpner', 'CMeIE_test-dual-2.jsonl')
        num_examples = 4482
        logger.info("***** Running Dual op2s-sp20 *****")
        logger.info("Num samples %d", num_examples)
        logger.info(f"***** write predict file to {output_dir} *****")
        pbar = ProgressBar(n_total=num_examples, desc='Dual')
        with jsonlines.open(output_dir, mode='w') as f:
            for step, (sp2o, op2s) in enumerate(zip(op2s_data, sp2o_dual_data)):
                pbar(step)
                dic = {'text': sp2o['text'], 'spo_list': []}
#                 for spo in (set(SPO(spo) for spo in sp2o['spo_list']) & set(SPO(spo) for spo in op2s['spo_list'])):
                for spo in (set(SPO(spo) for spo in sp2o['spo_list']) | set(SPO(spo) for spo in op2s['spo_list'])):
                    dic['spo_list'].append(spo.spo)
                f.write(dic)

    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if op2s_result == None:
                    for predicate in predicate_dic[entity_type]:
                        prefix_text = self.data_processor.add_prefix(text, entity, predicate, entity_type)
                        test_samples.append({'text': prefix_text if args.add_prefix else text,
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
#                         print('sp2o:', self.data_processor.add_reina_prefix_12(prefix_text, line['bm25_list'], entity, predicate))
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type) if args.add_prefix else text,
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result

class GPNER4Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER4Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_op2s(self, mode='test'):
        args = self.args
        logger = self.logger
        with jsonlines.open(os.path.join(args.result_output_dir, 'object_list.jsonl'), mode='r') as lines:
            samples = []
            for line in lines:
                samples.append(line)
        num_examples = len(samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, f'CMeIE-V2_{mode}.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        pbar = ProgressBar(n_total=num_examples, desc='Predicting')
                
        with jsonlines.open(output_dir, mode='w') as f:
            for step, sample in enumerate(samples):
                pbar(step)
                text = sample['text']
                test_samples = []
                for entity_dic in sample['entity_list']:
                    entity = entity_dic['entity']
                    entity_type = entity_dic['entity_type']
                    for predicate in self.data_processor.object_predicate_dic[entity_type]:
                        test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False) 
                # 最终预测的结果
                dic = {'text': text, 'spo_list': []}
                for data in predict_data1:
                    obj = data['entity']
                    object_type = data['entity_type']
                    predicate = data['predicate']
                    for entity_dic in data['entity_list']:
                        entity = entity_dic['entity']
                        entity_type = entity_dic['entity_type']
                        dic['spo_list'].append({'predicate': predicate, 'subject': entity, 'subject_type': entity_type,
                                                'object': {'@value': obj} , 'object_type': {'@value': object_type}})
                f.write(dic)
    
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if sp2o_result == None:
                    for predicate in predicate_dic[entity_type]:
                        test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type) if args.add_prefix else text,
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type) if args.add_prefix else text,
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                    
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self):
        args = self.args
        logger = self.logger
        self.predict_entity()
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='dev')
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
    
    def predict_sp2o_op2s_dual(self):
        args = self.args
        logger = self.logger
        with jsonlines.open(os.path.join('./result_output', 'dualgpner', 'CMeIE_test-f1-57.564.jsonl'), mode='r') as lines:
#         with jsonlines.open(os.path.join('./result_output', 'gpner9', 'CMeIE_test-dual-1.jsonl'), mode='r') as lines:
            sp2o_data = [line for line in lines]
        op2s_dual_data = self.predict_xp2x(is_sp2o=False, sp2o_result=sp2o_data)
        output_dir = os.path.join('./result_output', 'dualgpner', 'CMeIE_test-dual-1.jsonl')
#         output_dir = os.path.join('./result_output', 'gpner9', 'CMeIE_test-dual-2.jsonl')
        num_examples = 4482
        logger.info("***** Running Dual sp20-op2s *****")
        logger.info("Num samples %d", num_examples)
        logger.info(f"***** write predict file to {output_dir} *****")
        pbar = ProgressBar(n_total=num_examples, desc='Dual')
        with jsonlines.open(output_dir, mode='w') as f:
            for step, (sp2o, op2s) in enumerate(zip(sp2o_data, op2s_dual_data)):
                pbar(step)
                dic = {'text': sp2o['text'], 'spo_list': []}
#                 for spo in (set(SPO(spo) for spo in sp2o['spo_list']) & set(SPO(spo) for spo in op2s['spo_list'])):
                for spo in (set(SPO(spo) for spo in sp2o['spo_list']) | set(SPO(spo) for spo in op2s['spo_list'])):
                    dic['spo_list'].append(spo.spo)
                f.write(dic)
        
    def predict_dual(self, mode='all'):
        args = self.args
        if mode == 'all':
            file_name = 'CMeIE_test-dual.jsonl'
        elif mode == 'half':
            file_name = 'CMeIE-V2_test.jsonl'
        sp2o_file = os.path.join('./result_output', 'dualgpner', file_name)
        op2s_file = os.path.join(args.result_output_dir, file_name)
        result_output_dir = os.path.join(args.result_output_dir, f'{mode}_dual')
        if not os.path.exists(result_output_dir):
            os.mkdir(result_output_dir)
        output_dir = os.path.join(result_output_dir, 'CMeIE-V2_test.jsonl')
        pbar = ProgressBar(n_total=4482, desc='Dual')
        
        with jsonlines.open(output_dir, mode='w') as f:
            with jsonlines.open(sp2o_file, mode='r') as sp2o_data, jsonlines.open(op2s_file, mode='r') as op2s_data:
                for step, (sp2o, op2s) in enumerate(zip(sp2o_data, op2s_data)):
                    pbar(step)
                    dic = {'text': sp2o['text'], 'spo_list': []}
                    for spo in (set(SPO(spo) for spo in sp2o['spo_list']) & set(SPO(spo) for spo in op2s['spo_list'])):
                        dic['spo_list'].append(spo.spo)
                    f.write(dic)
    
class GPNER9Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER9Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_op2s(self, mode='test'):
        args = self.args
        logger = self.logger
        with jsonlines.open(os.path.join(args.result_output_dir, 'object_list.jsonl'), mode='r') as lines:
            samples = []
            for line in lines:
                samples.append(line)
        num_examples = len(samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, f'CMeIE-V2_{mode}.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        pbar = ProgressBar(n_total=num_examples, desc='Predicting')
                
        with jsonlines.open(output_dir, mode='w') as f:
            for step, sample in enumerate(samples):
                pbar(step)
                text = sample['text']
                test_samples = []
                for entity_dic in sample['entity_list']:
                    entity = entity_dic['entity']
                    entity_type = entity_dic['entity_type']
                    for predicate in self.data_processor.object_predicate_dic[entity_type]:
                        test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False) 
                # 最终预测的结果
                dic = {'text': text, 'spo_list': []}
                for data in predict_data1:
                    obj = data['entity']
                    object_type = data['entity_type']
                    predicate = data['predicate']
                    for entity_dic in data['entity_list']:
                        entity = entity_dic['entity']
                        entity_type = entity_dic['entity_type']
                        dic['spo_list'].append({'predicate': predicate, 'subject': entity, 'subject_type': entity_type,
                                                'object': {'@value': obj} , 'object_type': {'@value': object_type}})
                f.write(dic)
    
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if sp2o_result == None:
                    for predicate in predicate_dic[entity_type]:
                        test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                    
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self):
        args = self.args
        logger = self.logger
        self.predict_entity()
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='dev')
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
    
    def predict_sp2o_op2s_dual(self):
        args = self.args
        logger = self.logger
        with jsonlines.open(os.path.join('./result_output', 'dualgpner', 'CMeIE_test-f1-57.564.jsonl'), mode='r') as lines:
#         with jsonlines.open(os.path.join('./result_output', 'gpner9', 'CMeIE_test-dual-1.jsonl'), mode='r') as lines:
            sp2o_data = [line for line in lines]
        op2s_dual_data = self.predict_xp2x(is_sp2o=False, sp2o_result=sp2o_data)
        output_dir = os.path.join('./result_output', 'dualgpner', 'CMeIE_test-dual-1.jsonl')
#         output_dir = os.path.join('./result_output', 'gpner9', 'CMeIE_test-dual-2.jsonl')
        num_examples = 4482
        logger.info("***** Running Dual sp20-op2s *****")
        logger.info("Num samples %d", num_examples)
        logger.info(f"***** write predict file to {output_dir} *****")
        pbar = ProgressBar(n_total=num_examples, desc='Dual')
        with jsonlines.open(output_dir, mode='w') as f:
            for step, (sp2o, op2s) in enumerate(zip(sp2o_data, op2s_dual_data)):
                pbar(step)
                dic = {'text': sp2o['text'], 'spo_list': []}
#                 for spo in (set(SPO(spo) for spo in sp2o['spo_list']) & set(SPO(spo) for spo in op2s['spo_list'])):
                for spo in (set(SPO(spo) for spo in sp2o['spo_list']) | set(SPO(spo) for spo in op2s['spo_list'])):
                    dic['spo_list'].append(spo.spo)
                f.write(dic)
        
    def predict_dual(self, mode='all'):
        args = self.args
        if mode == 'all':
            file_name = 'CMeIE_test-dual.jsonl'
        elif mode == 'half':
            file_name = 'CMeIE-V2_test.jsonl'
        sp2o_file = os.path.join('./result_output', 'dualgpner', file_name)
        op2s_file = os.path.join(args.result_output_dir, file_name)
        result_output_dir = os.path.join(args.result_output_dir, f'{mode}_dual')
        if not os.path.exists(result_output_dir):
            os.mkdir(result_output_dir)
        output_dir = os.path.join(result_output_dir, 'CMeIE-V2_test.jsonl')
        pbar = ProgressBar(n_total=4482, desc='Dual')
        
        with jsonlines.open(output_dir, mode='w') as f:
            with jsonlines.open(sp2o_file, mode='r') as sp2o_data, jsonlines.open(op2s_file, mode='r') as op2s_data:
                for step, (sp2o, op2s) in enumerate(zip(sp2o_data, op2s_data)):
                    pbar(step)
                    dic = {'text': sp2o['text'], 'spo_list': []}
                    for spo in (set(SPO(spo) for spo in sp2o['spo_list']) & set(SPO(spo) for spo in op2s['spo_list'])):
                        dic['spo_list'].append(spo.spo)
                    f.write(dic)
                    
class GPNER2Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER2Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
     
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
            threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            subjects, objects = set(), set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for l, h, t in zip(*np.where(outputs > 0)):
                if l == 0:
                    subjects.add((h, t))
                else:
                    objects.add((h, t))

            subject_list, object_list = [], []
            for sh, st in subjects:
                subject_list.append(text[new_span[sh][0]:new_span[st][-1] + 1])
            for oh, ot in objects:
                object_list.append(text[new_span[oh][0]:new_span[ot][-1] + 1])
            data['subject_list'] = subject_list
            data['object_list'] = object_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self):
        args = self.args
        logger = self.logger
        test_samples = self.data_processor.get_test_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples)
            for data in predict_data0:
                f.write(data)
                
    def predict(self):
        self.predict_entity()
        
class GPNER5SubTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER5SubTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            if args.is_reina:
                for sample in test_samples:
                    sample["text"] = self.data_processor.add_reina_prefix_0(sample["text"], sample["bm25_list"])
#                     print('sub:', sample["text"])
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict(self):
        args = self.args
        logger = self.logger
        self.predict_entity()
    
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='dev')
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
                
            
class GPNER5Sp2oTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER5Sp2oTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
            
    
    def predict(self):
        args = self.args
        logger = self.logger
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
    
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='dev')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                

    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if op2s_result == None:
                    for predicate in predicate_dic[entity_type]:
                        prefix_text = self.data_processor.add_prefix(text, entity, predicate, entity_type)
                        test_samples.append({'text': self.data_processor.add_reina_prefix_12(prefix_text, sample['bm25_list'], entity, predicate) if self.args.is_reina \
                                                     else prefix_text,
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
#                         print('sp2o:', self.data_processor.add_reina_prefix_12(prefix_text, line['bm25_list'], entity, predicate))
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result
    
class GPNER6ObjTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER6ObjTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                

    def predict(self):
        args = self.args
        logger = self.logger
        self.predict_entity()
                
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='dev')

                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
    
         
class GPNER6Op2sTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER6Op2sTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
        
    
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if sp2o_result == None:
                    for predicate in predicate_dic[entity_type]:
                        test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                    
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self):
        args = self.args
        logger = self.logger
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_dev(self):
        args = self.args
        logger = self.logger
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
    
class GPNER7Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER7Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            if args.is_reina:
                for sample in test_samples:
                    sample["text"] = self.data_processor.add_reina_prefix_0(sample["text"], sample["bm25_list"])
#                     print('sub:', sample["text"])
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict(self):
        args = self.args
        logger = self.logger
        self.predict_entity()
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
    
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='dev')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = self.data_processor.del_entity_marker(data['entity']) if args.do_entity_marker else data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result
    
class GPNER8Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER8Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
        
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if sp2o_result == None:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                    

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = self.data_processor.del_entity_marker(data['entity']) if args.do_entity_marker else data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicate = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicate:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity,
                                            'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicate:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                            'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self):
        args = self.args
        logger = self.logger
        self.predict_entity()
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='dev')
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
    
class GPFilter78Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPFilter78Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )

    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_tail_labels = item
        logits1, logits2 = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)

        loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits1, mask_zero=True)
        loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits2, mask_zero=True)
        loss = sum([loss1, loss2]) / 2
        
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits1[::2],dim=-1), F.softmax(logits1[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits1[1::2],dim=-1), F.softmax(logits1[::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits2[::2],dim=-1), F.softmax(logits2[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits2[1::2],dim=-1), F.softmax(logits2[::2],dim=-1), reduction='sum')
            # ’/ 4 * self.args.rdrop_alpha‘三是公式里带的, '/ 2'是为了头尾求平均
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits1.shape[0] / 2
        
        loss.backward()

        p1, r1, f11 = self.cal_prf1(logits1, batch_head_labels)
        p2, r2, f12 = self.cal_prf1(logits2, batch_tail_labels)
        p = (p1 + p2) / 2 
        r = (r1 + r2) / 2
        f1 = (f11 + f12) / 2
        return loss.detach(), p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
                
    def predict_filter(self, read_dir=None, mode='test', epoch=None):
        args = self.args
        logger = self.logger
        model = self.model
        data_processor = self.data_processor
        schema = data_processor.schema
        tokenizer = self.tokenizer
        device = args.device
        
        id2predicate = data_processor.id2predicate
        predicate2id = data_processor.predicate2id
        model.to(device)
        model.eval()
        
        if not read_dir:
            read_dir = os.path.join('./result_output', 'merge', 'CMeIE-V2_test.jsonl')
        print(f'load test data from {read_dir}')
        
        with jsonlines.open(read_dir, mode='r') as r:
            test_samples = [line for line in r]
        num_examples = len(test_samples)
        
        logger.info("***** Running prediction filter *****")
        logger.info("Num samples %d", num_examples)
        
        if epoch == None:
            output_dir = os.path.join('./result_output', 'filter78')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_dir = output_dir + f'/CMeIE-V2_{mode}.jsonl'
        else:
            output_dir = os.path.join(self.output_dir, f'CMeIE-V2_test-epoch{epoch}.jsonl')
        
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            pbar = ProgressBar(n_total=num_examples, desc='Filtering')
            start_time = time.time()
            for step, data in enumerate(test_samples):
                pbar(step)
                text = data['text']
                token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=args.max_length)["offset_mapping"]
                new_span, entities = [], []
                for i in token2char_span_mapping:
                    if i[0] == i[1]:
                        new_span.append([])
                    else:
                        if i[0] + 1 == i[1]:
                            new_span.append([i[0]])
                        else:
                            new_span.append([i[0], i[-1] - 1])
                threshold = 0.0
                encoder_txt = tokenizer.encode_plus(text, max_length=args.max_length)
                input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
                token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
                attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
                scores = model(input_ids, attention_mask, token_type_ids)
                outputs = [o[0].data.cpu().numpy() for o in scores]

                dic = {'text': text, 'spo_list': []}

                for spo in data['spo_list']:
                    sub = spo['subject']
                    obj = spo['object']['@value']
                    relation_key = spo['predicate']
                    if relation_key not in schema:
                        continue
                    p = predicate2id[relation_key]
                    input_ids = encoder_txt["input_ids"]
                    if args.do_filter_search_all:
                        sub_tokens = self.tokenizer.encode(sub, add_special_tokens=False)
                        obj_tokens = self.tokenizer.encode(obj, add_special_tokens=False)
                        shs = self.data_processor.search_all(sub_tokens, input_ids)
                        ohs = self.data_processor.search_all(obj_tokens, input_ids)
                        if shs == []:
                            sub_tokens = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                            shs = self.data_processor.search_all(sub_tokens, input_ids)
                        if ohs == []:
                            obj_tokens = self.tokenizer.encode(' '+obj, add_special_tokens=False)
                            ohs = self.data_processor.search_all(obj_tokens, input_ids)
                        for sh in shs:
                            for oh in ohs:
#                                 print(sh,oh)
                                st = sh+len(sub_tokens)-1
                                ot = oh+len(obj_tokens)-1
                                if (outputs[0][p, sh, oh] > args.filter_head_threshold and outputs[1][p, st, ot] > args.filter_tail_threshold) or (args.do_is_inter and spo['is_inter'] and outputs[0][p, sh, oh] > args.inter_filter_head_threshold and outputs[1][p, st, ot] > args.inter_filter_tail_threshold):
                                    dic['spo_list'].append(spo)
                            
                    else:
                        s = tokenizer.encode(sub, add_special_tokens=False)
                        o = tokenizer.encode(obj, add_special_tokens=False)
                        sh = data_processor.search(s, input_ids)
                        oh = data_processor.search(o, input_ids)
                        st = sh + len(s) - 1
                        ot = oh + len(o) - 1

                        if sh != -1 and oh != -1:
                            if (outputs[0][p, sh, oh] > args.filter_head_threshold and outputs[1][p, st, ot] > args.filter_tail_threshold) or (args.do_is_inter and spo['is_inter'] and outputs[0][p, sh, oh] > args.inter_filter_head_threshold and outputs[1][p, st, ot] > args.inter_filter_tail_threshold):
                                dic['spo_list'].append(spo)
                f.write(dic)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_per_example = elapsed_time / num_examples
            print('总时间：', elapsed_time)
            print('样本数：', num_examples)
            print('每个样本所花时间：', time_per_example)
            
class GPNER17Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER17Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            

            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            if args.is_reina:
                for sample in test_samples:
                    sample["text"] = self.data_processor.add_reina_prefix_0(sample["text"], sample["bm25_list"])
#                     print('sub:', sample["text"])
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict(self):
        args = self.args
        logger = self.logger
        self.predict_entity()
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
    
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='dev')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result
    
class GPNER18Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER18Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)

            
            
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
        
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if sp2o_result == None:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                             'entity': entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': entity, 'predicate': predicate})
                    

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicate = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity,
                                            'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                            'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self):
        args = self.args
        logger = self.logger
        self.predict_entity()
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='dev')
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
        self.predict_entity(mode='train')
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
class CasRelTrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(CasRelTrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        self.loss_fn = nn.BCELoss()
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_sub_heads, batch_sub_tails, batch_sub_head, batch_sub_tail, batch_obj_heads, batch_obj_tails = item
        
        shs_feature, sts_feature, ohs_feature, ots_feature = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_sub_head, batch_sub_tail)
        active_index = batch_mask_ids.view(-1) == 1
        class_num = ohs_feature.shape[-1]
        loss = self.cal_loss(shs_feature, batch_sub_heads, active_index) +\
               self.cal_loss(sts_feature, batch_sub_tails, active_index) +\
               self.cal_loss(ohs_feature, batch_obj_heads, torch.repeat_interleave(active_index, repeats=class_num)) +\
               self.cal_loss(ots_feature, batch_obj_tails, torch.repeat_interleave(active_index, repeats=class_num))
            
        loss.backward()
        return loss.detach(), 0, 0, 0
    
    def cal_loss(self, logits, labels, active_index):
        active_labels = labels.view(-1)[active_index]
        active_logits = logits.view(-1)[active_index]
        return self.loss_fn(active_logits.float()[1:-1], active_labels.float()[1:-1])
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    

    def predict(self, mode='test'):
        args = self.args
        logger = self.logger
        id2class = self.data_processor.id2class
        model = self.model
        tokenizer = self.tokenizer
        device = args.device
        model.to(device)
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, f'CMeIE-V2_{mode}.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
            predict_data = []
            for step, data in enumerate(test_samples):
                text = data['text']
                pbar(step)
                model.eval()
                encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
                input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
                token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
                attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
#                 spo_list = model(input_ids, token_type_ids, attention_mask, tokens=tokenizer.tokenize(text, add_special_tokens=True), id2class=id2class)
#                 spo_list = model(input_ids, token_type_ids, attention_mask, tokens=['CLS']+list(text)+['SEP'], id2class=id2class)
                spo_list = model(input_ids, token_type_ids, attention_mask, tokens=tokenizer.tokenize(text, add_special_tokens=True), id2class=id2class, text=text)
                dic = {'text': text,
                       'spo_list': [{'predicate': p, 'subject': s, 'object':{'@value': o}} for s, p, o in spo_list]}
                f.write(dic)
            
class GPNER11Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER11Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            if args.is_reina:
                for sample in test_samples:
                    sample["text"] = self.data_processor.add_reina_prefix_0(sample["text"], sample["bm25_list"])
#                     print('sub:', sample["text"])
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict(self):
        args = self.args
        logger = self.logger
#         self.predict_entity()
#         self.predict_sp2o()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
    
    def predict_dev(self):
        args = self.args
        logger = self.logger
#         self.predict_entity(mode='dev')
#         self.predict_sp2o()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
        
                
    def predict_train(self):
        args = self.args
        logger = self.logger
#         self.predict_entity(mode='train')
#         self.predict_sp2o()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
        
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open('./result_output/gpner12/entity_list.jsonl', mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = self.data_processor.del_entity_marker(data['entity']) if args.do_entity_marker else data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result
    
class GPNER12Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER12Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
        
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            with jsonlines.open('./result_output/gpner11/entity_list.jsonl', mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if sp2o_result == None:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                    

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = self.data_processor.del_entity_marker(data['entity']) if args.do_entity_marker else data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicate = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicate:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity,
                                            'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicate:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                            'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self):
        args = self.args
        logger = self.logger
#         self.predict_entity()
#         self.predict_op2s()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
        
                
    def predict_dev(self):
        args = self.args
        logger = self.logger
#         self.predict_entity(mode='dev')
#         self.predict_op2s()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_train(self):
        args = self.args
        logger = self.logger
#         self.predict_entity(mode='train')
#         self.predict_op2s()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_train.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
class GPNER13Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER13Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, mode, isPbar=True, threshold=0.0, xp2x=False):
        """
        mode: ['sub', 'obj']，用于确定抽取主体还是抽取客体
        """
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = self.data_processor.add_prefix(data['text'], mode=mode)
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_sub_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'sub_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'sub', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_obj_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'obj_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'obj', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    

                
    
class GPNER14Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER14Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
                
    def predict_sp2o(self):
        args = self.args
        logger = self.logger
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_op2s(self):
        args = self.args
        logger = self.logger
        sp20_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
    
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        entity_file_name = 'sub_entity_list.jsonl' if is_sp2o else 'obj_entity_list.jsonl'
        if op2s_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner13', entity_file_name), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = self.data_processor.del_entity_marker(data['entity']) if args.do_entity_marker else data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result
    
class GPNER15Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER15Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, mode=None, isPbar=True, threshold=0.0, xp2x=False):
        """
        mode: ['sub', 'obj']，用于确定抽取主体还是抽取客体
        """
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            if xp2x:
                text = data['text']
            else:
                text = self.data_processor.add_entity_prefix(data['text'], mode=mode)
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
                
    def predict_sub_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'sub_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'sub', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_obj_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'obj_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'obj', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)

    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        entity_file_name = 'sub_entity_list.jsonl' if is_sp2o else 'obj_entity_list.jsonl'
        if op2s_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner15', entity_file_name), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_xp2x_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_xp2x_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = self.data_processor.del_entity_marker(data['entity']) if args.do_entity_marker else data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result
    
    def predict_sp2o(self):
        args = self.args
        logger = self.logger
        self.predict_sub_entity()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_op2s(self):
        args = self.args
        logger = self.logger
        self.predict_obj_entity()
        sp20_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
class GPNER21Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER21Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            if args.is_reina:
                for sample in test_samples:
                    sample["text"] = self.data_processor.add_reina_prefix_0(sample["text"], sample["bm25_list"])
#                     print('sub:', sample["text"])
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
    
class GPNER23Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER23Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
                
    def predict(self):
        args = self.args
        logger = self.logger
#         self.predict_entity()
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner21', 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = self.data_processor.del_entity_marker(data['entity']) if args.do_entity_marker else data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result
    
class GPNER22Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER22Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                         
class GPNER24Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER24Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner22', 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if sp2o_result == None:
                    test_samples.append({'text': self.data_processor.add_entity_marker(text,entity,predicate) if args.do_entity_marker else self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
                                         'entity': self.data_processor.entity_marker(entity) if args.do_entity_marker else entity, 'predicate': predicate})
                    

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = self.data_processor.del_entity_marker(data['entity']) if args.do_entity_marker else data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicate = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicate:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity,
                                            'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicate:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                            'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self):
        args = self.args
        logger = self.logger
#         self.predict_entity()
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
class GPNER25Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER25Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_tail_labels, batch_head_positions, batch_tail_positions = item
        logits1, logits2 = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits1, mask_zero=True)
        loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits2, mask_zero=True)
        loss = (loss1 + loss2) / 2
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p1, r1, f11 = self.cal_prf1(logits1, batch_head_labels)
        p2, r2, f12 = self.cal_prf1(logits2, batch_tail_labels)
        p = (p1 + p2) / 2
        r = (r1 + r2) / 2
        f1 = (f11 + f12) / 2
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            spo_list = model(input_ids, attention_mask, token_type_ids, text=text, id2class=id2class, new_span=new_span)
            data['spo_list'] = spo_list
            predict_data.append(data)
        return predict_data
    
    def predict(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    
    
class GPNER26Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER26Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_tail_labels, batch_head_positions, batch_tail_positions = item
        logits1, logits2 = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits1, mask_zero=True)
        loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits2, mask_zero=True)
        loss = (loss1 + loss2) / 2
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p1, r1, f11 = self.cal_prf1(logits1, batch_head_labels)
        p2, r2, f12 = self.cal_prf1(logits2, batch_tail_labels)
        p = (p1 + p2) / 2
        r = (r1 + r2) / 2
        f1 = (f11 + f12) / 2
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            spo_list = model(input_ids, attention_mask, token_type_ids, text=text, id2class=id2class, new_span=new_span)
            data['spo_list'] = spo_list
            predict_data.append(data)
        return predict_data
    
    def predict(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)

class GPNER31Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER31Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner32', 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if sp2o_result == None:
                    for predicate in predicate_dic[entity_type]:
                        test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                    
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self):
        args = self.args
        logger = self.logger
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                        
class GPNER32Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER32Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    
    
    def predict(self):
        args = self.args
        logger = self.logger
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
    
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner31', 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if op2s_result == None:
                    for predicate in predicate_dic[entity_type]:
                        prefix_text = self.data_processor.add_prefix(text, entity, predicate, entity_type)
                        test_samples.append({'text': prefix_text,
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
#                         print('sp2o:', self.data_processor.add_reina_prefix_12(prefix_text, line['bm25_list'], entity, predicate))
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result
    
class GPNER33Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER33Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, mode, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = self.data_processor.add_prefix(data['text'], mode=mode)
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_sub_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'sub_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'sub', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
    
    def predict_obj_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running object prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'obj_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'obj', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    
                    
class GPNER34Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER34Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    
    
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        entity_file_name = 'sub_entity_list.jsonl' if is_sp2o else 'obj_entity_list.jsonl'
        if sp2o_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner33', entity_file_name), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if sp2o_result == None:
                    for predicate in predicate_dic[entity_type]:
                        test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                    
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict_op2s(self):
        args = self.args
        logger = self.logger
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_sp2o(self):
        args = self.args
        logger = self.logger
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
class GPNER35Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER35Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, mode=None, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            if xp2x:
                text = data['text']
            else:
                text = self.data_processor.add_entity_prefix(data['text'], mode=mode)
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_sub_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'sub_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'sub', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
    
    def predict_obj_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running object prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'obj_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'obj', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        entity_file_name = 'sub_entity_list.jsonl' if is_sp2o else 'obj_entity_list.jsonl'
        if sp2o_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, entity_file_name), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if sp2o_result == None:
                    for predicate in predicate_dic[entity_type]:
                        test_samples.append({'text': self.data_processor.add_xp2x_prefix(text, entity, predicate, entity_type),
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_xp2x_prefix(text, entity, predicate, entity_type),
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                    
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict_op2s(self):
        args = self.args
        logger = self.logger
        self.predict_obj_entity()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_sp2o(self):
        args = self.args
        logger = self.logger
        self.predict_sub_entity()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
class GPNER36Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER36Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            if args.is_reina:
                for sample in test_samples:
                    sample["text"] = self.data_processor.add_reina_prefix_0(sample["text"], sample["bm25_list"])
#                     print('sub:', sample["text"])
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    
class GPNER38Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER38Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict(self):
        args = self.args
        logger = self.logger
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
    
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner36', 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if op2s_result == None:
                    for predicate in predicate_dic[entity_type]:
                        prefix_text = self.data_processor.add_prefix(text, entity, predicate, entity_type)
                        test_samples.append({'text': self.data_processor.add_reina_prefix_12(prefix_text, sample['bm25_list'], entity, predicate) if self.args.is_reina \
                                                     else prefix_text,
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
#                         print('sp2o:', self.data_processor.add_reina_prefix_12(prefix_text, line['bm25_list'], entity, predicate))
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result
    
class GPNER37Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER37Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                    
class GPNER39Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER39Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            score = model(input_ids, attention_mask, token_type_ids)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner37', 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if sp2o_result == None:
                    for predicate in predicate_dic[entity_type]:
                        test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate, entity_type),
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                    
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self):
        args = self.args
        logger = self.logger
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)

# R7T
class GPNER41Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER41Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            # #！
            # print("Shape of outputs:", outputs.shape)
            # print("Outputs values:", outputs)
            # indices = np.where(outputs > threshold)
            # print("np.where results:", indices)
            # zipped_results = list(zip(*np.where(outputs > threshold)))
            # print("Length of zipped results:", len(zipped_results))
            # if len(zipped_results) > 0:
            #     print("Shape of first zipped result:", len(zipped_results[0]))
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        if args.do_train:
            output_dir = os.path.join(self.output_dir, 'entity_list.jsonl')
        else:
            output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict(self, epoch=None):
        args = self.args
        logger = self.logger
        start_time = time.time()
        self.predict_entity()
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        if args.do_train:
            output_dir = os.path.join(self.output_dir, f'CMeIE-V2_test-epoch{epoch}.jsonl')
        else:
            output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_example = elapsed_time / 4482
        print('总时间：', elapsed_time)
        print('样本数：', 4482)
        print('每个样本所花时间：', time_per_example)
                
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity('dev')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            if args.do_train:
                read_file_name = os.path.join(self.output_dir, 'entity_list.jsonl')
            else:
                read_file_name = os.path.join(args.result_output_dir, 'entity_list.jsonl')
            with jsonlines.open(read_file_name, mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            temp_entity_dic = defaultdict(set)
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                temp_entity_dic[entity].add(predicate)
            for entity, predicates in temp_entity_dic.items():
                test_samples.append({'text': self.data_processor.add_prefix(text, entity), 'entity': entity, 'predicates': predicates})
#             for entity_dic in sample['entity_list']:
#                 entity = entity_dic['entity']
#                 predicate = entity_dic['predicate']
#                 if op2s_result == None:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity),
#                                          'entity': entity, 'predicate': predicate})
#                 else:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity),
#                                          'entity': entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicates = data['predicates']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicate = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or out_predicate in in_predicates:
                            dic['spo_list'].append({'predicate': out_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or out_predicate in in_predicates:
                            dic['spo_list'].append({'predicate': out_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
                            
#             for data in predict_data1:
#                 pre_entity = data['entity']
#                 in_predicate = data['predicate']
#                 for entity_dic in data['entity_list']:
#                     post_entity = entity_dic['entity']
#                     out_predicarte = entity_dic['predicate']
#                     prob = entity_dic['prob']
#                     if is_sp2o:
#                         if not args.do_predicate_verify or in_predicate == out_predicarte:
#                             dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity, 
#                                                 'object': {'@value': post_entity}, 'prob': prob})
#                     else:
#                         if not args.do_predicate_verify or in_predicate == out_predicarte:
#                             dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
#                                                 'object': {'@value': pre_entity}, 'prob': prob})
            
            result.append(dic)
        return result

# R8T
class GPNER42Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER42Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        if args.do_train:
            output_dir = os.path.join(self.output_dir, 'entity_list.jsonl')
        else:
            output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
        
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            if args.do_train:
                read_file_name = os.path.join(self.output_dir, 'entity_list.jsonl')
            else:
                read_file_name = os.path.join(args.result_output_dir, 'entity_list.jsonl')
            with jsonlines.open(read_file_name, mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            
            temp_entity_dic = defaultdict(set)
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                temp_entity_dic[entity].add(predicate)
            for entity, predicates in temp_entity_dic.items():
                test_samples.append({'text': self.data_processor.add_prefix(text, entity), 'entity': entity, 'predicates': predicates})
            
#             for entity_dic in sample['entity_list']:
#                 entity = entity_dic['entity']
#                 predicate = entity_dic['predicate']
#                 if sp2o_result == None:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity),
#                                          'entity': entity, 'predicate': predicate})
#                 else:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity),
#                                          'entity': entity, 'predicate': predicate})
                    

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicates = data['predicates']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicate = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or out_predicate in in_predicates:
                            dic['spo_list'].append({'predicate': out_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or out_predicate in in_predicates:
                            dic['spo_list'].append({'predicate': out_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            
#             for data in predict_data1:
#                 pre_entity = data['entity']
#                 in_predicate = data['predicate']
#                 for entity_dic in data['entity_list']:
#                     post_entity = entity_dic['entity']
#                     out_predicate = entity_dic['predicate']
#                     prob = entity_dic['prob']
#                     if is_sp2o:
#                         if not args.do_predicate_verify or in_predicate == out_predicate:
#                             dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity,
#                                             'object': {'@value': post_entity}, 'prob': prob})
#                     else:
#                         if not args.do_predicate_verify or in_predicate == out_predicate:
#                             dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
#                                             'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self, epoch=None):
        args = self.args
        logger = self.logger
        start_time = time.time()
        self.predict_entity()
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        if args.do_train:
            output_dir = os.path.join(self.output_dir, f'CMeIE-V2_test-epoch{epoch}.jsonl')
        else:
            output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_per_example = elapsed_time / 4482
        print('总时间：', elapsed_time)
        print('样本数：', 4482)
        print('每个样本所花时间：', time_per_example)
                
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity('dev')
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)

# R41T
class GPNER51Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER51Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            if args.is_reina:
                for sample in test_samples:
                    sample["text"] = self.data_processor.add_reina_prefix_0(sample["text"], sample["bm25_list"])
#                     print('sub:', sample["text"])
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict(self):
        args = self.args
        logger = self.logger
#         self.predict_entity()
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner52', 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result

# R41T
class GPNER52Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER52Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            if args.is_reina:
                for sample in test_samples:
                    sample["text"] = self.data_processor.add_reina_prefix_0(sample["text"], sample["bm25_list"])
#                     print('sub:', sample["text"])
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict(self):
        args = self.args
        logger = self.logger
#         self.predict_entity()
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner51', 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result
    
# R41T
class GPNER53Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER53Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, mode, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = self.data_processor.add_prefix(data['text'], mode=mode)
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_sub_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'sub_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'sub', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_obj_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running object prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'obj_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'obj', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
# R41T
class GPNER54Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER54Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
           
    def predict_sp2o(self):
        args = self.args
        logger = self.logger
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
    
    def predict_op2s(self):
        args = self.args
        logger = self.logger
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        entity_file_name = 'sub_entity_list.jsonl' if is_sp2o else 'obj_entity_list.jsonl'
        if op2s_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner53', entity_file_name), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result
    
# R41T
class GPNER55Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER55Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, mode=None, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            if xp2x:
                text = data['text']
            else:
                text = self.data_processor.add_entity_prefix(data['text'], mode=mode)
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_sub_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'sub_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'sub', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_obj_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running object prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'obj_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'obj', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_sp2o(self):
        args = self.args
        logger = self.logger
        self.predict_sub_entity()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_op2s(self):
        args = self.args
        logger = self.logger
        self.predict_obj_entity()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        entity_file_name = 'sub_entity_list.jsonl' if is_sp2o else 'obj_entity_list.jsonl'
        if op2s_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, entity_file_name), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_xp2x_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_xp2x_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result

# R41T
class GPNER56Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER56Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            if args.is_reina:
                for sample in test_samples:
                    sample["text"] = self.data_processor.add_reina_prefix_0(sample["text"], sample["bm25_list"])
#                     print('sub:', sample["text"])
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
# R41T
class GPNER58Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER58Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
            
    def predict(self):
        args = self.args
        logger = self.logger
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner56', 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result

# R42T
class GPNER57Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER57Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
        
# R42T
class GPNER59Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER59Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner57', 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if sp2o_result == None:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                    

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicate = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicate:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': pre_entity,
                                            'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicate:
                            dic['spo_list'].append({'predicate': in_predicate, 'subject': post_entity,
                                            'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self):
        args = self.args
        logger = self.logger
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
          
        
# R41T
class GPNER61Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER61Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    
# R41T
class GPNER62Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER62Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
                
    def predict(self):
        args = self.args
        logger = self.logger
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner61', 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': out_predicarte, 'subject': pre_entity, 
                                                'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': out_predicarte, 'subject': post_entity,
                                                'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result
    
# R42T
class GPNER63Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER63Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
        
# R42T
class GPNER64Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER64Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            with jsonlines.open(os.path.join('./result_output/gpner63', 'entity_list.jsonl'), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if sp2o_result == None:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                    

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicate = entity_dic['predicate']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicate:
                            dic['spo_list'].append({'predicate': out_predicate, 'subject': pre_entity,
                                            'object': {'@value': post_entity}, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicate:
                            dic['spo_list'].append({'predicate': out_predicate, 'subject': post_entity,
                                            'object': {'@value': pre_entity}, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self):
        args = self.args
        logger = self.logger
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)

# R41T
class GPNER41ACE05Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER41ACE05Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            if args.model_type == 'roberta':
                token_type_ids = None
#                 print('roberta模型trainer-predict构造token_type_ids')
            else:
                token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
#                 print('cccccccc')
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
#             print('len(entities):', len(entities))
            for entity_type, sh, st, prob in entities:
                temp_entity = text[new_span[sh][0]:new_span[st][-1] + 1]
                if temp_entity[0] == ' ':
                    temp_entity = temp_entity[1:]
                entity_list.append({'entity': temp_entity, 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        if args.do_train:
            output_dir = os.path.join(self.output_dir, 'entity_list.jsonl')
        else:
            output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                del data['spo_list']
                f.write(data)
                
    def predict(self, output_dir='', epoch=None):
        args = self.args
        logger = self.logger
        self.predict_entity()
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        f1, p, r = cal_ace_prf(sp20_data, mode='data')
        if output_dir == '':
            output_dir = os.path.join(args.result_output_dir, 'ace05_test.jsonl')
        else:
            output_dir = os.path.join(output_dir, f'ace05_test-epoch{epoch}-{f1}.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
        return f1, p, r
                
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity('dev')
#         self.predict_sp2o()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'ace05_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predicate2type(self, predicate):
        arr = predicate.split('-')
        assert len(arr) in [3, 4]
        if len(arr) == 4:
            sub_type, p1, p2, obj_type = arr
            p = p1+'-'+p2
        elif len(arr) == 3:
            sub_type, p, obj_type = arr
        return sub_type, p, obj_type
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if op2s_result == None:
            if args.do_train:
                read_file_name = os.path.join(self.output_dir, 'entity_list.jsonl')
            else:
                read_file_name = os.path.join(args.result_output_dir, 'entity_list.jsonl')
            with jsonlines.open(read_file_name, mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if op2s_result == None:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicarte = entity_dic['predicate']
                    sub_type, predicate, obj_type = self.predicate2type(in_predicate)
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': sub_type,
                                                'object': post_entity, 'object_type': obj_type, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicarte:
                            dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': sub_type,
                                                'object': pre_entity, 'object_type': obj_type, 'prob': prob})
            result.append(dic)
        return result

# R42T
class GPNER42ACE05Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER42ACE05Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
#             loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='mean') +\
#                       F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='mean')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
#             loss = loss + loss_kl / 4 * self.args.rdrop_alpha
            
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
            
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
#         wandb.log({"loss": loss.item()})
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
    def _get_predict_entity_list(self, test_samples, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            text = data['text']
            if isPbar:
                pbar(step)
            model.eval()
#             print(text)
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            if args.model_type == 'roberta':
                token_type_ids = None
            else:
                token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))
            entity_list = []
            for entity_type, sh, st, prob in entities:
                temp_entity = text[new_span[sh][0]:new_span[st][-1] + 1]
                if temp_entity[0] == ' ':
                    temp_entity = temp_entity[1:]
                entity_list.append({'entity': temp_entity, 'predicate':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode =='dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        if args.do_train:
            output_dir = os.path.join(self.output_dir, 'entity_list.jsonl')
        else:
            output_dir = os.path.join(args.result_output_dir, 'entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
    
    def predicate2type(self, predicate):
        arr = predicate.split('-')
        assert len(arr) in [3, 4]
        if len(arr) == 4:
            sub_type, p1, p2, obj_type = arr
            p = p1+'-'+p2
        elif len(arr) == 3:
            sub_type, p, obj_type = arr
        return sub_type, p, obj_type
        
    def predict_xp2x(self, is_sp2o=True, sp2o_result=None):
        args = self.args
        logger = self.logger
        samples = []
        if sp2o_result == None:
            if args.do_train:
                read_file_name = os.path.join(self.output_dir, 'entity_list.jsonl')
            else:
                read_file_name = os.path.join(args.result_output_dir, 'entity_list.jsonl')
            with jsonlines.open(read_file_name, mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in sp2o_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['object']['@value'], 'entity_type': spo['object_type']['@value'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        for step, sample in enumerate(samples):
            pbar(step)
            text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                predicate = entity_dic['predicate']
                if sp2o_result == None:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                else:
                    test_samples.append({'text': self.data_processor.add_prefix(text, entity),
                                         'entity': entity, 'predicate': predicate})
                    

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                in_predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    out_predicate = entity_dic['predicate']
                    sub_type, predicate, obj_type = self.predicate2type(in_predicate)
                    prob = entity_dic['prob']
                    if is_sp2o:
                        if not args.do_predicate_verify or in_predicate == out_predicate:
                            dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': sub_type,
                                            'object': post_entity, 'object_type': obj_type, 'prob': prob})
                    else:
                        if not args.do_predicate_verify or in_predicate == out_predicate:
                            dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': sub_type,
                                            'object': pre_entity, 'object_type': obj_type, 'prob': prob})
            result.append(dic)
        return result    
    
    def predict(self, output_dir='', epoch=None):
        args = self.args
        logger = self.logger
        self.predict_entity()
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        f1, p, r = cal_ace_prf(op2s_data, mode='data')
        if output_dir == '':
            output_dir = os.path.join(args.result_output_dir, 'ace05_test.jsonl')
        else:
            output_dir = os.path.join(output_dir, f'ace05_test-epoch{epoch}-{f1}.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
        return f1, p, r
                
    def predict_dev(self):
        args = self.args
        logger = self.logger
        self.predict_entity('dev')
#         self.predict_op2s()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'ace05_dev.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)

# GPFilter78T
class GPFilter78ACE05Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPFilter78ACE05Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )

    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_labels, batch_tail_labels = item
        logits1, logits2 = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)

        loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits1, mask_zero=True)
        loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits2, mask_zero=True)
        loss = sum([loss1, loss2]) / 2
        
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits1[::2],dim=-1), F.softmax(logits1[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits1[1::2],dim=-1), F.softmax(logits1[::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits2[::2],dim=-1), F.softmax(logits2[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits2[1::2],dim=-1), F.softmax(logits2[::2],dim=-1), reduction='sum')
            # ’/ 4 * self.args.rdrop_alpha‘三是公式里带的, '/ 2'是为了头尾求平均
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits1.shape[0] / 2
        
        loss.backward()

        p1, r1, f11 = self.cal_prf1(logits1, batch_head_labels)
        p2, r2, f12 = self.cal_prf1(logits2, batch_tail_labels)
        p = (p1 + p2) / 2 
        r = (r1 + r2) / 2
        f1 = (f11 + f12) / 2
        return loss.detach(), p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1
    
                
    def predict_filter(self, read_dir=None, mode='test', output_dir='', epoch=None):
        args = self.args
        logger = self.logger
        model = self.model
        data_processor = self.data_processor
        schema = data_processor.schema
        tokenizer = self.tokenizer
        device = args.device
        
        id2predicate = data_processor.id2predicate
        predicate2id = data_processor.predicate2id
        model.to(device)
        model.eval()
        
        if not read_dir:
            read_dir = os.path.join('./result_output', 'merge_ace', 'ace_test.jsonl')
        print(f'load test data from {read_dir}')
        
        with jsonlines.open(read_dir, mode='r') as r:
            test_samples = [line for line in r]
        num_examples = len(test_samples)
        
        logger.info("***** Running prediction filter *****")
        logger.info("Num samples %d", num_examples)
        
           
        filter_data = []
        logger.info(f"***** write predict file to {output_dir} *****")
        
        pbar = ProgressBar(n_total=num_examples, desc='Filtering')
        for step, data in enumerate(test_samples):
            pbar(step)
            text = data['text']
            token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=args.max_length)["offset_mapping"]
            new_span, entities = [], []
            for i in token2char_span_mapping:
                if i[0] == i[1]:
                    new_span.append([])
                else:
                    if i[0] + 1 == i[1]:
                        new_span.append([i[0]])
                    else:
                        new_span.append([i[0], i[-1] - 1])
            threshold = 0.0
            encoder_txt = tokenizer.encode_plus(text, max_length=args.max_length)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            if args.model_type == 'roberta':
                token_type_ids = None
            else:
                token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            scores = model(input_ids, attention_mask, token_type_ids)
            outputs = [o[0].data.cpu().numpy() for o in scores]

            dic = {'text': text, 'spo_list': []}

            for spo in data['spo_list']:
                sub = spo['subject']
                obj = spo['object']
                relation_key = spo['subject_type']+'-'+spo['predicate']+'-'+spo['object_type']
                if relation_key not in schema:
                    continue
                p = predicate2id[relation_key]
                input_ids = encoder_txt["input_ids"]
                if args.do_search_all:
                    sub_tokens = self.tokenizer.encode(sub, add_special_tokens=False)
                    obj_tokens = self.tokenizer.encode(obj, add_special_tokens=False)
                    shs = self.data_processor.search_all(sub_tokens, input_ids)
                    ohs = self.data_processor.search_all(obj_tokens, input_ids)
                    if shs == []:
                        sub_tokens = self.tokenizer.encode(' '+sub, add_special_tokens=False)
                        shs = self.data_processor.search_all(sub_tokens, input_ids)
                    if ohs == []:
                        obj_tokens = self.tokenizer.encode(' '+obj, add_special_tokens=False)
                        ohs = self.data_processor.search_all(obj_tokens, input_ids)
                    for sh in shs:
                        for oh in ohs:
#                                 print(sh,oh)
                            st = sh+len(sub_tokens)-1
                            ot = oh+len(obj_tokens)-1
                            if ((outputs[0][p, sh, oh] > args.filter_head_threshold and outputs[1][p, st, ot] > args.filter_tail_threshold)) or (args.do_is_inter and spo['is_inter'] and outputs[0][p, sh, oh] > args.inter_filter_head_threshold and outputs[1][p, st, ot] > args.inter_filter_tail_threshold):
                                dic['spo_list'].append(spo)

                else:
                    s = tokenizer.encode(sub, add_special_tokens=False)
                    o = tokenizer.encode(obj, add_special_tokens=False)
                    sh = data_processor.search(s, input_ids)
                    oh = data_processor.search(o, input_ids)
                    if sh == -1:
                        s = tokenizer.encode(' '+sub, add_special_tokens=False)
                        sh = data_processor.search(s, input_ids)
                    if oh == -1:
                        o = tokenizer.encode(' '+obj, add_special_tokens=False)
                        oh = data_processor.search(o, input_ids) 

                    if sh != -1 and oh != -1:
                        st = sh + len(s) - 1
                        ot = oh + len(o) - 1
                        if (outputs[0][p, sh, oh] > args.filter_head_threshold and outputs[1][p, st, ot] > args.filter_tail_threshold) or (args.do_is_inter and spo['is_inter'] and outputs[0][p, sh, oh] > args.inter_filter_head_threshold and outputs[1][p, st, ot] > args.inter_filter_tail_threshold):
                            dic['spo_list'].append(spo)

                # 去重
                filter_set = set(ACESPO(spo,5) for spo in dic['spo_list'])
                dic['spo_list'] = []
                for spo in filter_set:
                    dic['spo_list'].append(spo.spo)
#             f.write(dic)
            filter_data.append(dic)
            
        f1, p, r = cal_ace_prf(filter_data, mode='data')
        if output_dir == '':
            output_dir = os.path.join('./result_output', 'filter78ace05')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_dir = output_dir + f'/ace05_{mode}.jsonl'
        else:
            output_dir = os.path.join(self.output_dir, f'ace_test-epoch{epoch}-{f1}.jsonl')
        with jsonlines.open(output_dir, mode='w') as f: 
            for dic in filter_data:
                f.write(dic)
        return f1, p, r
        
# R3T
class GPNER75Trainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(GPNER75Trainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
        )
        
    def training_step(self, model, item):
        model.train()
        device = self.args.device
        item = [i.to(device) for i in item]
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_positions, batch_tail_positions = item
        logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_head_positions, batch_tail_positions)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
        if self.args.do_rdrop:
            loss_kl = F.kl_div(F.log_softmax(logits[::2],dim=-1), F.softmax(logits[1::2],dim=-1), reduction='sum') +\
                      F.kl_div(F.log_softmax(logits[1::2],dim=-1), F.softmax(logits[::2],dim=-1), reduction='sum')
            
#             loss_kl = F.kl_div(logits[::2].softmax(dim=-1).log(), logits[1::2].softmax(dim=-1), reduction='sum') + \
#                   F.kl_div(logits[1::2].softmax(dim=-1).log(), logits[::2].softmax(dim=-1), reduction='sum')
#             print('\n', loss_kl.item() / logits.shape[0], loss.item(), logits.shape[0])
            loss = loss + loss_kl / 4 * self.args.rdrop_alpha / logits.shape[0]
        loss.backward()

        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return loss.detach(), p, r, f1
    
    def evaluate(self, model):
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        device = self.args.device
        for step, item in enumerate(eval_dataloader):
            pbar(step)
            model.eval()
            text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = item
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels = \
                    batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device)
            logits = model(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss = self.sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)
            
        p, r, f1 = self.cal_prf1(logits, batch_entity_labels)
        return p, r, f1
    
    def cal_prf1(self, y_pred, labels):
        batch_size = labels.shape[0]
        ent_type_size = labels.shape[1]
        ent_num = labels.shape[2]
        h = torch.arange(batch_size).repeat_interleave(ent_type_size * ent_num,-1).reshape(batch_size, ent_type_size,-1)
        i = torch.arange(ent_type_size).repeat_interleave(ent_num,-1).reshape(ent_type_size,-1).repeat(batch_size,1,1)
        j = labels[...,0]
        k = labels[...,1]
        y_true = torch.zeros_like(y_pred)
        y_true[h,i,j,k] = 1
        
        y_pred = torch.greater(y_pred, 0)
        p = torch.sum(y_true * y_pred).item() / torch.sum(y_pred).item() if torch.sum(y_pred).item() != 0 else 0
        r = torch.sum(y_true * y_pred).item() / torch.sum(y_true).item() if torch.sum(y_true).item() != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return p, r, f1    
    
    def _get_predict_entity_list(self, test_samples, mode=None, isPbar=True, threshold=0.0, xp2x=False):
        args = self.args
        model = self.model
        device = args.device
        id2class = self.data_processor.id2class
        num_examples = len(test_samples)
        model.to(device)
        if isPbar:
            pbar = ProgressBar(n_total=num_examples, desc='Predicting')
        predict_data = []
        for step, data in enumerate(test_samples):
            if xp2x:
                text = data['text']
            else:
                text = self.data_processor.add_entity_prefix(data['text'], mode=mode)
            if isPbar:
                pbar(step)
            model.eval()
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
#             threshold = 0.0
            encoder_txt = self.tokenizer.encode_plus(text, max_length=args.max_length, truncation=True)
            input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
            token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
            head_position, tail_position = [0], [0]
            if xp2x:
                prefix_entity = data['entity']
#                 print(prefix_entity)
                sub = self.tokenizer.encode(prefix_entity, add_special_tokens=False)
                sh = self.data_processor.search(sub, encoder_txt["input_ids"])
                if sh != -1:
                    head_position = [sh]
                    tail_position = [sh+len(sub)-1]
#                     print(head_position, tail_position)
            head_position = torch.tensor(head_position).to(device)
            tail_position = torch.tensor(tail_position).to(device)
            score = model(input_ids, attention_mask, token_type_ids, head_position, tail_position)
            outputs = score[0].data.cpu().numpy()
            entities = set()
            outputs[:, [0, -1]] -= np.inf
            outputs[:, :, [0, -1]] -= np.inf
            for entity_type, h, t in zip(*np.where(outputs > threshold)):
                entities.add((entity_type ,h, t, round(torch.tensor(outputs)[entity_type,h,t].item(), 3)))

            entity_list = []
            for entity_type, sh, st, prob in entities:
                entity_list.append({'entity': text[new_span[sh][0]:new_span[st][-1] + 1], 'entity_type':id2class[entity_type], 'prob': prob})
            data['entity_list'] = entity_list
            predict_data.append(data)
        return predict_data
    
    def predict_sub_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running subject prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'sub_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'sub', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_obj_entity(self, mode='test'):
        args = self.args
        logger = self.logger
        if mode == 'test':
            test_samples = self.data_processor.get_test_sample()
        elif mode == 'dev':
            test_samples = self.data_processor.get_dev_sample()
        elif mode =='train':
            test_samples = self.data_processor.get_train_text_sample()
        num_examples = len(test_samples)
        logger.info("***** Running object prediction *****")
        logger.info("Num samples %d", num_examples)
        output_dir = os.path.join(args.result_output_dir, 'obj_entity_list.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            # 第0类数据的预测结果
            predict_data0 = self._get_predict_entity_list(test_samples, 'obj', threshold=args.entity_threshold)
            for data in predict_data0:
                f.write(data)
                
    def predict_sp2o(self):
        args = self.args
        logger = self.logger
        self.predict_sub_entity()
        sp20_data = self.predict_xp2x(is_sp2o=True)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in sp20_data:
                f.write(dic)
                
    def predict_op2s(self):
        args = self.args
        logger = self.logger
        self.predict_obj_entity()
        op2s_data = self.predict_xp2x(is_sp2o=False)
        output_dir = os.path.join(args.result_output_dir, 'CMeIE-V2_test.jsonl')
        logger.info(f"***** write predict file to {output_dir} *****")
        with jsonlines.open(output_dir, mode='w') as f:
            for dic in op2s_data:
                f.write(dic)
    
                
    def predict_xp2x(self, is_sp2o=True, op2s_result=None):
        args = self.args
        logger = self.logger
        samples = []
        entity_file_name = 'sub_entity_list.jsonl' if is_sp2o else 'obj_entity_list.jsonl'
        if op2s_result == None:
            with jsonlines.open(os.path.join(args.result_output_dir, entity_file_name), mode='r') as lines:
                for line in lines:
                    samples.append(line)
        else:
            for data in op2s_result:
                sample = {'text': data['text'], 'entity_list': []}
                for spo in data['spo_list']:
                    sample['entity_list'].append({'entity': spo['subject'], 'entity_type': spo['subject_type'], 'predicate': spo['predicate']})
                samples.append(sample)
        num_examples = len(samples)
        task = 'sp2o' if is_sp2o else 'op2s'
        logger.info(f"***** Running {task} prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=num_examples, desc=f'Predicting {task}')
                
        result = []
        predicate_dic = self.data_processor.subject_predicate_dic if is_sp2o else self.data_processor.object_predicate_dic
        for step, sample in enumerate(samples):
            pbar(step)
            if self.args.is_reina:
                text = sample['text'].split('[SEP]')[-1]
            else:
                text = sample['text']
            test_samples = []
            for entity_dic in sample['entity_list']:
                entity = entity_dic['entity']
                entity_type = entity_dic['entity_type']
                if op2s_result == None:
                    for predicate in predicate_dic[entity_type]:
                        prefix_text = self.data_processor.add_xp2x_prefix(text, entity, predicate, entity_type)
                        test_samples.append({'text': prefix_text if args.add_prefix else text,
                                             'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
#                         print('sp2o:', self.data_processor.add_reina_prefix_12(prefix_text, line['bm25_list'], entity, predicate))
                else:
                    predicate = entity_dic['predicate']
                    test_samples.append({'text': self.data_processor.add_xp2x_prefix(text, entity, predicate, entity_type) if args.add_prefix else text,
                                         'entity': entity, 'entity_type': entity_type, 'predicate': predicate})
                
#                 for predicate in predicate_dic[entity_type]:
#                     test_samples.append({'text': self.data_processor.add_prefix(text, entity, predicate),
#                                          'entity': entity, 'entity_type': entity_type, 'predicate': predicate})

            predict_data1 = self._get_predict_entity_list(test_samples, isPbar=False, threshold=args.xp2x_threshold, xp2x=True) 

            # 最终预测的结果
            dic = {'text': text, 'spo_list': []}
            for data in predict_data1:
                pre_entity = data['entity']
                pre_entity_type = data['entity_type']
                predicate = data['predicate']
                for entity_dic in data['entity_list']:
                    post_entity = entity_dic['entity']
                    post_entity_type = entity_dic['entity_type']
                    prob = entity_dic['prob']
                    if is_sp2o:
                        dic['spo_list'].append({'predicate': predicate, 'subject': pre_entity, 'subject_type': pre_entity_type,
                                            'object': {'@value': post_entity} , 'object_type': {'@value': post_entity_type}, 'prob': prob})
                    else:
                        dic['spo_list'].append({'predicate': predicate, 'subject': post_entity, 'subject_type': post_entity_type,
                                            'object': {'@value': pre_entity} , 'object_type': {'@value': pre_entity_type}, 'prob': prob})
            result.append(dic)
        return result