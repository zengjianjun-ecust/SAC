import torch
import json
from datetime import datetime
from torch import nn
import logging
import random
import os
import sys
import time
import numpy as np
import unicodedata
import pynvml
import jsonlines

def get_time(fmt='%Y-%m-%d %H:%M:%S'):
    """
    获取当前时间
    """
    ts = time.time()
    ta = time.localtime(ts)
    t = time.strftime(fmt, ta)
    return t

def save_args(args, path):
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        f.writelines('------------------- start -------------------\n')
        for arg, value in args.__dict__.items():
            f.writelines(f'{arg}: {str(value)}\n')
        f.writelines(f"Python 版本:{sys.version}\n")
        f.writelines(f"PyTorch 版本:{torch.__version__}\n")
        f.writelines(f"CUDA Toolkit 版本:{torch.version.cuda}\n")
        f.writelines(f"当前环境的Python解释器路径：{sys.executable}\n")
        f.writelines('------------------- end -------------------')
        
def save_cur_epoch(cur_epoch, path):
    with open(os.path.join(path, 'cur_epoch.txt'), 'w') as f:
        f.writelines(str(cur_epoch))
        
def get_cur_epoch(path):
    with open(os.path.join(path, 'cur_epoch.txt'), 'r') as f:
        a = int(f.readlines()[0])
    return a
        
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    print(log_file)
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
    return logger


def get_devices(devices_id):
    devices = []
    for i in devices_id:
        if i == '-1':
            devices.append(torch.device('cpu'))
        else:
            devices.append(torch.device(f'cuda:{i}'))
#     return [torch.device(f'cuda:{i}') for i in devices_id]
    return devices

class ProgressBar(object):
    """
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='Training')
        >>> step = 2
        >>> pbar(step=step)
    """
    def __init__(self, n_total,width=30,desc = 'Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current< self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')
    
class TokenRematch:
    def __init__(self):
        self._do_lower_case = True

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
#         print(tokens)
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
#                 print(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                # offset的作用是避免文本中有重复字符
                offset = end

        return token_mapping
    
class SPO():
    def __init__(self, spo):
        self.spo = spo
        
    def __str__(self):
        return self.spo.__str__()
        
    def __eq__(self, other):
        return self.spo['predicate'] == other.spo['predicate'] and \
               self.spo['subject'] == other.spo['subject'] and self.spo['subject_type'] == other.spo['subject_type'] and \
               self.spo['object']["@value"] == other.spo['object']["@value"] and self.spo['object_type']["@value"] == other.spo['object_type']["@value"]
    
    def __hash__(self):
        return hash(self.spo['predicate'] + self.spo['subject'] + self.spo['subject_type'] + self.spo['object']["@value"] + self.spo['object_type']["@value"])
    
class SPO_No_Type():
    def __init__(self, spo):
        self.spo = spo
        
    def __str__(self):
        return self.spo.__str__()
        
    def __eq__(self, other):
        return self.spo['predicate'] == other.spo['predicate'] and \
               self.spo['subject'] == other.spo['subject'] and \
               self.spo['object']["@value"] == other.spo['object']["@value"]
    
    def __hash__(self):
        return hash(self.spo['predicate'] + self.spo['subject'] + self.spo['object']["@value"])

class SPO_No_Type_Strip():
    def __init__(self, spo):
        self.spo = spo
        
    def __str__(self):
        return self.spo.__str__()
        
    def __eq__(self, other):
        return self.spo['predicate'].strip() == other.spo['predicate'].strip() and \
               self.spo['subject'].strip() == other.spo['subject'].strip() and \
               self.spo['object']["@value"].strip() == other.spo['object']["@value"].strip()
    
    def __hash__(self):
        return hash(self.spo['predicate'].strip() + self.spo['subject'].strip() + self.spo['object']["@value"].strip())
    
class ACESPO():
    def __init__(self, spo, mode=3):
        self.spo = spo
        self.mode = mode
        
    def __str__(self):
        return self.spo.__str__()
        
    def __eq__(self, other):
        if self.mode == 5:
            return self.spo['predicate'] == other.spo['predicate'] and \
               self.spo['subject'] == other.spo['subject'] and self.spo['subject_type'] == other.spo['subject_type'] and\
               self.spo['object'] == other.spo['object'] and self.spo['object_type'] == other.spo['object_type']
        elif self.mode == 3:
            return self.spo['predicate'] == other.spo['predicate'] and \
               self.spo['subject'] == other.spo['subject'] and \
               self.spo['object'] == other.spo['object']
        elif self.mode == 2:
            return self.spo['subject'] == other.spo['subject'] and \
                   self.spo['object'] == other.spo['object']
    
    def __hash__(self):
        if self.mode == 5:
            return hash(self.spo['predicate'] + self.spo['subject'] + self.spo['subject_type'] + self.spo['object'] + self.spo['object_type'])
        elif self.mode == 3:
            return hash(self.spo['predicate'] + self.spo['subject'] + self.spo['object'])
        elif self.mode == 2:
            return hash(self.spo['subject'] + self.spo['object'])
        
def zhank(need_memory=20, gpu_num=4, skip_list=[]):
    pynvml.nvmlInit()
    flag=1
    print('*******占卡中*******')
    
    while flag:
        time.sleep(1)
        for i in range(gpu_num):
            if i in skip_list:
                continue
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)# 这里是GPU id
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_free = meminfo.free / 1024 /1024 /1024    # G
    #         print("第{0}块显卡剩余{1}G".format(i,round(gpu_free,1))) #显卡剩余显存大小
            if gpu_free > need_memory:
                device = f'{i}'
                flag=0
                break
    print(f'占到{device}卡')
    return device

def percent(num):
    return round(num*100,3)

def load_schema():
    with open('./CMeIE-V2/53_schemas.json', 'r', encoding='utf-8') as f:
        schema = {}
        relations = []
        for idx, item in enumerate(f):
            item = json.loads(item.rstrip())
            relations.append(item["subject_type"]+"_"+item["predicate"]+"_"+item["object_type"])
        predicate2id = {v: i for i, v in enumerate(relations)}
        id2predicate = {i: v for i, v in enumerate(relations)}
    return relations, predicate2id, id2predicate

def percent(num):
    return round(num*100,4)

def cal_ace_prf(test_file, mode='file', gold_file=''):
    # file：传文件
    # data: 传预测数据
    
    
    if mode == 'file':
        pred_data = []
        with jsonlines.open(test_file, mode='r') as rs:
            pred_data = [data for data in rs]
    elif mode == 'data':
        pred_data = test_file
            
    if gold_file == '':
        gold_file = './ACE05-DyGIE/processed_data/test.json'
    with open(gold_file, 'r', encoding='utf-8') as lines:
        gold_data = []
        for line in lines:
            gold_data.append(json.loads(line))
        pred_true_num, pred_num, true_num = 0, 0, 0
        pred_data = [i for i in pred_data]

        for i, (gold, pred) in enumerate(zip(gold_data, pred_data)):
#             pred_num += len(pred['spo_list'])
#             true_num += len(gold['spo_list'])
            pred_num += len(set(ACESPO(spo, 5) for spo in pred['spo_list']))
            true_num += len(set(ACESPO(spo, 5) for spo in gold['spo_list']))
            pred_true_num += len(set(ACESPO(spo, 5) for spo in pred['spo_list']) & set(ACESPO(spo, 5) for spo in gold['spo_list']))

        print('预测对的：', pred_true_num)
        print('预测的：', pred_num)
        print('对的：', true_num)
        if pred_num != 0:
            p = pred_true_num / pred_num
            r = pred_true_num / true_num
        else:
            p = 0
            r = 0
        if p + r != 0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0
    
    f1 = percent(f1)
    p = percent(p)
    r = percent(r)
    print('f1: ', f1)
    print('p: ', p)
    print('r: ', r)
    return f1, p, r