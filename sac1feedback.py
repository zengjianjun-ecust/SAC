#!/usr/bin/env python
# coding: utf-8

"""
cp from nas/ws_ft/chuang_同义句数据增强/测试预测vav0.ipynb
"""
# 131特供设置
import json
import datetime,time
import os
import shutil
import sys



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    sys.path.append('./codes')

import sys
import torch
print("Python 版本:", sys.version)
print("PyTorch 版本:", torch.__version__)
print("CUDA Toolkit 版本:", torch.version.cuda)


import warnings
warnings.filterwarnings('ignore')


import os
import argparse
import torch
import shutil

from models import GPNER41Model
from trainer import GPNER41Trainer
from data import GPNER41Dataset, GPNER41DataProcessor
from utils import init_logger, seed_everything, get_devices, get_time

import torch.utils.data as Data
from torch import nn
from d2l import torch as d2l
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModelForMaskedLM
MODEL_CLASS = {
    'bert': (BertTokenizerFast, BertModel),
    'roberta': (AutoTokenizer, AutoModelForMaskedLM),
    'mcbert': (AutoTokenizer, AutoModelForMaskedLM),
    'roformer': (AutoTokenizer, AutoModelForMaskedLM)
}


def get_args():
    parser = argparse.ArgumentParser()
    
    # 方法名：baseline required=True
    parser.add_argument("--method_name", default='gpner41', type=str,
                        help="The name of method.")
    
    # 数据集存放位置：./CMeIE required=True
    parser.add_argument("--data_dir", default='./data_dir', type=str,
                        help="The task data directory.")
    
    parser.add_argument("--train_file", default='CMeIE-V2_train.jsonl', type=str,
                        help="The train file.")
    
    parser.add_argument("--dev_file", default='CMeIE-V2_dev.jsonl', type=str,
                        help="The dev file.")
    
    parser.add_argument("--test_file", default='CMeIE-V2_test.jsonl', type=str,
                        help="The test file.")
    
    # 增强数据集存放位置：./enhanced_data required=True
    parser.add_argument("--enhanced_data_dir", default='./enhanced_data', type=str,
                        help="The path of enhanced_data produced by doing dual eval with test set.")
    
    # 是否做数据增强
    parser.add_argument("-do_enhance", default=False, type=bool,
                    help="Whether to do data enhance.")
    
    # 预训练模型存放位置: ../../pretrained_model required=True
    parser.add_argument("--model_dir", default='./pretrained_model', type=str,
                        help="The directory of pretrained models")
    
    # 模型类型: bert required=True
    parser.add_argument("--model_type", default='bert', type=str, 
                        help="The type of selected pretrained models.")
    
    # 预训练模型: bert-base-chinese required=True
    parser.add_argument("--pretrained_model_name", default='RoBERTa_zh_Large_PyTorch', type=str,
                        help="The path or name of selected pretrained models.")
    
    # 微调模型: er required=True
    parser.add_argument("--finetuned_model_name", default='gpner41', type=str,
                        help="The name of finetuned model")
    
    # 微调模型参数存放位置：./checkpoint required=True
    parser.add_argument("--output_dir", default='./checkpoint', type=str,
                        help="The path of result data and models to be saved.")
    
    # 是否训练：True
    parser.add_argument("--do_train", default='True', type=bool,
                        help="Whether to run training.")
    
    # 是否预测：False required=True
    parser.add_argument("--do_predict", default='True', type=bool,
                        help="Whether to run the models in inference mode on the test set.")
    
    # 预测时加载的模型版本，如果做预测，该参数是必需的
    parser.add_argument("--model_version", default='', type=str,
                        help="model's version when do predict")
    
    # 提交结果保存目录：./result_output required=True
    parser.add_argument("--result_output_dir", default='./result_output', type=str,
                        help="the directory of commit result to be saved")
    
    # 设备：-1：CPU， i：cuda:i(i>0), i可以取多个，以逗号分隔 required=True
    parser.add_argument("--devices", default='0', type=str,
                        help="the directory of commit result to be saved")
    
    parser.add_argument("--loss_show_rate", default=200, type=int,
                        help="liminate loss to [0,1] where show on the train graph")
    
    # models param
    

    
    # 序列最大长度：128
    parser.add_argument("--max_length", default=256, type=int,
                        help="the max length of sentence.")
    
    # 训练batch_size：32
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    
    # 评估batch_size：64
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    
    # 学习率：3e-5
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    
    # 权重衰退：取默认值
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    
    # 极小值：取默认值
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    
    # epochs：7
    parser.add_argument("--epochs", default=7, type=int,
                        help="Total number of training epochs to perform.")
    
    # 线性学习率比例：0.1
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for, "
                             "E.g., 0.1 = 10% of training.")
    
    # earlystop_patience：100 （earlystop_patience step 没有超过最高精度则停止训练）
    parser.add_argument("--earlystop_patience", default=100, type=int,
                        help="The patience of early stop")
    
    # 多少step后打印一次：200
    parser.add_argument('--logging_steps', type=int, default=200,
                        help="Log every X updates steps.")
    
    
    # 随机数种子：2021
    parser.add_argument('--seed', type=int, default=2021,
                        help="random seed for initialization")
    
    # 训练时保存 save_metric 最大存取模型 required=True
    parser.add_argument("--save_metric", default='r', type=str,
                        help="the metric determine which model to save.")
    
    # 是否做rdrop（变相的数据增强）
    parser.add_argument('--do_rdrop', type=bool, default=False,
                        help="whether to do r-drop")
    
    # rdrop 中的参数，alpha越大则loss越偏向kl散度
    parser.add_argument('--rdrop_alpha', type=int, default=4,
                        help="hyper-parameter in rdrop")
    
    # 正则化手段，dropout
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="dropout rate")
    
    # 对偶验证时考虑的门槛，若baseline生成的spo的置信度低于门槛，且未通过对偶验证，才会被删掉
    # 当threshold大于1时则不起作用
    parser.add_argument('--dual_threshold', type=float, default=0.99,
                    help="")
    
    # 预测主体时的阈值，测试才用到的参数
    parser.add_argument('--entity_threshold', type=float, default=0.0,
                        help="the threshold of predicating subject")
    
    # 预测sp2o时的阈值，测试才用到的参数
    parser.add_argument('--xp2x_threshold', type=float, default=0.0,
                        help="the threshold of predicating sp2o")
    
    # 是否使用CMeEE数据增强实体抽取
    parser.add_argument('--do_enhance_CMeEE', type=bool, default=False,
                        help="whether to use CMeEE data to enhance entity extraction")
    
    
    # py版本
    # args = parser.parse_args()
    # jupyter版本
    args = parser.parse_known_args()[0]
    # vscode版本
#     args = parser.parse_args(args=[])

    # 测试用
    args.do_train = False
    args.do_predict = True
    args.do_predict_dev = False
    args.do_predict_train = False
    args.do_entity_marker = False
    args.load_model_name = 'pytorch_model.pt'   # ！

    # args.data_dir = './CMeIE-V2'
    args.train_file = 'CMeIE-V2_train.jsonl'
    args.dev_file = 'CMeIE-V2_dev.jsonl'
    args.test_file = 'CMeIE-V2_test.jsonl' # ！
    args.finetuned_model_name = 'gpner41'
    args.do_rdrop = False
    args.do_predicate_verify = False
    args.prefix_merge_mode = 'atten2'
    args.prefix_mode = 'entity'
    args.atten_dim = 2048
    args.inner_dim = 64
    print('args.do_predicate_verify:', args.do_predicate_verify)
    print('args.prefix_merge_mode:', args.prefix_merge_mode)
    print('args.prefix_mode:', args.prefix_mode)
    print('args.atten_dim:', args.atten_dim)
    print('args.inner_dim:', args.inner_dim)
    
    args.model_type = 'bert'
    args.atten_mode = 'sigmoid'
    args.entity_threshold = 0   # ！
    args.xp2x_threshold = 0 # ！
    print('entity:', args.entity_threshold)
    print('xp2x:', args.xp2x_threshold)
    
    
    
    args.save_metric = 'step'
    args.predict_threshold = 0.4
    args.devices = '0'
    args.dual_threshold = 0.962

    # args.model_version = '11-23-06-23'  # text_base
    args.model_version = '09-14-13-38'  # mini_200
    args.pretrained_model_name = 'RoBERTa_zh_Large_PyTorch'
    args.epochs = 20
    args.train_batch_size = 16
    args.eval_batch_size = 8
    args.logging_steps = 2000
    args.learning_rate = 4e-5
    args.max_grad_norm = 1

    args.devices = get_devices(args.devices.split(','))
    args.device = args.devices[0]
    args.distributed = True if len(args.devices) > 1  else False 
    seed_everything(args.seed)
    args.time = get_time(fmt='%m-%d-%H-%M')
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.method_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.pretrained_model_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.finetuned_model_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    print(args.output_dir)    
    args.result_output_dir = os.path.join(args.result_output_dir, args.finetuned_model_name) 
    if not os.path.exists(args.result_output_dir):
        os.mkdir(args.result_output_dir)
        
    if not os.path.exists(args.enhanced_data_dir):
        shutil.copytree(args.data_dir, args.enhanced_data_dir)
    
    if args.do_enhance == True:
        args.data_dir = args.enhanced_data_dir
    if args.do_train and args.do_predict:
        args.model_version = args.time
    if args.do_predict == True and args.model_version == '':
        raise Exception('做预测的话必须提供加载的模型版本')    
    return args


data_dir=


result_output_dir=data_dir+output # 输出路径


model_version # 模型设置

class singleton_main():
    """
    make main le
    """
    def init_model(__self__, model_version):
        

        args = get_args()
        args.model_version = model_version
        logger = init_logger(os.path.join(args.output_dir, 'log.txt'))
        tokenizer_class, model_class = MODEL_CLASS[args.model_type]
        additional_special_tokens = [f'[unused{i+1}]' for i in range(99)]

        load_dir = os.path.join(args.output_dir, args.model_version)
        logger.info(f'load tokenizer from {load_dir}')
        tokenizer = tokenizer_class.from_pretrained(load_dir)
        tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
        model = GPNER41Model(model_class, args)
        
        __self__.model_version = model_version
        __self__.model = model
        __self__.tokenizer = tokenizer
        
    def predict(__self__, data_dir):
        
        args = get_args()
        
        args.model_version = __self__.model_version
        model = __self__.model
        tokenizer = __self__.tokenizer
        
        args.output_dir = data_dir + '/output/'
        
        data_processor = GPNER41DataProcessor(args)

        logger = init_logger(os.path.join(args.output_dir, 'log.txt'))
        trainer = GPNER41Trainer(args=args, model=model, data_processor=data_processor,
                            tokenizer=tokenizer, logger=logger)
        trainer.load_checkpoint(model_name=args.load_model_name)
        trainer.predict()
        

def main():
    
    args = get_args()
    logger = init_logger(os.path.join(args.output_dir, 'log.txt'))
    tokenizer_class, model_class = MODEL_CLASS[args.model_type]
    additional_special_tokens = [f'[unused{i+1}]' for i in range(99)]
        
    if args.do_predict:
        load_dir = os.path.join(args.output_dir, args.model_version)
        logger.info(f'load tokenizer from {load_dir}')
        tokenizer = tokenizer_class.from_pretrained(load_dir)
        tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
        
        data_processor = GPNER41DataProcessor(args)
        model = GPNER41Model(model_class, args)
        
        trainer = GPNER41Trainer(args=args, model=model, data_processor=data_processor,
                            tokenizer=tokenizer, logger=logger)
        trainer.load_checkpoint(model_name=args.load_model_name)
        trainer.predict()
        
if __name__ == '__main__':
    main()