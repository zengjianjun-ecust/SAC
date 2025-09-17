import os
import json
import jsonlines
from collections import defaultdict
from constant import spot_labels, spot_prompt, asoc_prompt
from utils import random

class ERDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.train_path = os.path.join(root, 'CMeIE-V2_train.jsonl')
        self.dev_path = os.path.join(root, 'CMeIE-V2_dev.jsonl')
        self.test_path = os.path.join(root, 'CMeIE-V2_test.jsonl')
        self.args = args
    
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, mode='val')

    def get_test_sample(self):
        return self._pre_process(self.test_path, mode='test')

    def _pre_process(self, path, mode):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            result = {'text': [], 'spo_list': []}
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in lines:
                data = json.loads(line)
                text = data['text']

                if mode != 'test':
                    one_spo_list = []
                    for spo in data['spo_list']:
                        s = spo['subject']
                        p = spo['predicate']
                        tmp_ob_type = [v for k, v in spo['object_type'].items()]
                        tmp_ob = [v for k, v in spo['object'].items()]
                        for i in range(len(tmp_ob)):
                            # p_o 后面用不上
                            p_o = p + '|' + tmp_ob_type[i]
                            one_spo_list.append((s, p_o, tmp_ob[i]))
                else:
                    one_spo_list = None
                for i in range(iter_num):
                    result['text'].append(text)
                    result['spo_list'].append(one_spo_list)

            return result

    def search(self, sequence, pattern):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回0。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return 0

    def _extract_entity(self, start_logits, end_logits, text_start_id, text_end_id):
        threshold = self.args.predict_threshold
        # logits: seq
        start_ids = (start_logits[text_start_id:text_end_id] >= threshold).long()
        end_ids = (end_logits[text_start_id:text_end_id] >= threshold).long()

        start_end_tuple_list = []
        for i, start_id in enumerate(start_ids):
            if start_id == 0:
                continue  # 不是起点
            if end_ids[i] == 1:  # 起点和终点重合
                start_end_tuple_list.append((i, i))
                continue
            j = i + 1
            find_end_tag = False
            while j < len(end_ids):
                if start_ids[j] == 1:
                    break  # 终点前遇到新的起点，停止搜索
                if end_ids[j] == 1:
                    start_end_tuple_list.append((i, j))
                    find_end_tag = True
                    break
                else:
                    j += 1
            if not find_end_tag:  # 没找到终点->孤立点
                start_end_tuple_list.append((i, i))
        return start_end_tuple_list

    def extract_arg(self, start_logits, end_logits, text_start_id, text_end_id, text, text_mapping):
        arg_tuple = self._extract_entity(start_logits, end_logits, text_start_id, text_end_id)

        one_role_args = []
        for k in arg_tuple:
            # 感觉没有作用
            if len(text_mapping) > 3:
                # len(text_mapping) : token size
                # k0: 起点    k1: 终点
                start_split = text_mapping[k[0]]
                end_split = text_mapping[k[1]]
                if start_split != [] and end_split != []:
                    tmp = text[start_split[0]:end_split[-1] + 1]
                    one_role_args.append(tmp)
        return one_role_args

class REDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.train_path = os.path.join(root, 'CMeIE-V2_train.jsonl')
        self.dev_path = os.path.join(root, 'CMeIE-V2_dev.jsonl')
        self.test_path = os.path.join(root, 'CMeIE-V2_test.jsonl')

        self.schema_path = os.path.join(root, '53_schemas.json')
        self.pre_sub_obj = None
        self.predicate2id = None
        self.id2predicate = None
        self.s_entity_type = None
        self.o_entity_type = None
        self.args = args
        self._load_schema()

        self.num_labels = len(self.predicate2id.keys())

    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, mode='val')

    def get_test_sample(self, path):
        """ Need new test file generated from the result of ER prediction
        """
        with open(path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            samples = []
            for line in lines:
                data = json.loads(line)
                samples.append(data)
        return samples

    def _pre_process(self, path, mode):
        with open(path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            result = {'text': [], 'label': [], 'flag': []}
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in lines:
                data = json.loads(line)
                text = data['text']
                s_dict = {}  # sub : sub_type
                o_dict = {}  # obj : obj_type
                spo_dict = {}  # sub|obj : predicate|obj_type
                for spo in data['spo_list']:
                    sub = spo['subject']
                    # s_dict[spo['subject_type']] = spo['subject']
                    s_dict[spo['subject']] = spo['subject_type']
                    pre = spo['predicate']
                    p_o = pre + '|' + spo['object_type']['@value']
                    spo_dict[sub + '|' + spo['object']['@value']] = p_o
                    # o_dict[spo['object_type']['@value']] = spo['object']['@value']
                    o_dict[spo['object']['@value']] = spo['object_type']['@value']
                for sv, sk in s_dict.items():
                    for ov, ok in o_dict.items():
                        s_flag = self.s_entity_type[sk]  # '<s>, </s>'
                        o_flag = self.o_entity_type[ok]
                        s_start = self.search(text, sv)
                        s_end = s_start + len(sv)
                        text1 = text[:s_start] + s_flag[0] + sv + s_flag[1] + text[s_end:]
                        o_start = self.search(text1, ov)
                        o_end = o_start + len(ov)
                        text2 = text1[:o_start] + o_flag[0] + ov + o_flag[1] + text1[o_end:]
                        if sv + '|' + ov in spo_dict.keys():
                            labels = self.predicate2id[spo_dict[sv + '|' + ov]]
                        else:
                            labels = 0
                        for i in range(iter_num):
                            result['text'].append(text2)
                            result['label'].append(labels)
                            result['flag'].append((s_flag[0], o_flag[0]))
            return result

    def _load_schema(self, ):
        with open(self.schema_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            predicate_list = ["无关系"]
            s_entity = []
            o_entity = []
            pre_sub_obj = {}
            for line in lines:
                data = json.loads(line)
                if data['subject_type'] not in s_entity:
                    s_entity.append(data['subject_type'])
                if data['object_type'] not in o_entity:
                    o_entity.append(data['object_type'])
                predicate_list.append(data['predicate'] + '|' + data['object_type'])
                pre_sub_obj[data['predicate'] + '|' + data['object_type']] = [data['subject_type'], data['object_type']]

            s_entity_type = {}
            for i, e in enumerate(s_entity):  # 主语
                s_entity_type[e] = ('<s>', '</s>')  # unused4 unused5

            o_entity_type = {}
            for i, e in enumerate(o_entity):
                o_entity_type[e] = ('<o>', '</o>')

            predicate2id = {v: i for i, v in enumerate(predicate_list)}
            id2predicate = {i: v for i, v in enumerate(predicate_list)}

            self.pre_sub_obj = pre_sub_obj
            self.predicate2id = predicate2id
            self.id2predicate = id2predicate
            self.s_entity_type = s_entity_type
            self.o_entity_type = o_entity_type

    def search(self, sequence, pattern):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回0。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return 0

    def build_text(self, data):
        text = data['text']
        result = []
        outputs = {'text': [], 'flag': [], "spo_list": []}
        for sub in data['sub_list']:
            for obj in data['obj_list']:
                if sub == obj:
                    continue
                sub_flag = ['<s>', '</s>']
                obj_flag = ['<o>', '</o>']
                sub_start = self.search(text, sub)  # sub在text的起点
                sub_end = sub_start + len(sub)
                text2 = text[:sub_start] + sub_flag[0] + sub + sub_flag[1] + text[sub_end:]
                obj_start = self.search(text2, obj)
                obj_end = obj_start + len(obj)
                text3 = text2[:obj_start] + obj_flag[0] + obj + obj_flag[1] + text2[obj_end:]
                result.append(
                    {'text': text3, 'flag': (sub_flag[0], obj_flag[0]), 'spo_list': {'subject': sub, 'object': obj}})
                outputs['text'].append(text3)
                outputs['flag'].append((sub_flag[0], obj_flag[0]))
                outputs['spo_list'].append({'subject': sub, 'object': obj})
        return result, outputs

class GPNER2DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        print(self.train_path)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, mode='dev')

    def get_test_sample(self):
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
#         predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
#                           '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
#                           '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
#                           '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
#                           '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
#         self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
#         self.predicate2id = {v: i+1 for i, v in enumerate(predicates)}
#         relations = []
#         subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
#         with open(self.schema_path, 'r', encoding='utf-8') as f:
#             predicate2outype = {}
#             for idx, item in enumerate(f):
#                 item = json.loads(item.rstrip())
#                 subject_predicate_dic[item["subject_type"]].append(item["predicate"])
#                 object_predicate_dic[item["object_type"]].append(item["predicate"])
#                 predicate2outype[item["predicate"]] = item["object_type"]
#         predicate2outype['同义词'] == '同义词'
#         self.predicate2outype = predicate2outype   
#         self.subject_predicate_dic = subject_predicate_dic
#         self.object_predicate_dic = object_predicate_dic
        
    def _pre_process(self, path, mode):
        new_data = []
        with open(path, 'r', encoding='utf-8') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["subject"], spo["object"]["@value"]) for spo in spo_list]
                    })

#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data
    

    
class GPLinkerDataProcessor(object):
    def __init__(self, root):
        self.train_path = os.path.join(root, 'CMeIE-V2_train.jsonl')
        self.dev_path = os.path.join(root, 'CMeIE-V2_dev.jsonl')
        self.test_path = os.path.join(root, 'CMeIE-V2_test.jsonl')
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        self.num_labels = len(self.predicate2id.keys())
        
    def get_train_sample(self):
        print('加载训练数据：', self.train_path)
        return self._pre_process(self.train_path)

    def get_dev_sample(self):
        print('加载验证数据：', self.dev_path)
        with open(self.dev_path) as f:
            text_list = [json.loads(text.rstrip())["text"] for text in f.readlines()]
        return text_list

    def get_test_sample(self):
        print('加载测试数据：', self.test_path)
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip())["text"] for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            schema = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                schema[item["subject_type"]+"_"+item["predicate"]+"_"+item["object_type"]] = idx
        
        self.schema = schema  
        self.predicate2id = {v: i for i, v in enumerate(schema.keys())}
        self.id2predicate = {i: v for i, v in enumerate(schema.keys())}
    
    def _pre_process(self, path):
        new_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                new_data.append({
                    "text":line["text"],
                    "spo_list":[(spo["subject"], spo["predicate"], spo["object"]["@value"], spo["subject_type"], spo["object_type"]["@value"])
                                for spo in line["spo_list"]]
                })
        return new_data

class GPFilterDataProcessor(object):
    def __init__(self, args):
        self.args = args
        root = args.data_dir
        self.train_path = os.path.join(root, args.train_file)
        print(self.train_path)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        self.num_labels = len(self.predicate2id.keys())
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, mode='dev')

    def get_test_sample(self):
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip())["text"] for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            schema = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                schema[item["subject_type"]+"_"+item["predicate"]+"_"+item["object_type"]] = idx
        
        self.schema = schema  
        self.predicate2id = {v: i for i, v in enumerate(schema.keys())}
        self.id2predicate = {i: v for i, v in enumerate(schema.keys())}
    
    def _pre_process(self, path, mode):
        new_data = []
        with open(path, 'r', encoding='utf-8') as f:
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                line = json.loads(line)
                for _ in range(iter_num):
                    new_data.append({
                        "text":line["text"],
                        "spo_list":[(spo["subject"], spo["predicate"], spo["object"]["@value"], spo["subject_type"], spo["object_type"]["@value"])
                                    for spo in line["spo_list"]]
                    })
        return new_data
    
    def regular(self, spo):
        """
        判断spo是否符合规则
        return bool 
        """
        sub = spo['subject']
        sub_type = spo['subject_type']
        if sub_type == '疾病' and len(sub) == 1 and sub != '痔':
            return False
        return True
    
class GPFilterace05DataProcessor(object):
    def __init__(self, args):
        self.args = args
        root = args.data_dir
        self.train_path = os.path.join(root, args.train_file)
        print(self.train_path)
        self.dev_path = os.path.join(root, 'dev.json')
        self.test_path = os.path.join(root, 'test.json')
        self.schema_path = os.path.join(root, '65_schemas.json')
        self._load_schema()
        self.num_labels = len(self.predicate2id.keys())
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, mode='dev')

    def get_test_sample(self):
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip())["text"] for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        if self.args.with_type:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                schema = []
                for idx, item in enumerate(f):
                    item = json.loads(item.rstrip())
                    schema.append(item["subject_type"]+"_"+item["predicate"]+"_"+item["object_type"])
        else:
            schema = ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE']
        self.schema = schema
        self.num_predicates = len(schema)
        self.predicate2id = {v: i for i, v in enumerate(schema)}
        self.id2predicate = {i: v for i, v in enumerate(schema)}
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with open(path, 'r', encoding='utf-8') as f:
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                line = json.loads(line)
                for _ in range(iter_num):
                    new_data.append({
                        "text":line["text"],
                        "spo_list":[(spo["subject"], spo["predicate"], spo["object"], spo["subject_type"], spo["object_type"]) for spo in line["spo_list"]] if args.with_type\
                                    else [(spo["subject"], spo["predicate"], spo["object"], '', '') for spo in line["spo_list"]]
                    })
        return new_data
    

    
class GPNERDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
#         print(self.train_path)
#         print(self.dev_path)
#         print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据:', self.train_path)
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train_text数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data
    
    def get_dev_sample(self):
        print('读取dev数据:', self.dev_path)
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据:', self.test_path)
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i+1 for i, v in enumerate(predicates)}
        relations = []
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["object_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype   
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate, entity_type=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
            
    def build_data12(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate, input_entity_type)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # S+P 抽客体
                    "input_entity_type": input_entity_type,
                    "predicate": predicate,
                    "text":  self.add_reina_prefix_12(prefix_text, line['bm25_list'], input_entity, predicate) if self.args.is_reina and mode == 'train' else prefix_text, 
                    "entity_list":[] # 必须是list
                    
                } 
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for _ in range(iter_num):
                        new_data.append(data)
#             if data["entity_list"] != []:
#                 print('正样本')
#                 for _ in range(iter_num):
#                     new_data.append(data)
#             elif data['predicate'] in self.subject_predicate_dic[data['input_entity_type']]:
#                 if random.random() < args.neg_wrong_rel_same_type_rate:
#                     print('same')
#                     for _ in range(iter_num):
#                         new_data.append(data)
#             else:
#                 if random.random() < args.neg_wrong_rel_diff_type_rate:
#                     print('diff')
#                     for _ in range(iter_num):
#                         new_data.append(data)
        if args.add_wrong_entity_neg_sample:
            entity2type_dic = {}
            for spo in line['entity_list']:
                entity2type_dic[spo['entity']] = spo['entity_type']
            gold_entities = set([spo['subject'] for spo in line['spo_list']])
            pred_entities = set([spo['entity'] for spo in line['entity_list']])
            wrong_entities = list(pred_entities-gold_entities)
#             print('wrong_entities:',wrong_entities)
            for input_entity in wrong_entities:
                entity_type = entity2type_dic[input_entity]
                for predicate in entity2predicate_dic[entity_type]:
                    if random.random() < args.wrong_entity_neg_sample_rate:
                        prefix_text = self.add_prefix(text, input_entity, predicate, entity_type)
                        data = {
                            "type": data_type, # O+P 抽主体
                            "text": prefix_text, 
                            "entity_list":[], # 必须是list
                            "neg_type": 'wrong_entity'
                        }
                        for i in range(iter_num):
                            new_data.append(data)
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            if args.add_wrong_entity_neg_sample:
                with jsonlines.open('./result_output/dualgpner/entity_list_train.jsonl') as r:
                    f1 = [line for line in r]
            else:
                f = [i for i in f]
                f1 = [0 for i in f]
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line, line1 in zip(f, f1):
                if args.add_wrong_entity_neg_sample:
                    line['entity_list'] = line1['entity_list']
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_reina_prefix_0(text, line['bm25_list']) if self.args.is_reina and mode == 'train' else text,
                        "entity_list": [(spo["subject"], spo["subject_type"])
                                    for spo in spo_list]
                    })
                new_data.extend(self.build_data12(line, self.subject_predicate_dic, 1, self.args.negative_samples_rate, mode))
#                 new_data.extend(self.build_data12(line, self.subject_predicate_dic, 1, 1, mode))

#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNERACE05DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, 'train.json')
        self.dev_path = os.path.join(root, 'dev.json')
        self.test_path = os.path.join(root, 'test.json')
        self.schema_path = os.path.join(root, '65_schemas.json')
        self._load_schema()
        
    def get_train_sample(self):
        return self._pre_process(self.train_path)

    def get_dev_sample(self):
        return self._pre_process(self.dev_path)

    def get_test_sample(self):
        with jsonlines.open(self.test_path, 'r') as f:
            data_list = [line for line in f]
        return data_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
#         predicates = ['User-Owner-Inventor-Manufacturer',
#                       'Citizen-Resident-Religion-Ethnicity', 'Org-Location',
#                       'Employment', 'Founder', 'Ownership', 'Student-Alum', 'Sports-Affiliation', 'Investor-Shareholder', 'Membership',
#                       'Artifact',  'Geographical', 'Subsidiary',
#                       'Business', 'Family', 'Lasting-Personal',  
#                       'Located', 'Near']
        
        labels = ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']
        predicates = ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE']
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        self.labels = labels
        self.predicates = predicates
        self.num_predicates = len(predicates)
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        self.id2predicate = {i: v for i, v in enumerate(predicates)}
        
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def add_prefix(self, text, entity, predicate):
        return f"entity: {entity}, relation: {predicate}, {text}"
#         return f"{entity}[unused1]{predicate}[unused2]{text}"

    def build_data12(self, text, spo_list, entity2predicate_dic, data_type=1, keep_rate=1):
        args = self.args
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                temp = (spo['object'], spo['object_type']) if args.with_type else spo['object']
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append(temp)
                input_entity_types.append(spo["subject"])
            else:
                temp = (spo['subject'], spo['subject_type']) if args.with_type else spo['subject']
                positive_dic[f"{spo['object']}{spo['predicate']}"].append(temp)
                input_entity_types.append(spo['object'])
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        
        for input_entity in input_entity_types:
            predicates = self.predicates if args.with_type else entity2predicate_dic[input_entity]
            for predicate in predicates:
                # 1：S+P抽O，2：O+P抽S
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # O+P 抽主体
                    "text": self.add_prefix(text, input_entity, predicate), # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                    "entity_list":[] # 必须是list
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                new_data.append(data)
        return new_data
    
    def _pre_process(self, path):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                new_data.append({
                    "type": 0, # 抽主体
                    "text": text,
                    "entity_list":[(spo["subject"], spo['subject_type']) if args.with_type else spo["subject"] for spo in spo_list] if self.args.finetuned_model_name == 'gpnerace05' \
                                  else [(spo["object"], spo["object_type"]) if args.with_type else spo["object"] for spo in spo_list]
                })
                if self.args.finetuned_model_name == 'gpnerace05':
                    new_data.extend(self.build_data12(text, spo_list, self.subject_predicate_dic, 1, self.args.negative_samples_rate))
                else:
                    new_data.extend(self.build_data12(text, spo_list, self.object_predicate_dic, 2, self.args.negative_samples_rate))
#                 new_data.extend(self.build_data12(text, spo_list, self.object_predicate_dic, 2, 0.25))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data


class GPNER3DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i+1 for i, v in enumerate(predicates)}
        relations = []
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["object_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype   
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate, entity_type=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
        
    def build_data12(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate, input_entity_type)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # S+P 抽客体
                    "text": prefix_text if args.add_prefix else text, 
                    "entity_list":[], # 必须是list
                    "prefix_entity": input_entity
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for _ in range(iter_num):
                    new_data.append(data)
        
        if args.add_wrong_entity_neg_sample:
            entity2type_dic = {}
            for spo in line['entity_list']:
                entity2type_dic[spo['entity']] = spo['entity_type']
            gold_entities = set([spo['subject'] for spo in line['spo_list']])
            pred_entities = set([spo['entity'] for spo in line['entity_list']])
            wrong_entities = list(pred_entities-gold_entities)
#             print('wrong_entities:',wrong_entities)
            for input_entity in wrong_entities:
                entity_type = entity2type_dic[input_entity]
                for predicate in entity2predicate_dic[entity_type]:
                    if random.random() < args.wrong_entity_neg_sample_rate:
                        prefix_text = self.add_prefix(text, input_entity, predicate)
                        data = {
                            "type": data_type, # O+P 抽主体
                            "text": prefix_text, 
                            "entity_list":[], # 必须是list
                            "prefix_entity": input_entity,
                            "neg_type": 'wrong_entity'
                        }
                        for i in range(iter_num):
                            new_data.append(data)
        
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            if args.add_wrong_entity_neg_sample:
                with jsonlines.open('./result_output/gpner3/entity_list_train.jsonl') as r:
                    f1 = [line for line in r]
            else:
                f =  [i for i in f]
                f1 = [0 for i in f] # 假数据，方便后续zip不报错
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line, line1 in zip(f, f1):
                if args.add_wrong_entity_neg_sample:
                    line['entity_list'] = line1['entity_list']
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_reina_prefix_0(text, line['bm25_list']) if self.args.is_reina and mode == 'train' else text,
                        "entity_list": [(spo["subject"], spo["subject_type"])
                                    for spo in spo_list]
                    })
                new_data.extend(self.build_data12(line, self.subject_predicate_dic, 1, self.args.negative_samples_rate, mode))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER4DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["subject_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def add_prefix(self, text, entity, predicate, entity_type=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
    
    def build_data12(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate, input_entity_type)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # O+P 抽主体
                    "text": prefix_text if args.add_prefix else text, 
                    "entity_list":[], # 必须是list
                    "prefix_entity": input_entity
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for i in range(iter_num):
                    new_data.append(data)
        
        if args.add_wrong_entity_neg_sample:
            entity2type_dic = {}
            for spo in line['entity_list']:
                entity2type_dic[spo['entity']] = spo['entity_type']
            gold_entities = set([spo['object']['@value'] for spo in line['spo_list']])
            pred_entities = set([spo['entity'] for spo in line['entity_list']])
            wrong_entities = list(pred_entities-gold_entities)
#             print('wrong_entities:',wrong_entities)
            for input_entity in wrong_entities:
                entity_type = entity2type_dic[input_entity]
                for predicate in entity2predicate_dic[entity_type]:
                    if random.random() < args.wrong_entity_neg_sample_rate:
                        prefix_text = self.add_prefix(text, input_entity, predicate, entity_type)
                        data = {
                            "type": data_type, # O+P 抽主体
                            "text": prefix_text, 
                            "entity_list":[], # 必须是list
                            "prefix_entity": input_entity,
                            "neg_type": 'wrong_entity'
                        }
                        for i in range(iter_num):
                            new_data.append(data)
        
        return new_data
    
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            if args.add_wrong_entity_neg_sample:
                with jsonlines.open('./result_output/gpner4/entity_list_train.jsonl') as r:
                    f1 = [line for line in r]
            else:
                f =  [i for i in f]
                f1 = [0 for i in f] # 假数据，方便后续zip不报错
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line, line1 in zip(f, f1):
                if args.add_wrong_entity_neg_sample:
                    line['entity_list'] = line1['entity_list']
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_reina_prefix_0(text, line['bm25_list']) if self.args.is_reina and mode == 'train' else text,
                        "entity_list":[(spo["object"]["@value"], spo["object_type"]["@value"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
#                 new_data.extend(self.build_data12(text, spo_list, self.subject_predicate_dic, 1, 0.1))
                new_data.extend(self.build_data12(line, self.object_predicate_dic, 2, self.args.negative_samples_rate, mode))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER9DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
#         print(self.train_path)
#         print(self.dev_path)
#         print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据:', self.train_path)
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train_text数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data
    
    def get_dev_sample(self):
        print('读取dev数据:', self.dev_path)
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据:', self.test_path)
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["subject_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def add_prefix(self, text, entity, predicate, entity_type=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
    
    def build_data12(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate, input_entity_type)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # O+P 抽主体
                    "text": self.add_reina_prefix_12(prefix_text, line['bm25_list'], input_entity, predicate) if self.args.is_reina and mode == 'train' else prefix_text, 
                    "entity_list":[] # 必须是list
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for i in range(iter_num):
                    new_data.append(data)
        if args.add_wrong_entity_neg_sample:
            entity2type_dic = {}
            for spo in line['entity_list']:
                entity2type_dic[spo['entity']] = spo['entity_type']
            gold_entities = set([spo['object']['@value'] for spo in line['spo_list']])
            pred_entities = set([spo['entity'] for spo in line['entity_list']])
            wrong_entities = list(pred_entities-gold_entities)
#             print('wrong_entities:', wrong_entities)
            for input_entity in wrong_entities:
                entity_type = entity2type_dic[input_entity]
                for predicate in entity2predicate_dic[entity_type]:
                    if random.random() < args.wrong_entity_neg_sample_rate:
                        prefix_text = self.add_prefix(text, input_entity, predicate, entity_type)
                        data = {
                            "type": data_type, # O+P 抽主体
                            "text": prefix_text, 
                            "entity_list":[], # 必须是list
                            "neg_type": 'wrong_entity'
                        }
                        for i in range(iter_num):
                            new_data.append(data)
        return new_data
    
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
#             if args.add_wrong_entity_neg_sample:
#                 with jsonlines.open('./result_output/gpner9/entity_list_train.jsonl') as r:
#                     f1 = [line for line in r]
#             else:
#                 f = [i for i in f]
#                 f1 = [0 for i in f]
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
#             for line, line1 in zip(f, f1):
            for line in f:
#                 if args.add_wrong_entity_neg_sample:
#                     line['entity_list'] = line1['entity_list']
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["object"]["@value"], spo["object_type"]["@value"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
#                 new_data.extend(self.build_data12(text, spo_list, self.subject_predicate_dic, 1, 0.1))
                new_data.extend(self.build_data12(line, self.object_predicate_dic, 2, self.args.negative_samples_rate, mode))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER5SubDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i+1 for i, v in enumerate(predicates)}
        relations = []
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["object_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype   
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate, entity_type=None, bm25=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
        
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo["subject"], spo["subject_type"])
                                    for spo in spo_list]
                    })

#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER5Sp2oDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i+1 for i, v in enumerate(predicates)}
        relations = []
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["object_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype   
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate, entity_type=None, bm25=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
        
    def build_data12(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate, input_entity_type)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # S+P 抽客体
                    "input_entity_type": input_entity_type,
                    "predicate": predicate,
                    "text":  self.add_reina_prefix_12(prefix_text, line['bm25_list'], input_entity, predicate) if self.args.is_reina and mode == 'train' else prefix_text, 
                    "entity_list":[] # 必须是list
                    
                } 
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for _ in range(iter_num):
                        new_data.append(data)
#             if data["entity_list"] != []:
#                 print('正样本')
#                 for _ in range(iter_num):
#                     new_data.append(data)
#             elif data['predicate'] in self.subject_predicate_dic[data['input_entity_type']]:
#                 if random.random() < args.neg_wrong_rel_same_type_rate:
#                     print('same')
#                     for _ in range(iter_num):
#                         new_data.append(data)
#             else:
#                 if random.random() < args.neg_wrong_rel_diff_type_rate:
#                     print('diff')
#                     for _ in range(iter_num):
#                         new_data.append(data)
        if args.add_wrong_entity_neg_sample:
            entity2type_dic = {}
            for spo in line['entity_list']:
                entity2type_dic[spo['entity']] = spo['entity_type']
            gold_entities = set([spo['subject'] for spo in line['spo_list']])
            pred_entities = set([spo['entity'] for spo in line['entity_list']])
            wrong_entities = list(pred_entities-gold_entities)
#             print('wrong_entities:',wrong_entities)
            for input_entity in wrong_entities:
                entity_type = entity2type_dic[input_entity]
                for predicate in entity2predicate_dic[entity_type]:
                    if random.random() < args.wrong_entity_neg_sample_rate:
                        prefix_text = self.add_prefix(text, input_entity, predicate)
                        data = {
                            "type": data_type, # O+P 抽主体
                            "text": prefix_text, 
                            "entity_list":[], # 必须是list
                            "neg_type": 'wrong_entity'
                        }
                        for i in range(iter_num):
                            new_data.append(data)
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            if args.add_wrong_entity_neg_sample:
                with jsonlines.open('./result_output/gpner5/entity_list_train.jsonl') as r:
                    f1 = [line for line in r]
            else:
                f =  [i for i in f]
                f1 = [0 for i in f] # 假数据，方便后续zip不报错
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line, line1 in zip(f, f1):
                if args.add_wrong_entity_neg_sample:
                    line['entity_list'] = line1['entity_list']
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
#                 for _ in range(iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": self.add_reina_prefix_0(text, line['bm25_list']) if self.args.is_reina and mode == 'train' else text,
#                         "entity_list": [(spo["subject"], spo["subject_type"])
#                                     for spo in spo_list]
#                     })
                new_data.extend(self.build_data12(line, self.subject_predicate_dic, 1, self.args.negative_samples_rate, mode))
#                 new_data.extend(self.build_data12(line, self.subject_predicate_dic, 1, 1, mode))

#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data


class GPNER6ObjDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["subject_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def add_prefix(self, text, entity, predicate, entity_type=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
    
    
    
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["object"]["@value"], spo["object_type"]["@value"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER6Op2sDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["subject_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def add_prefix(self, text, entity, predicate, entity_type=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
    
    
    def build_data12(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # O+P 抽主体
                    "text": self.add_reina_prefix_12(prefix_text, line['bm25_list'], input_entity, predicate) if self.args.is_reina and mode == 'train' else prefix_text, 
                    "entity_list":[] # 必须是list
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for i in range(iter_num):
                    new_data.append(data)
        if args.add_wrong_entity_neg_sample:
            entity2type_dic = {}
            for spo in line['entity_list']:
                entity2type_dic[spo['entity']] = spo['entity_type']
            gold_entities = set([spo['object']['@value'] for spo in line['spo_list']])
            pred_entities = set([spo['entity'] for spo in line['entity_list']])
            wrong_entities = list(pred_entities-gold_entities)
#             print('wrong_entities:', wrong_entities)
            for input_entity in wrong_entities:
                entity_type = entity2type_dic[input_entity]
                for predicate in entity2predicate_dic[entity_type]:
                    if random.random() < args.wrong_entity_neg_sample_rate:
                        prefix_text = self.add_prefix(text, input_entity, predicate)
                        data = {
                            "type": data_type, # O+P 抽主体
                            "text": prefix_text, 
                            "entity_list":[], # 必须是list
                            "neg_type": 'wrong_entity'
                        }
                        for i in range(iter_num):
                            new_data.append(data)
        return new_data
    
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            if args.add_wrong_entity_neg_sample:
                with jsonlines.open('./result_output/gpner6/entity_list_train.jsonl') as r:
                    f1 = [line for line in r]
            else:
                f =  [i for i in f]
                f1 = [0 for i in f] # 假数据，方便后续zip不报错
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line, line1 in zip(f, f1):
                if args.add_wrong_entity_neg_sample:
                    line['entity_list'] = line1['entity_list']
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
#                 for i in range (iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": self.add_reina_prefix_0(text, line['bm25_list']) if self.args.is_reina and mode == 'train' else text,
#                         "entity_list":[(spo["object"]["@value"], spo["object_type"]["@value"])
#                                     for spo in spo_list],
#                         "prefix_entity": ''
#                     })
#                 new_data.extend(self.build_data12(text, spo_list, self.subject_predicate_dic, 1, 0.1))
                new_data.extend(self.build_data12(line, self.object_predicate_dic, 2, self.args.negative_samples_rate, mode))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER7DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    def add_prefix(self, text, entity, predicate):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.class2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        
#     def add_prefix(self, text, entity, predicate):
#         return f"{entity}[unused1]{predicate}[unused2]{text}"
    
    def entity_marker(self, entity):
        return '[unused3]' + entity + '[unused4]'
    
    def del_entity_marker(self, entity):
        return entity[9:-9]
    
    def add_entity_marker(self, text, entity, predicate):
        entity_start = self.search(entity, text)
        entity_end = entity_start + len(entity)
        text2 = text[:entity_start] + self.entity_marker(entity) + text[entity_end:]
        return self.add_prefix(text2, entity, predicate)
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'],spo['predicate'])].append(spo['object']['@value'])
            
        for key, objs in positive_dic.items():
            sub, predicate = key
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_entity_marker(text, sub, predicate) if args.do_entity_marker else self.add_prefix(text, sub, predicate), 
                    "entity_list":[(obj, predicate) for obj in objs], # 必须是list
                    "prefix_entity": self.entity_marker(sub) if args.do_entity_marker else sub
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo["subject"], spo["predicate"])
                                    for spo in spo_list]
                    })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER8DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    def add_prefix(self, text, entity, predicate):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.class2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        
#     def add_prefix(self, text, entity, predicate):
#         return f"{entity}[unused1]{predicate}[unused2]{text}"
    
    def entity_marker(self, entity):
        return '[unused3]' + entity + '[unused4]'
    
    def del_entity_marker(self, entity):
        return entity[9:-9]
    
    def add_entity_marker(self, text, entity, predicate):
        entity_start = self.search(entity, text)
        entity_end = entity_start + len(entity)
        text2 = text[:entity_start] + self.entity_marker(entity) + text[entity_end:]
        return self.add_prefix(text2, entity, predicate)
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['object']['@value'],spo['predicate'])].append(spo['subject'])
            
        for key, subjs in positive_dic.items():
            obj, predicate = key
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_entity_marker(text, obj, predicate) if args.do_entity_marker else self.add_prefix(text, obj, predicate), 
                    "entity_list":[(sub, predicate) for sub in subjs], # 必须是list
                    "prefix_entity": self.entity_marker(obj) if args.do_entity_marker else obj
                })
        return new_data
        
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["object"]["@value"], spo["predicate"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPFilter78DataProcessor(object):
    def __init__(self, args):
        self.args = args
        root = args.data_dir
        self.train_path = os.path.join(root, args.train_file)
        print(self.train_path)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        self.num_labels = len(self.predicate2id.keys())
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, mode='dev')

    def get_test_sample(self):
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip())["text"] for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        schema = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                  '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                  '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                  '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                  '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.schema = schema
        self.predicate2id = {v: i for i, v in enumerate(schema)}
        self.id2predicate = {i: v for i, v in enumerate(schema)}
    
    def _pre_process(self, path, mode):
        new_data = []
        with open(path, 'r', encoding='utf-8') as f:
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                line = json.loads(line)
                for _ in range(iter_num):
                    new_data.append({
                        "text":line["text"],
                        "spo_list":[(spo["subject"], spo["predicate"], spo["object"]["@value"])
                                    for spo in line["spo_list"]]
                    })
        return new_data
    
    def regular(self, spo):
        """
        判断spo是否符合规则
        return bool 
        """
        sub = spo['subject']
        sub_type = spo['subject_type']
        if sub_type == '疾病' and len(sub) == 1 and sub != '痔':
            return False
        return True
    
class GPNER17DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate):
        return f"{entity}[unused1]{predicate}[unused2]{text}"
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'],spo['predicate'])].append(spo['object']['@value'])
            
        for key, objs in positive_dic.items():
            sub, predicate = key
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, sub, predicate), 
                    "entity_list":[(obj, predicate) for obj in objs] # 必须是list
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo["subject"], spo["predicate"])
                                    for spo in spo_list]
                    })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER18DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate):
        return f"{entity}[unused1]{predicate}[unused2]{text}"
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['object']['@value'],spo['predicate'])].append(spo['subject'])
            
        for key, subjs in positive_dic.items():
            obj, predicate = key
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, obj, predicate), 
                    "entity_list":[(sub, predicate) for sub in subjs] # 必须是list
                })
        return new_data
        
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["object"]["@value"], spo["predicate"])
                                    for spo in spo_list]
                    })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class CasRelDataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                if spo_list != []:
                    new_data.append({
                        "text": text,
                        "spo_list":[(spo["subject"], spo["predicate"], spo["object"]["@value"])
                                    for spo in spo_list],
                    })
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER11DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate):
        return f"{entity}[unused1]{predicate}[unused2]{text}"
    
    def entity_marker(self, entity):
        return '[unused3]' + entity + '[unused4]'
    
    def del_entity_marker(self, entity):
        return entity[9:-9]
    
    def add_entity_marker(self, text, entity, predicate):
        entity_start = self.search(entity, text)
        entity_end = entity_start + len(entity)
        text2 = text[:entity_start] + self.entity_marker(entity) + text[entity_end:]
        return self.add_prefix(text2, entity, predicate)
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['object']['@value'],spo['predicate'])].append(spo['subject'])
            
        for key, subjs in positive_dic.items():
            obj, predicate = key
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_entity_marker(text, obj, predicate) if args.do_entity_marker else self.add_prefix(text, obj, predicate), 
                    "entity_list":[(sub, predicate) for sub in subjs], # 必须是list
                    "prefix_entity": self.entity_marker(obj) if args.do_entity_marker else obj
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo["subject"], spo["predicate"])
                                    for spo in spo_list]
                    })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER12DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate):
        return f"{entity}[unused1]{predicate}[unused2]{text}"
    
    def entity_marker(self, entity):
        return '[unused3]' + entity + '[unused4]'
    
    def del_entity_marker(self, entity):
        return entity[9:-9]
    
    def add_entity_marker(self, text, entity, predicate):
        entity_start = self.search(entity, text)
        entity_end = entity_start + len(entity)
        text2 = text[:entity_start] + self.entity_marker(entity) + text[entity_end:]
        return self.add_prefix(text2, entity, predicate)
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'],spo['predicate'])].append(spo['object']['@value'])
            
        for key, objs in positive_dic.items():
            sub, predicate = key
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_entity_marker(text, sub, predicate) if args.do_entity_marker else self.add_prefix(text, sub, predicate), 
                    "entity_list":[(obj, predicate) for obj in objs], # 必须是list
                    "prefix_entity": self.entity_marker(sub) if args.do_entity_marker else sub
                })
        return new_data
        
    
        
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["object"]["@value"], spo["predicate"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER13DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, mode='sub'):
        if mode == 'sub':
            return f"[unused1][unused2][unused3]{text}"
        elif mode == 'obj':
            return f"[unused4][unused5][unused6]{text}"
        
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_prefix(text, 'sub'),
                        "entity_list": [(spo["subject"], spo["predicate"])
                                    for spo in spo_list]
                    })
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽客体
                        "text": self.add_prefix(text, 'obj'),
                        "entity_list": [(spo["object"]['@value'], spo["predicate"])
                                    for spo in spo_list]
                    })
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER14DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate):
        return f"{entity}[unused1]{predicate}[unused2]{text}"
    
    def entity_marker(self, entity):
        return '[unused3]' + entity + '[unused4]'
    
    def del_entity_marker(self, entity):
        return entity[9:-9]
    
    def add_entity_marker(self, text, entity, predicate):
        entity_start = self.search(entity, text)
        entity_end = entity_start + len(entity)
        text2 = text[:entity_start] + self.entity_marker(entity) + text[entity_end:]
        return self.add_prefix(text2, entity, predicate)
        
    def build_data12_sp2o(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'],spo['predicate'])].append(spo['object']['@value'])
            
        for key, objs in positive_dic.items():
            sub, predicate = key
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_entity_marker(text, sub, predicate) if args.do_entity_marker else self.add_prefix(text, sub, predicate), 
                    "entity_list":[(obj, predicate) for obj in objs], # 必须是list
                    "prefix_entity": self.entity_marker(sub) if args.do_entity_marker else sub
                })
        return new_data
    
    def build_data12_op2s(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['object']['@value'],spo['predicate'])].append(spo['subject'])
            
        for key, subjs in positive_dic.items():
            obj, predicate = key
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_entity_marker(text, obj, predicate) if args.do_entity_marker else self.add_prefix(text, obj, predicate), 
                    "entity_list":[(sub, predicate) for sub in subjs], # 必须是list
                    "prefix_entity": self.entity_marker(obj) if args.do_entity_marker else obj
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
#                 for _ in range(iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": text,
#                         "entity_list": [(spo["subject"], spo["predicate"])
#                                     for spo in spo_list]
#                     })
                new_data.extend(self.build_data12_sp2o(line))
                new_data.extend(self.build_data12_op2s(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER15DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    # 构建类型为1 和 2 的数据
    def add_xp2x_prefix(self, text, entity, predicate):
        return f"{entity}[unused1]{predicate}[unused2]{text}"
    
    def add_entity_prefix(self, text, mode='sub'):
        if mode == 'sub':
            return f"[unused1][unused2][unused3]{text}"
        elif mode == 'obj':
            return f"[unused4][unused5][unused6]{text}"
    
    def entity_marker(self, entity):
        return '[unused3]' + entity + '[unused4]'
    
    def del_entity_marker(self, entity):
        return entity[9:-9]
    
    def add_entity_marker(self, text, entity, predicate):
        entity_start = self.search(entity, text)
        entity_end = entity_start + len(entity)
        text2 = text[:entity_start] + self.entity_marker(entity) + text[entity_end:]
        return self.add_prefix(text2, entity, predicate)
        
    def build_data12_sp2o(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'],spo['predicate'])].append(spo['object']['@value'])
            
        for key, objs in positive_dic.items():
            sub, predicate = key
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_entity_marker(text, sub, predicate) if args.do_entity_marker else self.add_xp2x_prefix(text, sub, predicate), 
                    "entity_list":[(obj, predicate) for obj in objs], # 必须是list
                    "prefix_entity": self.entity_marker(sub) if args.do_entity_marker else sub
                })
        return new_data
    
    def build_data12_op2s(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['object']['@value'],spo['predicate'])].append(spo['subject'])
            
        for key, subjs in positive_dic.items():
            obj, predicate = key
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_entity_marker(text, obj, predicate) if args.do_entity_marker else self.add_xp2x_prefix(text, obj, predicate), 
                    "entity_list":[(sub, predicate) for sub in subjs], # 必须是list
                    "prefix_entity": self.entity_marker(obj) if args.do_entity_marker else obj
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_entity_prefix(text, 'sub'),
                        "entity_list": [(spo["subject"], spo["predicate"])
                                    for spo in spo_list]
                    })
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽客体
                        "text": self.add_entity_prefix(text, 'obj'),
                        "entity_list": [(spo["object"]['@value'], spo["predicate"])
                                    for spo in spo_list]
                    })
                
                new_data.extend(self.build_data12_sp2o(line))
                new_data.extend(self.build_data12_op2s(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER21DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate):
        return f"{entity}[unused1]{predicate}[unused2]{text}"
    
    def entity_marker(self, entity):
        return '[unused3]' + entity + '[unused4]'
    
    def del_entity_marker(self, entity):
        return entity[9:-9]
    
    def add_entity_marker(self, text, entity, predicate):
        entity_start = self.search(entity, text)
        entity_end = entity_start + len(entity)
        text2 = text[:entity_start] + self.entity_marker(entity) + text[entity_end:]
        return self.add_prefix(text2, entity, predicate)
        
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo["subject"], spo["predicate"])
                                    for spo in spo_list]
                    })
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER23DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate):
        return f"{entity}[unused1]{predicate}[unused2]{text}"
    
    def entity_marker(self, entity):
        return '[unused3]' + entity + '[unused4]'
    
    def del_entity_marker(self, entity):
        return entity[9:-9]
    
    def add_entity_marker(self, text, entity, predicate):
        entity_start = self.search(entity, text)
        entity_end = entity_start + len(entity)
        text2 = text[:entity_start] + self.entity_marker(entity) + text[entity_end:]
        return self.add_prefix(text2, entity, predicate)
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'],spo['predicate'])].append(spo['object']['@value'])
            
        for key, objs in positive_dic.items():
            sub, predicate = key
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_entity_marker(text, sub, predicate) if args.do_entity_marker else self.add_prefix(text, sub, predicate), 
                    "entity_list":[(obj, predicate) for obj in objs], # 必须是list
                    "prefix_entity": self.entity_marker(sub) if args.do_entity_marker else sub
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
#                 for _ in range(iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": text,
#                         "entity_list": [(spo["subject"], spo["predicate"])
#                                     for spo in spo_list]
#                     })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER22DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate):
        return f"{entity}[unused1]{predicate}[unused2]{text}"
    
    def entity_marker(self, entity):
        return '[unused3]' + entity + '[unused4]'
    
    def del_entity_marker(self, entity):
        return entity[9:-9]
    
    def add_entity_marker(self, text, entity, predicate):
        entity_start = self.search(entity, text)
        entity_end = entity_start + len(entity)
        text2 = text[:entity_start] + self.entity_marker(entity) + text[entity_end:]
        return self.add_prefix(text2, entity, predicate)
        
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["object"]["@value"], spo["predicate"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER24DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate):
        return f"{entity}[unused1]{predicate}[unused2]{text}"
    
    def entity_marker(self, entity):
        return '[unused3]' + entity + '[unused4]'
    
    def del_entity_marker(self, entity):
        return entity[9:-9]
    
    def add_entity_marker(self, text, entity, predicate):
        entity_start = self.search(entity, text)
        entity_end = entity_start + len(entity)
        text2 = text[:entity_start] + self.entity_marker(entity) + text[entity_end:]
        return self.add_prefix(text2, entity, predicate)
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['object']['@value'],spo['predicate'])].append(spo['subject'])
            
        for key, subjs in positive_dic.items():
            obj, predicate = key
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_entity_marker(text, obj, predicate) if args.do_entity_marker else self.add_prefix(text, obj, predicate), 
                    "entity_list":[(sub, predicate) for sub in subjs], # 必须是list
                    "prefix_entity": self.entity_marker(obj) if args.do_entity_marker else obj
                })
        return new_data
        
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
#                 for i in range (iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": text,
#                         "entity_list":[(spo["object"]["@value"], spo["predicate"])
#                                     for spo in spo_list],
#                         "prefix_entity": ''
#                     })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER25DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
         
    def build_data(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[spo['subject']].append((spo['object']['@value'], spo['predicate']))
            
        sub_list = [spo['subject'] for spo in spo_list]
        for sub, objs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "text": text,
                    "entity1_list": sub_list,
                    "entity1": sub,
                    "entity2_list": [(obj, predicate) for obj, predicate in objs]
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
#                 for _ in range(iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": text,
#                         "entity_list": [(spo["subject"], spo["predicate"])
#                                     for spo in spo_list]
#                     })
                new_data.extend(self.build_data(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER26DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        
    def build_data(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[spo['object']['@value']].append((spo['subject'], spo['predicate']))
            
        obj_list = [spo['object']['@value'] for spo in spo_list]
        for obj, subjs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "text": text,
                    "entity1_list": obj_list,
                    "entity1": obj,
                    "entity2_list": [(sub, predicate) for sub, predicate in subjs]
                })
        return new_data
        
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
#                 for i in range (iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": text,
#                         "entity_list":[(spo["object"]["@value"], spo["predicate"])
#                                     for spo in spo_list],
#                         "prefix_entity": ''
#                     })
                new_data.extend(self.build_data(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# gpner复制版
class GPNER91DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据:', self.train_path)
        return self._pre_process(self.train_path, mode='train')
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i+1 for i, v in enumerate(predicates)}
        relations = []
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["object_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype   
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate, entity_type=None, bm25=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
        
    def build_data12(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate, input_entity_type)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # S+P 抽客体
                    "input_entity_type": input_entity_type,
                    "predicate": predicate,
                    "text":  self.add_reina_prefix_12(prefix_text, line['bm25_list'], input_entity, predicate) if self.args.is_reina and mode == 'train' else prefix_text, 
                    "entity_list":[] # 必须是list
                    
                } 
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for _ in range(iter_num):
                        new_data.append(data)

        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo["subject"], spo["subject_type"])
                                    for spo in spo_list]
                    })
                new_data.extend(self.build_data12(line, self.subject_predicate_dic, 1, self.args.negative_samples_rate, mode))

#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# gpner9复制版
class GPNER99DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据:', self.train_path)
        return self._pre_process(self.train_path, mode='train')
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["subject_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def add_prefix(self, text, entity, predicate, entity_type=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
    
    def build_data12(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # O+P 抽主体
                    "text": prefix_text, 
                    "entity_list":[] # 必须是list
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for i in range(iter_num):
                    new_data.append(data)
        return new_data
    
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["object"]["@value"], spo["object_type"]["@value"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
                new_data.extend(self.build_data12(line, self.object_predicate_dic, 2, self.args.negative_samples_rate, mode))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER31DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["subject_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def add_prefix(self, text, entity, predicate, entity_type=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
    
    def build_data12(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # O+P 抽主体
                    "text": prefix_text, 
                    "entity_list":[] # 必须是list
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for i in range(iter_num):
                    new_data.append(data)
        return new_data
    
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["subject"], spo["subject_type"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
                new_data.extend(self.build_data12(line, self.object_predicate_dic, 2, self.args.negative_samples_rate, mode))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER32DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i+1 for i, v in enumerate(predicates)}
        relations = []
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["object_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype   
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate, entity_type=None, bm25=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
        
    def build_data12(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate, input_entity_type)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # S+P 抽客体
                    "input_entity_type": input_entity_type,
                    "predicate": predicate,
                    "text":  self.add_reina_prefix_12(prefix_text, line['bm25_list'], input_entity, predicate) if self.args.is_reina and mode == 'train' else prefix_text, 
                    "entity_list":[] # 必须是list
                    
                } 
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for _ in range(iter_num):
                        new_data.append(data)

        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo["object"]['@value'], spo["object_type"]['@value'])
                                    for spo in spo_list]
                    })
                new_data.extend(self.build_data12(line, self.subject_predicate_dic, 1, self.args.negative_samples_rate, mode))

#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER33DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据:', self.train_path)
        return self._pre_process(self.train_path, mode='train')
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["subject_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def add_prefix(self, text, mode='sub'):
        if mode == 'sub':
            return f"[unused1][unused2][unused3]{text}"
        elif mode == 'obj':
            return f"[unused4][unused5][unused6]{text}"
    
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_prefix(text, 'sub'),
                        "entity_list":[(spo["subject"], spo["subject_type"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_prefix(text, 'obj'),
                        "entity_list":[(spo["object"]["@value"], spo["object_type"]["@value"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER34DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据:', self.train_path)
        return self._pre_process(self.train_path, mode='train')
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["subject_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def add_prefix(self, text, entity, predicate, entity_type=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
    
    def build_data12_op2s(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # O+P 抽主体
                    "text": prefix_text, 
                    "entity_list":[] # 必须是list
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for i in range(iter_num):
                    new_data.append(data)
        return new_data
    
    def build_data12_sp2o(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate, input_entity_type)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # S+P 抽客体
                    "input_entity_type": input_entity_type,
                    "predicate": predicate,
                    "text":  self.add_reina_prefix_12(prefix_text, line['bm25_list'], input_entity, predicate) if self.args.is_reina and mode == 'train' else prefix_text, 
                    "entity_list":[] # 必须是list
                    
                } 
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for _ in range(iter_num):
                        new_data.append(data)

        return new_data
    
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
#                 for i in range (iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": text,
#                         "entity_list":[(spo["object"]["@value"], spo["object_type"]["@value"])
#                                     for spo in spo_list],
#                         "prefix_entity": ''
#                     })
                new_data.extend(self.build_data12_op2s(line, self.object_predicate_dic, 2, self.args.negative_samples_op2s_rate, mode))
                new_data.extend(self.build_data12_sp2o(line, self.subject_predicate_dic, 1, self.args.negative_samples_sp2o_rate, mode))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER35DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据:', self.train_path)
        return self._pre_process(self.train_path, mode='train')
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["subject_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
    
    
    def add_entity_prefix(self, text, mode='sub'):
        if mode == 'sub':
            return f"[unused1][unused2][unused3]{text}"
        elif mode == 'obj':
            return f"[unused4][unused5][unused6]{text}"
    
    def add_xp2x_prefix(self, text, entity, predicate, entity_type=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
    
    def build_data12_op2s(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_xp2x_prefix(text, input_entity, predicate)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # O+P 抽主体
                    "text": prefix_text, 
                    "entity_list":[] # 必须是list
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for i in range(iter_num):
                    new_data.append(data)
        return new_data
    
    def build_data12_sp2o(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_xp2x_prefix(text, input_entity, predicate, input_entity_type)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # S+P 抽客体
                    "input_entity_type": input_entity_type,
                    "predicate": predicate,
                    "text": prefix_text, 
                    "entity_list":[] # 必须是list
                    
                } 
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for _ in range(iter_num):
                        new_data.append(data)

        return new_data
    
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_entity_prefix(text, 'sub'),
                        "entity_list":[(spo["subject"], spo["subject_type"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_entity_prefix(text, 'obj'),
                        "entity_list":[(spo["object"]["@value"], spo["object_type"]["@value"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
                new_data.extend(self.build_data12_op2s(line, self.object_predicate_dic, 2, self.args.negative_samples_op2s_rate, mode))
                new_data.extend(self.build_data12_sp2o(line, self.subject_predicate_dic, 1, self.args.negative_samples_sp2o_rate, mode))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER36DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据:', self.train_path)
        return self._pre_process(self.train_path, mode='train')
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i+1 for i, v in enumerate(predicates)}
        relations = []
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["object_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype   
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo["subject"], spo["subject_type"])
                                    for spo in spo_list]
                    })

#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER38DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据:', self.train_path)
        return self._pre_process(self.train_path, mode='train')
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i+1 for i, v in enumerate(predicates)}
        relations = []
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["object_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype   
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    # 构建类型为1 和 2 的数据
    def add_prefix(self, text, entity, predicate, entity_type=None, bm25=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
        
    def build_data12(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate, input_entity_type)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # S+P 抽客体
                    "input_entity_type": input_entity_type,
                    "predicate": predicate,
                    "text":  self.add_reina_prefix_12(prefix_text, line['bm25_list'], input_entity, predicate) if self.args.is_reina and mode == 'train' else prefix_text, 
                    "entity_list":[] # 必须是list
                    
                } 
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for _ in range(iter_num):
                        new_data.append(data)

        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
#                 for _ in range(iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": text,
#                         "entity_list": [(spo["subject"], spo["subject_type"])
#                                     for spo in spo_list]
#                     })
                new_data.extend(self.build_data12(line, self.subject_predicate_dic, 1, self.args.negative_samples_rate, mode))

#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER37DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据:', self.train_path)
        return self._pre_process(self.train_path, mode='train')
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["subject_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["object"]["@value"], spo["object_type"]["@value"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

class GPNER39DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据:', self.train_path)
        return self._pre_process(self.train_path, mode='train')
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i for i, v in enumerate(predicates)}
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["subject_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
    def add_prefix(self, text, entity, predicate, entity_type=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
    
    def build_data12(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_prefix(text, input_entity, predicate)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # O+P 抽主体
                    "text": prefix_text, 
                    "entity_list":[] # 必须是list
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for i in range(iter_num):
                    new_data.append(data)
        return new_data
    
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
#                 for i in range (iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": text,
#                         "entity_list":[(spo["object"]["@value"], spo["object_type"]["@value"])
#                                     for spo in spo_list],
#                         "prefix_entity": ''
#                     })
                new_data.extend(self.build_data12(line, self.object_predicate_dic, 2, self.args.negative_samples_rate, mode))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R7D
class GPNER41DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
    
    def add_prefix(self, text, entity):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{text[:start]}[unused1]{entity}[unused2]{text[end:]}"
        elif args.prefix_mode == 'entity':
            return f"{entity}[unused1]{text}"
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'])].append((spo['object']['@value'], spo['predicate']))
            
        for sub, objs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, sub), 
                    "entity_list":[(obj, predicate) for obj, predicate in objs], # 必须是list
                    "prefix_entity": sub
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo["subject"], spo["predicate"])
                                    for spo in spo_list]
                    })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R8D
class GPNER42DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
            
    def add_prefix(self, text, entity):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{text[:start]}[unused1]{entity}[unused2]{text[end:]}"
        elif args.prefix_mode == 'entity':
            return f"{entity}[unused1]{text}"
    
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[spo['object']['@value']].append((spo['subject'],spo['predicate']))
            
        for obj, subjs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, obj), 
                    "entity_list":[(sub, predicate) for sub, predicate in subjs], # 必须是list
                    "prefix_entity": obj
                })
        return new_data
        
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["object"]["@value"], spo["predicate"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R41D
class GPNER51DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
    
    def add_prefix(self, text, entity):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{text[:start]}[unused1]{entity}[unused2]{text[end:]}"
        elif args.prefix_mode == 'entity':
            return f"{entity}[unused1]{text}"
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[spo['object']['@value']].append((spo['subject'],spo['predicate']))
            
        for obj, subjs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, obj), 
                    "entity_list":[(sub, predicate) for sub, predicate in subjs], # 必须是list
                    "prefix_entity": obj
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo["subject"], spo["predicate"])
                                    for spo in spo_list]
                    })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R41D
class GPNER52DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
    
    def add_prefix(self, text, entity):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{text[:start]}[unused1]{entity}[unused2]{text[end:]}"
        elif args.prefix_mode == 'entity':
            return f"{entity}[unused1]{text}"
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'])].append((spo['object']['@value'], spo['predicate']))
            
        for sub, objs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, sub), 
                    "entity_list":[(obj, predicate) for obj, predicate in objs], # 必须是list
                    "prefix_entity": sub
                })
        return new_data
    
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["object"]["@value"], spo["predicate"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R41D
class GPNER53DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
    
    def add_prefix(self, text, mode='sub'):
        if mode == 'sub':
            return f"[unused1][unused2][unused3]{text}"
        elif mode == 'obj':
            return f"[unused4][unused5][unused6]{text}"
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'])].append((spo['object']['@value'], spo['predicate']))
            
        for sub, objs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, sub), 
                    "entity_list":[(obj, predicate) for obj, predicate in objs], # 必须是list
                    "prefix_entity": sub
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_prefix(text, 'sub'),
                        "entity_list": [(spo["subject"], spo["predicate"])
                                    for spo in spo_list]
                    })
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_prefix(text, 'obj'),
                        "entity_list": [(spo["object"]["@value"], spo["predicate"])
                                    for spo in spo_list]
                    })
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R41D
class GPNER54DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
    
    def add_prefix(self, text, entity):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{text[:start]}[unused1]{entity}[unused2]{text[end:]}"
        elif args.prefix_mode == 'entity':
            return f"{entity}[unused1]{text}"
        
    def build_data12_s2po(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'])].append((spo['object']['@value'], spo['predicate']))
            
        for sub, objs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, sub), 
                    "entity_list":[(obj, predicate) for obj, predicate in objs], # 必须是list
                    "prefix_entity": sub
                })
        return new_data
    
    def build_data12_o2ps(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[spo['object']['@value']].append((spo['subject'],spo['predicate']))
            
        for obj, subjs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, obj), 
                    "entity_list":[(sub, predicate) for sub, predicate in subjs], # 必须是list
                    "prefix_entity": obj
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                new_data.extend(self.build_data12_s2po(line))
                new_data.extend(self.build_data12_o2ps(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R41D
class GPNER55DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
    
    def add_entity_prefix(self, text, mode='sub'):
        if mode == 'sub':
            return f"[unused1][unused2][unused3]{text}"
        elif mode == 'obj':
            return f"[unused4][unused5][unused6]{text}"
    
    def add_xp2x_prefix(self, text, entity):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{text[:start]}[unused1]{entity}[unused2]{text[end:]}"
        elif args.prefix_mode == 'entity':
            return f"{entity}[unused1]{text}"
        
    def build_data12_s2po(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'])].append((spo['object']['@value'], spo['predicate']))
            
        for sub, objs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_xp2x_prefix(text, sub), 
                    "entity_list":[(obj, predicate) for obj, predicate in objs], # 必须是list
                    "prefix_entity": sub
                })
        return new_data
    
    def build_data12_o2ps(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[spo['object']['@value']].append((spo['subject'],spo['predicate']))
            
        for obj, subjs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_xp2x_prefix(text, obj), 
                    "entity_list":[(sub, predicate) for sub, predicate in subjs], # 必须是list
                    "prefix_entity": obj
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_entity_prefix(text, 'sub'),
                        "entity_list": [(spo["subject"], spo["predicate"])
                                    for spo in spo_list]
                    })
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_entity_prefix(text, 'obj'),
                        "entity_list": [(spo["object"]["@value"], spo["predicate"])
                                    for spo in spo_list]
                    })
                new_data.extend(self.build_data12_s2po(line))
                new_data.extend(self.build_data12_o2ps(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R41D
class GPNER56DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
            
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo["subject"], spo["predicate"])
                                    for spo in spo_list]
                    })
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R41D
class GPNER58DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
    
    def add_prefix(self, text, entity):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{text[:start]}[unused1]{entity}[unused2]{text[end:]}"
        elif args.prefix_mode == 'entity':
            return f"{entity}[unused1]{text}"
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'])].append((spo['object']['@value'], spo['predicate']))
            
        for sub, objs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, sub), 
                    "entity_list":[(obj, predicate) for obj, predicate in objs], # 必须是list
                    "prefix_entity": sub
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
#                 for _ in range(iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": text,
#                         "entity_list": [(spo["subject"], spo["predicate"])
#                                     for spo in spo_list]
#                     })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R42D
class GPNER57DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
                
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["object"]["@value"], spo["predicate"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R42D
class GPNER59DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
            
    def add_prefix(self, text, entity):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{text[:start]}[unused1]{entity}[unused2]{text[end:]}"
        elif args.prefix_mode == 'entity':
            return f"{entity}[unused1]{text}"
    
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[spo['object']['@value']].append((spo['subject'],spo['predicate']))
            
        for obj, subjs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, obj), 
                    "entity_list":[(sub, predicate) for sub, predicate in subjs], # 必须是list
                    "prefix_entity": obj
                })
        return new_data
        
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
#                 for i in range (iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": text,
#                         "entity_list":[(spo["object"]["@value"], spo["predicate"])
#                                     for spo in spo_list],
#                         "prefix_entity": ''
#                     })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R41D
class GPNER61DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['主体']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
    
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo["subject"], '主体') for spo in spo_list]
                    })
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R41D
class GPNER62DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
    
    def add_prefix(self, text, entity):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{text[:start]}[unused1]{entity}[unused2]{text[end:]}"
        elif args.prefix_mode == 'entity':
            return f"{entity}[unused1]{text}"
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'])].append((spo['object']['@value'], spo['predicate']))
            
        for sub, objs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, sub), 
                    "entity_list":[(obj, predicate) for obj, predicate in objs], # 必须是list
                    "prefix_entity": sub
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
#                 for _ in range(iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": text,
#                         "entity_list": [(spo["subject"], spo["predicate"])
#                                     for spo in spo_list]
#                     })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R42D
class GPNER63DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为客体的实体类型
        labels = ['客体']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
            
    
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo["object"]["@value"], '客体') for spo in spo_list],
                        "prefix_entity": ''
                    })
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R42D
class GPNER64DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_object.json')
        # 临时修改
#         self.test_path = os.path.join('./其他数据', '首次病程.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        if args.do_enhance_CMeEE:
            print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
            
    def add_prefix(self, text, entity):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{text[:start]}[unused1]{entity}[unused2]{text[end:]}"
        elif args.prefix_mode == 'entity':
            return f"{entity}[unused1]{text}"
    
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[spo['object']['@value']].append((spo['subject'],spo['predicate']))
            
        for obj, subjs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, obj), 
                    "entity_list":[(sub, predicate) for sub, predicate in subjs], # 必须是list
                    "prefix_entity": obj
                })
        return new_data
        
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
#                 for i in range (iter_num):
#                     new_data.append({
#                         "type": 0, # 抽主体
#                         "text": text,
#                         "entity_list":[(spo["object"]["@value"], spo["predicate"])
#                                     for spo in spo_list],
#                         "prefix_entity": ''
#                     })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R41D
class GPNER41ACE05DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '65_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = []
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                labels.append(item['subject_type']+'-'+item['predicate']+'-'+item['object_type'])
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
    
    def add_prefix(self, text, entity):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{text[:start]}[unused1]{entity}[unused2]{text[end:]}"
        elif args.prefix_mode == 'entity':
            return f"{entity}[unused1]{text}"
        elif args.prefix_mode == 'entity-ace':
            return f"entity: {entity}, {text}"
        
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[(spo['subject'])].append((spo['object'], spo['subject_type']+'-'+spo['predicate']+'-'+spo['object_type']))
            
        for sub, objs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, sub), 
                    "entity_list":[(obj, predicate) for obj, predicate in objs], # 必须是list
                    "prefix_entity": sub
                })
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list": [(spo['subject'], spo['subject_type']+'-'+spo['predicate']+'-'+spo['object_type'])
                                    for spo in spo_list]
                    })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# R42D
class GPNER42ACE05DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
        self.schema_path = os.path.join(root, '65_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip()) for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = []
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                labels.append(item['subject_type']+'-'+item['predicate']+'-'+item['object_type'])
        self.num_labels = len(labels)
        self.labels = labels
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
            
    def add_prefix(self, text, entity):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{text[:start]}[unused1]{entity}[unused2]{text[end:]}"
        elif args.prefix_mode == 'entity':
            return f"{entity}[unused1]{text}"
        elif args.prefix_mode == 'entity-ace':
            return f"entity: {entity}, {text}"
    
    def build_data12(self, line):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            positive_dic[spo['object']].append((spo['subject'],spo['subject_type']+'-'+spo['predicate']+'-'+spo['object_type']))
            
        for obj, subjs in positive_dic.items():
            iter_num = 2 if self.args.do_rdrop else 1
            for _ in range(iter_num):
                new_data.append({
                    "type": 1,
                    "text": self.add_prefix(text, obj), 
                    "entity_list":[(sub, predicate) for sub, predicate in subjs], # 必须是list
                    "prefix_entity": obj
                })
        return new_data
        
    def _pre_process(self, path, mode):
        new_data = []
        args = self.args
        with jsonlines.open(path, 'r') as f:
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                num += 1
                text = line['text']
                spo_list = line['spo_list']
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": text,
                        "entity_list":[(spo['object'], spo['subject_type']+'-'+spo['predicate']+'-'+spo['object_type'])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
                new_data.extend(self.build_data12(line))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data

# GPFilter78D
class GPFilter78ACE05DataProcessor(object):
    def __init__(self, args):
        self.args = args
        root = args.data_dir
        self.train_path = os.path.join(root, args.train_file)
        print(self.train_path)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
        self.schema_path = os.path.join(root, '65_schemas.json')
        self._load_schema()
        self.num_labels = len(self.predicate2id.keys())
        
    def get_train_sample(self):
        return self._pre_process(self.train_path, mode='train')

    def get_dev_sample(self):
        return self._pre_process(self.dev_path, mode='dev')

    def get_test_sample(self):
        with open(self.test_path) as f:
            text_list = [json.loads(text.rstrip())["text"] for text in f.readlines()]
        return text_list
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def search_all(self, pattern, sequence):
        """从sequence中寻找所有子串pattern
        """
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i) 
        return res
    
    def _load_schema(self):
        schema = []
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                schema.append(item['subject_type']+'-'+item['predicate']+'-'+item['object_type'])
        self.schema = schema
        self.predicate2id = {v: i for i, v in enumerate(schema)}
        self.id2predicate = {i: v for i, v in enumerate(schema)}
    
    def _pre_process(self, path, mode):
        new_data = []
        with open(path, 'r', encoding='utf-8') as f:
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line in f:
                line = json.loads(line)
                for _ in range(iter_num):
                    new_data.append({
                        "text":line["text"],
                        "spo_list":[(spo["subject"], spo["subject_type"]+'-'+spo["predicate"]+'-'+spo["object_type"], spo["object"])
                                    for spo in line["spo_list"]]
                    })
        return new_data
    
# R3D
class GPNER75DataProcessor(object):
    def __init__(self, args):
        root = args.data_dir
        self.args = args
        self.train_path = os.path.join(root, args.train_file)
        self.dev_path = os.path.join(root, args.dev_file)
        self.test_path = os.path.join(root, args.test_file)
#         self.enhance_CMeEE_path = os.path.join(root, 'CMeEE_enhance_subject.json')
        print(self.train_path)
        print(self.dev_path)
        print(self.test_path)
#         if args.do_enhance_CMeEE:
#             print(self.enhance_CMeEE_path)
        self.schema_path = os.path.join(root, '53_schemas.json')
        self._load_schema()
        
        
    def get_train_sample(self):
        print('读取train数据')
        return self._pre_process(self.train_path, mode='train')
    
    def get_train_text_sample(self):
        print('读取train数据:', self.train_path)
        with jsonlines.open(self.train_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_dev_sample(self):
        print('读取dev数据')
        with jsonlines.open(self.dev_path, 'r') as f:
            data = [{'text': line['text'], 'spo_list': []} for line in f]
        return data

    def get_test_sample(self):
        print('读取test数据')
        with jsonlines.open(self.test_path, 'r') as f:
            data = [line for line in f]
        return data
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
    
    def _load_schema(self):
        # 可以作为主体的实体类型
        labels = ['其他', '其他治疗', '手术治疗', '检查', '流行病学', '疾病', '症状', '社会学', '药物', '部位', '预后']
        predicates = ['发病年龄', '鉴别诊断', '治疗后症状', '侵及周围组织转移的症状', '相关（转化）', '病理生理', '预后状况', '遗传因素',\
                          '辅助治疗', '临床表现', '相关（导致）', '病因', '发病部位', '发病机制', '手术治疗', '多发群体', '多发地区', '筛查',\
                          '并发症', '同义词', '转移部位', '病史', '化疗', '传播途径', '预后生存率', '风险评估因素', '内窥镜检查', '多发季节',\
                          '预防', '阶段', '影像学检查', '放射治疗', '组织学检查', '发病率', '发病性别倾向', '病理分型', '相关（症状）', '辅助检查',\
                          '就诊科室', '实验室检查', '药物治疗', '死亡率', '外侵部位', '高危因素']
        self.labels = labels
        self.predicates = predicates
        self.num_labels = len(labels)
        self.class2id = {v: i for i, v in enumerate(labels)}
        self.id2class = {i: v for i, v in enumerate(labels)}
        # 没有unsed0
        self.predicate2id = {v: i+1 for i, v in enumerate(predicates)}
        relations = []
        subject_predicate_dic, object_predicate_dic = defaultdict(list), defaultdict(list)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            predicate2outype = {}
            for idx, item in enumerate(f):
                item = json.loads(item.rstrip())
                subject_predicate_dic[item["subject_type"]].append(item["predicate"])
                object_predicate_dic[item["object_type"]].append(item["predicate"])
                predicate2outype[item["predicate"]] = item["object_type"]
        predicate2outype['同义词'] == '同义词'
        self.predicate2outype = predicate2outype   
        self.subject_predicate_dic = subject_predicate_dic
        self.object_predicate_dic = object_predicate_dic
        
        
    def add_entity_prefix(self, text, mode='sub'):
        if mode == 'sub':
            return f"[unused1][unused2][unused3]{text}"
        elif mode == 'obj':
            return f"[unused4][unused5][unused6]{text}"
    
    def add_xp2x_prefix(self, text, entity, predicate, entity_type=None):
        args = self.args
        if args.prefix_mode == 'entity-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            return f"{predicate}[unused1]{text[:start]}[unused2]{entity}[unused3]{text[end:]}"
        elif args.prefix_mode == 'relation-marker':
            start = self.search(text, entity)
            end = start + len(entity)
            p = self.predicate2id[predicate]
            return f"{text[:start]}[unused{p}]{entity}[unused{p}]{text[end:]}"
        elif args.prefix_mode == 'entity-entity_type-predicate':
            return f"{entity}[unused1]{entity_type}[unused2]{predicate}[unused3]{text}"
        elif args.prefix_mode == 'entity-predicate':
            return f"{entity}[unused1]{predicate}[unused2]{text}"
        elif args.prefix_mode == 'entity-out_type':
            out_type = self.predicate2outype[predicate]
            return f"{entity}[unused1]{out_type}[unused2]{text}"
        elif args.prefix_mode == 'unsed':
            return f"{entity}[unsed1]{predicate}[unsed2]{text}[unsed3]"
        
    def build_data12_sp2o(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_xp2x_prefix(text, input_entity, predicate, input_entity_type)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # S+P 抽客体
                    "text": prefix_text if args.add_prefix else text, 
                    "entity_list":[], # 必须是list
                    "prefix_entity": input_entity
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for _ in range(iter_num):
                    new_data.append(data)
        
        if args.add_wrong_entity_neg_sample:
            entity2type_dic = {}
            for spo in line['entity_list']:
                entity2type_dic[spo['entity']] = spo['entity_type']
            gold_entities = set([spo['subject'] for spo in line['spo_list']])
            pred_entities = set([spo['entity'] for spo in line['entity_list']])
            wrong_entities = list(pred_entities-gold_entities)
#             print('wrong_entities:',wrong_entities)
            for input_entity in wrong_entities:
                entity_type = entity2type_dic[input_entity]
                for predicate in entity2predicate_dic[entity_type]:
                    if random.random() < args.wrong_entity_neg_sample_rate:
                        prefix_text = self.add_xp2x_prefix(text, input_entity, predicate, entity_type)
                        data = {
                            "type": data_type, # O+P 抽主体
                            "text": prefix_text, 
                            "entity_list":[], # 必须是list
                            "prefix_entity": input_entity,
                            "neg_type": 'wrong_entity'
                        }
                        for i in range(iter_num):
                            new_data.append(data)
        
        return new_data
    
    
    def build_data12_op2s(self, line, entity2predicate_dic, data_type=1, keep_rate=1, mode='train'):
        args = self.args
        text = line['text']
        spo_list = line['spo_list']
        new_data = []
        input_entity_types = []
        positive_dic = defaultdict(list)
        for spo in spo_list:
            if data_type == 1:
                positive_dic[f"{spo['subject']}{spo['predicate']}"].append((spo['object']['@value'], spo['object_type']['@value']))
                input_entity_types.append((spo["subject_type"], spo["subject"]))
            else:
                positive_dic[f"{spo['object']['@value']}{spo['predicate']}"].append((spo['subject'], spo['subject_type']))
                input_entity_types.append((spo["object_type"]['@value'], spo['object']['@value']))
        input_entity_types = list(set(input_entity_types))
        prefix2data_dic = {}
        for input_entity_type, input_entity in input_entity_types:
            for predicate in entity2predicate_dic[input_entity_type]:
                # 1：S+P抽O，2：O+P抽S
                # f"{input_entity}[unsed1]{predicate}[unsed2]{text}",
                prefix_text = self.add_xp2x_prefix(text, input_entity, predicate, input_entity_type)
                prefix2data_dic[f"{input_entity}{predicate}"] = {
                    "type": data_type, # O+P 抽主体
                    "text": prefix_text if args.add_prefix else text, 
                    "entity_list":[], # 必须是list
                    "prefix_entity": input_entity
                }
                if f"{input_entity}{predicate}" in positive_dic.keys():
                    prefix2data_dic[f"{input_entity}{predicate}"]["entity_list"] = positive_dic[f"{input_entity}{predicate}"]
        
        iter_num = 2 if self.args.do_rdrop else 1
        for data in prefix2data_dic.values():
            if (data["entity_list"] == [] and random.random() < keep_rate) or data["entity_list"] != []:
                for i in range(iter_num):
                    new_data.append(data)
        
        if args.add_wrong_entity_neg_sample:
            entity2type_dic = {}
            for spo in line['entity_list']:
                entity2type_dic[spo['entity']] = spo['entity_type']
            gold_entities = set([spo['object']['@value'] for spo in line['spo_list']])
            pred_entities = set([spo['entity'] for spo in line['entity_list']])
            wrong_entities = list(pred_entities-gold_entities)
#             print('wrong_entities:',wrong_entities)
            for input_entity in wrong_entities:
                entity_type = entity2type_dic[input_entity]
                for predicate in entity2predicate_dic[entity_type]:
                    if random.random() < args.wrong_entity_neg_sample_rate:
                        prefix_text = self.add_xp2x_prefix(text, input_entity, predicate, entity_type)
                        data = {
                            "type": data_type, # O+P 抽主体
                            "text": prefix_text, 
                            "entity_list":[], # 必须是list
                            "prefix_entity": input_entity,
                            "neg_type": 'wrong_entity'
                        }
                        for i in range(iter_num):
                            new_data.append(data)
        
        return new_data
    
    def _pre_process(self, path, mode):
        args = self.args
        new_data = []
        with jsonlines.open(path, 'r') as f:
            if args.add_wrong_entity_neg_sample:
                with jsonlines.open('./result_output/gpner3/entity_list_train.jsonl') as r:
                    f1 = [line for line in r]
            else:
                f =  [i for i in f]
                f1 = [0 for i in f] # 假数据，方便后续zip不报错
            num = 0
            iter_num = 2 if self.args.do_rdrop and mode == 'train' else 1
            for line, line1 in zip(f, f1):
                if args.add_wrong_entity_neg_sample:
                    line['entity_list'] = line1['entity_list']
                num += 1
#                 line = json.loads(line)
                text = line['text']
                spo_list = line['spo_list']
                for _ in range(iter_num):
                    new_data.append({
                        "type": 0, # 抽主体
                        "text": self.add_entity_prefix(text, 'sub'),
                        "entity_list": [(spo["subject"], spo["subject_type"])
                                    for spo in spo_list]
                    })
                for i in range (iter_num):
                    new_data.append({
                        "type": 0, # 抽客体
                        "text": self.add_entity_prefix(text, 'obj'),
                        "entity_list":[(spo["object"]["@value"], spo["object_type"]["@value"])
                                    for spo in spo_list],
                        "prefix_entity": ''
                    })
                new_data.extend(self.build_data12_sp2o(line, self.subject_predicate_dic, 1, self.args.negative_samples_sp2o_rate, mode))
                new_data.extend(self.build_data12_op2s(line, self.object_predicate_dic, 2, self.args.negative_samples_op2s_rate, mode))
#                 if num == 5:
#                     for data in new_data:
#                         print(data,'\n')
        return new_data