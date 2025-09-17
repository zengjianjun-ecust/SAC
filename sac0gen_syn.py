#!/usr/bin/env python
# coding: utf-8

# In[18]:


from openai import OpenAI
import jsonlines
import time
import sys
sys.path.append('./codes')
from utils import init_logger, get_time
client = OpenAI(api_key="", base_url="")
import re
mode = 'syn'
pattern = r'<sentence>(.*?)</sentence>'
with open(f'./{mode}/checkpoint.txt', 'r') as f:
    crt_idx = int(f.read())

import sac0gen_syn

# In[19]:


def rec_openai(params):
    try:
        return client.chat.completions.create(**params)
    except:
#         print('超速，休息10秒')
        logger.info('超速，休息10秒')
        time.sleep(10)
        return rec_openai(params)


# In[20]:


def get_entities(gold_data):
    entities = set()
    for spo in gold_data['spo_list']:
        sub = spo["subject"]
        obj = spo["object"]["@value"]
        entities.add(sub)
        entities.add(obj)
    return list(entities)


# In[21]:
import os
from datetime import datetime

# 获取当前时间并格式化为字符串



def check_is_fine(gold_data, check_text):
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join("./temp", current_time)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "CMeIE-V2_test.jsonl")
    
    # 保存给 PLM 预测。 ，一条一条来
    

    # 将数据保存为JSONL格式（每行一个JSON对象）
    with open(file_path, 'w', encoding='utf-8') as f:
        json_line = json.dumps({"sentence": check_text}, ensure_ascii=False)
        f.write(json_line + '\n')
    
    print(f"文件保存成功: {file_path}")
    
    PLM.predict(folder_path)
    
    # 它是一个只有一行的jsonl，因此可以用json.load去读
    # 结果文件
    file_path = os.path.join(folder_path, 'output', 'CMeIE-V2_test.jsonl')
    
    # 对比 gold_data 和 data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 因为只有一行，可以直接用json.load
        
    entities = get_entities(gold_data)
    pred_entities = get_entities(data)
    
    lack_entities = list(set(entities) - set(pred_entities))
    return lack_entities == [], lack_entities
    
    
    

# In[22]:


def gen_data(rough_output_file, fine_output_file, rough_all_output_file, params):
    input_file = './data/CMeIE-V2_train_4000.jsonl'
    with jsonlines.open(input_file, mode='r') as r:
        for i, gold_data in enumerate(r):
            if i <= crt_idx and crt_idx!=0:
                continue
#             print(f'正在处理第{i+1}条数据')
            logger.info('###########################################')
            logger.info(f'正在处理第{i+1}条数据')
            gold_data = fix_data(gold_data)
            entity_spos = []
            for spo in gold_data['spo_list']:
                sub = spo['subject']
                obj = spo["object"]["@value"]
                predicate = spo['predicate']
                entity_spos.append(f'{obj}是{sub}的{predicate}')
            str_entity_spos = '、'.join(entity_spos)
            text = gold_data['text']
            entities = get_entities(gold_data)
            str_entities = '、'.join(entities)
            prompt1 = f'请将接下来的句子生成8条同义句，但相互之间不能完全一样，且字数和原句相仿。句子：{text}注意：生成的所有句子必须显式包含以下词：{str_entities}，并且这些词需要满足以下关系：{str_entity_spos}，每条句子都必须用<sentence></sentence>标签包裹起来。'
            logger.info(f'prompt: {prompt1}')
            mess = [{"role": "user", "content": prompt1}]
            params['messages'] = mess

            response = rec_openai(params)
            generated_text = response.choices[0].message.content
            logger.info(f'gen: {generated_text}')
            generated_text_list = re.findall(pattern, generated_text)   #！
            while len(generated_text_list) < 8:
                response = rec_openai(params)
                generated_text = response.choices[0].message.content
                logger.info(f'regen: {generated_text}')
                generated_text_list = re.findall(pattern, generated_text)
            generated_text_list = generated_text_list[:8]
            rough_data = {'id': i+1, 'text_list': [g for g in generated_text_list]}
            rough_data_all = {'id': i+1, 'text_list': [g for g in generated_text_list]}
            fine_data = {'id': i+1, 'text_list': [], 'rough_ids': []}
            
            with jsonlines.open(rough_output_file, mode='a') as w_r:
                w_r.write(rough_data)
                w_r._fp.flush()
            
            for j, gen_text in enumerate(generated_text_list):
                flag, lack_entities = check_is_fine(gold_data, gen_text)
                if flag:
                    fine_data['text_list'].append(gen_text)
                    logger.info(f'{j+1}:find_data添加一条一次数据')
                else:
                    fine_gen_text, fine_num = refine(gold_data, lack_entities, gen_text, rough_data_all, params, 1)
                    fine_data['text_list'].append(fine_gen_text)
                    if fine_num == 4:
                        fine_data['rough_ids'].append(len(fine_data['text_list'])-1)    # 没有反思改写成功的
                    logger.info(f'{j+1}:find_data添加一条fine数据')
                    
            with jsonlines.open(fine_output_file, mode='a') as w_f:
                w_f.write(fine_data)
                w_f._fp.flush()
                
            with jsonlines.open(rough_all_output_file, mode='a') as w_ra:
                w_ra.write(rough_data_all)
                w_ra._fp.flush()
            
            with open(f'./{mode}/checkpoint.txt', 'w') as f:
                f.write(str(i))


# In[25]:


os.environ["CUDA_VISIBLE_DEVICES"] = '3'
PLM = sac0gen_syn.singleton_main()

# set this after the first state traininig 
PLM.init_model('09-14-13-38')




cur_time = get_time(fmt='%m-%d-%H-%M')
logger = init_logger(f'./{mode}/log_{cur_time}.txt')
params = {
    "model": "qwen2.5",
    "messages": [],
    "temperature": 0.5,
    "extra_body":{
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
    }
}
rough_output_file = f'./{mode}/CMeIE-V2_train_rough_file.jsonl'
fine_output_file = f'./{mode}/CMeIE-V2_train_fine_file.jsonl'
rough_all_output_file = f'./{mode}/CMeIE-V2_train_rough_all_file.jsonl'
gen_data(rough_output_file, fine_output_file, rough_all_output_file, params)
with open(f'./{mode}/checkpoint.txt', 'w') as f:
    f.write(str(0))


# In[ ]:




