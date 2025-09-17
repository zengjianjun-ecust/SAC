

import jsonlines
import time
import sys
sys.path.append('./codes')
from utils import init_logger, get_time
import re

import sac0gen_syn



def annotate(l_check_text):
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join("./temp", current_time)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "CMeIE-V2_test.jsonl")
    
    # 保存给 PLM 预测。 ，一条一条来
    

    # 将数据保存为JSONL格式（每行一个JSON对象）
    with open(file_path, 'w', encoding='utf-8') as f:
        for check_text in  l_check_text:
            json_line = json.dumps({"sentence": check_text}, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"文件保存成功: {file_path}")
    
    PLM.predict(folder_path)
    
    # 它是一个只有一行的jsonl，因此可以用json.load去读
    # 结果文件
    file_path = os.path.join(folder_path, 'output', 'CMeIE-V2_test.jsonl')
    
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
PLM = sac0gen_syn.singleton_main()

# set this after the first state traininig 
PLM.init_model('09-14-13-38')

mode = 'syn'

# 初始化列表来存储处理后的数据
fine_data = []

# 打开并读取 JSONL 文件
with open(f'./{mode}/CMeIE-V2_train_fine_file.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        # 解析每一行的 JSON 数据
        data = json.loads(line.strip())

        
l_check_text = []
l_check_text.extend(fine_data['text_list'])

file_path = annotate(l_check_text)


print("generate and annotate syn file: ", file_path)

# you may need to merge file_path and original train set. or perform a stage fine-tuning

# !cp {out_file} train_and_{out_file} 
# !cat train.json >>train_and_{out_file} 
# !wc -l *.json

