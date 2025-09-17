# SAC Project Setup Guide

## 1. Environment Setup

### Create Conda Environment
```bash
conda create --name SAC python=3.10
conda activate SAC
```

### Verify Python and CUDA Versions
- **Python**: 3.10.9
- **PyTorch**: 2.5.1+cu124
- **CUDA Toolkit**: 12.4

### Install Dependencies
```bash
pip install -r pip-requirements.txt
conda install --file conda-requirements.txt
```

## 2. Model Installation

### Download Pre-trained Models

#### RoBERTa_zh_Large_Pytorch
1. Download from [Baidu Pan](https://pan.baidu.com/s/1XJYWN87DbMmECGvIP85sWg?pwd=1cqw)
2. Expected file structure:
```
./pretrained_model/RoBERTa_zh_Large_PyTorch/
    config.json
    pytorch_model.bin
    vocab.txt
```

#### ALBERT-xxlarge-v1
1. Download from [Hugging Face](https://huggingface.co/albert/albert-xxlarge-v1)
2. Expected file structure:
```
./pretrained_model/albert-xxlarge-v1/
    config.json
    eval.log
    pytorch_model.bin
    spiece.model
    tokenizer.json
```

## 3. Dataset Preparation

### ACE05 Dataset
1. Download ACE2005 dataset from [DyGIE repository](https://github.com/luanyi/DyGIE/tree/master/preprocessing/ace2005)
2. Convert to JSON format:
```bash
python ace2json.py
```
3. Process the data:
```bash
python preprocess_data.py
```
4. Expected processed data structure:
```
./ACE05-DyGIE/processed_data/
    65_schemas.json
    dev.json
    test.json
    train.json
```

### CMeIE Dataset
1. Download CMeIE-V2 dataset from [Tianchi](https://tianchi.aliyun.com/dataset/95414/submission)
2. Expected file structure:
```
./data_dir/
    CMeIE-V2_train.jsonl
    CMeIE-V2_dev.jsonl
    CMeIE-V2_test.jsonl
    # Additional generated files
```

## 4. Execution Pipeline

### Stage 0: Initialize PLM Annotator
```bash
cd SAC
export CUDA_VISIBLE_DEVICES="3"
export train_file="CMeIE-V2_train"
export data_dir="./data_dir"
py_exec="/opt/conda/envs/SAC/bin/python"

${py_exec} run_gpner41-CMeIE.py
```

**Note**: After execution, note the timestamp from logs (e.g., `09-14-13-34` found in `/checkpoint/gpner41/RoBERTa_zh_Large_PyTorch/gpner41/09-14-13-34`)

### Stage 1: Generate Synthetic Data
1. Update the timestamp in both `sac1feedback.py` and `sac2annotate.py`:
```python
PLM.init_model('09-14-13-38')  # Replace with your actual timestamp
```

2. Execute synthetic data generation:
```bash
python sac0gen_syn.py
python sac2annotate.py
```

3. Merge generated data with original training data:
```bash
cp {out_file} ./data_dir/train_and_{out_file}
cat ./data_dir/CMeIE-V2_train.jsonl >> ./data_dir/train_and_{out_file}
wc -l *.json  # Verify line counts
```

### Stage 2: Final Training
```bash
cd SAC
export CUDA_VISIBLE_DEVICES="3"
export train_file="CMeIE-V2_train"
export data_dir="./data_dir"
py_exec="/opt/conda/envs/SAC/bin/python"

${py_exec} run_gpner41-CMeIE.py
```

## Important Notes

1. **File Path Consistency**: Ensure all file paths in scripts match your actual directory structure
2. **Timestamp Replacement**: Always replace timestamp placeholders with actual values from your runs
3. **Memory Management**: Monitor GPU memory usage, especially when working with large models
4. **Data Verification**: Validate processed data files before proceeding to next stages
5. **Error Handling**: Check log files carefully at each stage for potential issues

## Troubleshooting

- If encountering CUDA out-of-memory errors, reduce batch size or use gradient accumulation
- For dataset processing issues, verify the original data formats match expected structures
- Ensure all dependencies are compatible with the specified Python and CUDA versions
