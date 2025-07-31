import os
import json
import time
import requests
from openai import OpenAI
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import random

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-d7c871360ad442e48ea6787e34b0e448"  # 替换为你的DeepSeek API密钥
DEEPSEEK_API_URL = "https://api.deepseek.com"  # 更新为不带v1的基础URL

# 输入输出配置
INPUT_DIR_BAD = "original_js_output_bad"  # 原始JavaScript文件目录 (坏样本)
OUTPUT_DIR_BAD = "deobfuscated_js_bad"  # 去混淆后的JavaScript文件保存目录 (坏样本)
# INPUT_DIR_GOOD = "original_js_output_good"  # 原始JavaScript文件目录 (好样本) - 注释掉
# OUTPUT_DIR_GOOD = "deobfuscated_js_good"  # 去混淆后的JavaScript文件保存目录 (好样本) - 注释掉

# 处理标记
PROCESSED_FILE = "processed_files.json"  # 记录已处理文件的JSON

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    os.makedirs(directory, exist_ok=True)

def load_processed_files():
    """加载已处理文件记录"""
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "processed": [],
        "failed": [],
        "stats": {
            "total_processed": 0,
            "successful": 0,
            "failed": 0
        }
    }

def save_processed_files(data):
    """保存已处理文件记录"""
    with open(PROCESSED_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def call_deepseek_api(code, max_retries=3, delay=2):
    """
    使用OpenAI SDK调用DeepSeek API进行代码去混淆
    :param code: 原始JavaScript代码
    :param max_retries: 最大重试次数
    :param delay: 重试延迟(秒)
    :return: (success, response)
    """
    # 初始化客户端
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_URL
    )
    
    # 准备消息内容
    system_message = "你是一个JavaScript代码专家，擅长分析和去混淆JavaScript代码。"
    user_message = f"""
请分析以下JavaScript代码:

```javascript
{code}
```

1. 判断这段代码是否混淆过。
2. 如果是混淆过的代码，请将其去混淆，保持功能完全一致，但使代码更加可读。
3. 如果不是混淆过的代码，返回原代码并说明'代码未混淆'。
4. 你的回复应该首先说明分析结果，然后是去混淆后的代码（如果需要）。
5. 去混淆后的代码应包含适当的注释，解释代码的功能。

只返回去混淆的结果和简单说明，不要进行详细解释。
"""
    
    for attempt in range(max_retries):
        try:
            # 使用OpenAI SDK发送请求
            response = client.chat.completions.create(
                model="deepseek-reasoner", # 或其他适用的模型ID
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,  # 低温度使输出更确定性
                max_tokens=8000   # 调整根据需要
            )
            
            return True, response
            
        except Exception as e:
            error_message = str(e)
            logging.error(f"API请求异常: {error_message}")
            
            # 处理速率限制错误
            if "rate_limit" in error_message.lower() or "429" in error_message:
                logging.warning(f"API速率限制，等待{delay}秒后重试...")
                time.sleep(delay)
            elif attempt < max_retries - 1:
                time.sleep(delay)
            else:
                return False, f"API请求异常: {error_message}"
    
    return False, "超过最大重试次数"

def extract_code_from_response(response):
    """
    从API响应中提取去混淆后的代码
    :param response: API响应
    :return: (是否混淆, 去混淆后的代码)
    """
    try:
        # 使用OpenAI SDK的响应格式
        content = response.choices[0].message.content
        
        # 判断代码是否混淆
        is_obfuscated = "代码未混淆" not in content
        
        # 提取代码块
        code_blocks = []
        in_code_block = False
        code_lines = []
        
        for line in content.split('\n'):
            if line.strip().startswith('```'):
                if in_code_block:
                    in_code_block = False
                    if code_lines:
                        code_blocks.append('\n'.join(code_lines))
                        code_lines = []
                else:
                    in_code_block = True
                    # 忽略语言标识符
                    continue  
            elif in_code_block:
                code_lines.append(line)
        
        # 如果有代码块，则返回第一个代码块
        if code_blocks:
            return is_obfuscated, code_blocks[0]
        
        return is_obfuscated, content
    except Exception as e:
        logging.error(f"解析响应失败: {str(e)}")
        return False, f"解析响应失败: {str(e)}"

def process_file(file_path, output_dir, processed_data):
    """
    处理单个JavaScript文件
    :param file_path: JavaScript文件路径
    :param output_dir: 输出目录
    :param processed_data: 已处理文件记录
    :return: 是否成功处理
    """
    file_path_str = str(file_path)
    
    # 检查是否已处理
    if file_path_str in processed_data["processed"]:
        logging.info(f"文件已处理，跳过: {file_path_str}")
        return True
    
    try:
        # 读取JavaScript文件
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()
        
        # 调用API
        success, response = call_deepseek_api(code)
        
        if not success:
            logging.error(f"处理文件失败: {file_path_str} - {response}")
            processed_data["failed"].append(file_path_str)
            processed_data["stats"]["failed"] += 1
            save_processed_files(processed_data)
            return False
        
        # 解析响应
        is_obfuscated, deobfuscated_code = extract_code_from_response(response)
        
        if deobfuscated_code:
            # 构建输出文件路径
            # 确定相对路径时需要使用正确的输入目录
            relative_input_dir = INPUT_DIR_BAD
                
            relative_path = os.path.relpath(file_path, relative_input_dir)
            output_file_path = os.path.join(output_dir, relative_path)
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # 保存去混淆代码
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(deobfuscated_code)
            
            # 保存元数据
            meta_file_path = output_file_path + ".meta.json"
            with open(meta_file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "original_file": file_path_str,
                    "is_obfuscated": is_obfuscated,
                    "process_time": time.time(),
                    "model": "deepseek-r1"
                }, f, indent=2, ensure_ascii=False)
            
            # 更新处理记录
            processed_data["processed"].append(file_path_str)
            processed_data["stats"]["total_processed"] += 1
            processed_data["stats"]["successful"] += 1
            save_processed_files(processed_data)
            
            return True
        else:
            logging.error(f"解析响应失败: {file_path_str}")
            processed_data["failed"].append(file_path_str)
            processed_data["stats"]["failed"] += 1
            save_processed_files(processed_data)
            return False
            
    except Exception as e:
        logging.error(f"处理文件异常: {file_path_str} - {str(e)}")
        processed_data["failed"].append(file_path_str)
        processed_data["stats"]["failed"] += 1
        save_processed_files(processed_data)
        return False

def load_analysis_files():
    """
    加载所有分析文件并筛选评分大于等于7的文件
    :return: 筛选后的文件列表，按类别和数据集分组
    """
    analysis_files = {
        "bad": {
            "train": "obfuscation_analysis_bad_train.json",
            "test": "obfuscation_analysis_bad_test.json",
            "val": "obfuscation_analysis_bad_val.json"
        }
    }
    
    filtered_files = {
        "bad": {"train": [], "test": [], "val": []}
    }
    
    # 用于去重的文件路径集合
    unique_files = set()
    
    # 加载和筛选文件
    category = "bad"
    for dataset in ["train", "test", "val"]:
        file_path = analysis_files[category][dataset]
        if file_path and os.path.exists(file_path):
            logging.info(f"加载分析文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # 筛选评分大于等于7的文件
            if "results" in analysis_data:
                for result in analysis_data["results"]:
                    if result.get("score", 0) >= 7:
                        file_path = result["file_path"]
                        # 检查文件是否已经添加过（去重）
                        if file_path not in unique_files:
                            unique_files.add(file_path)
                            filtered_files[category][dataset].append(file_path)
                
                logging.info(f"{category}_{dataset} 筛选出 {len(filtered_files[category][dataset])} 个文件")
    
    total_files = sum(len(files) for files in filtered_files["bad"].values())
    logging.info(f"总共筛选出 {total_files} 个唯一文件")
    
    return filtered_files

def sample_files(filtered_files, train_count=None, test_count=None, val_count=None):
    """
    获取所有筛选后的文件，不限制数量
    :param filtered_files: 按类别和数据集分组的文件列表
    :param train_count: 训练集数量限制（若为None则使用全部）
    :param test_count: 测试集数量限制（若为None则使用全部）
    :param val_count: 验证集数量限制（若为None则使用全部）
    :return: 筛选后的文件列表
    """
    sampled_files = {
        "bad": {"train": [], "test": [], "val": []}
    }
    
    category = "bad"
    # 对每个数据集进行处理
    for dataset in ["train", "test", "val"]:
        available_files = filtered_files[category][dataset]
        count = None
        if dataset == "train":
            count = train_count
        elif dataset == "test":
            count = test_count
        elif dataset == "val":
            count = val_count
            
        if available_files:
            if count is not None and len(available_files) > count:
                # 随机抽样指定数量
                sampled_files[category][dataset] = random.sample(available_files, count)
                logging.info(f"{category}_{dataset} 随机抽样 {count} 个文件")
            else:
                # 使用全部文件
                sampled_files[category][dataset] = available_files
                logging.info(f"{category}_{dataset} 使用全部 {len(available_files)} 个文件")
        else:
            logging.warning(f"{category}_{dataset} 没有符合条件的文件")
    
    # 计算总文件数
    total_files = sum(len(files) for files in sampled_files["bad"].values())
    logging.info(f"总共将处理 {total_files} 个文件")
    
    return sampled_files

def process_sampled_files(sampled_files):
    """
    处理抽样文件
    :param sampled_files: 抽样后的文件列表
    """
    processed_data = load_processed_files()
    
    # 处理所有抽样文件
    total_files = 0
    processed_files = 0
    
    category = "bad"
    # 设置正确的输出目录
    output_dir = OUTPUT_DIR_BAD
    ensure_dir(output_dir)
    
    for dataset in ["train", "test", "val"]:
        files = sampled_files[category][dataset]
        if not files:
            continue
        
        dataset_output_dir = os.path.join(output_dir, dataset)
        ensure_dir(dataset_output_dir)
        
        logging.info(f"处理 {category}_{dataset} 数据集中的 {len(files)} 个文件")
        
        for file_path in tqdm(files, desc=f"处理 {category}_{dataset}"):
            total_files += 1
            if process_file(file_path, dataset_output_dir, processed_data):
                processed_files += 1
    
    logging.info(f"所有数据集处理完成: 成功 {processed_files}/{total_files}")

def process_directory(dataset_dir="train", input_dir=None, output_dir=None):
    """
    处理指定数据集目录下的所有JavaScript文件
    :param dataset_dir: 数据集目录名(train/val/test)
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    """
    if input_dir is None:
        input_dir = INPUT_DIR_BAD
        
    if output_dir is None:
        output_dir = OUTPUT_DIR_BAD
        
    input_dataset_dir = os.path.join(input_dir, dataset_dir)
    output_dataset_dir = os.path.join(output_dir, dataset_dir)
    
    # 确保输出目录存在
    ensure_dir(output_dataset_dir)
    
    # 加载已处理文件记录
    processed_data = load_processed_files()
    
    # 获取所有JS文件
    js_files = []
    for root, _, files in os.walk(input_dataset_dir):
        for file in files:
            if file.endswith('.js') or '.' not in file:  # 包含JS文件和没有扩展名的文件
                js_files.append(os.path.join(root, file))
    
    logging.info(f"在 {input_dataset_dir} 中找到 {len(js_files)} 个JavaScript文件")
    
    # 处理文件
    success_count = 0
    for file_path in tqdm(js_files, desc=f"处理{dataset_dir}集"):
        if process_file(file_path, output_dataset_dir, processed_data):
            success_count += 1
    
    logging.info(f"{dataset_dir}集处理完成: 成功 {success_count}/{len(js_files)}")
    
    return success_count, len(js_files)

def main():
    """主函数"""
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='使用DeepSeek R1去混淆JavaScript代码')
    parser.add_argument('--dataset', choices=['train', 'val', 'test', 'all', 'sampled'], default='sampled',
                      help='要处理的数据集目录 (默认: sampled - 根据评分抽样)')
    parser.add_argument('--api-key', type=str, help='DeepSeek API密钥')
    parser.add_argument('--api-url', type=str, help='DeepSeek API URL')
    parser.add_argument('--input-dir-bad', type=str, help='坏样本输入目录')
    parser.add_argument('--output-dir-bad', type=str, help='坏样本输出目录')
    parser.add_argument('--train-count', type=int, help='训练集抽样数量限制 (默认: 不限制)')
    parser.add_argument('--test-count', type=int, help='测试集抽样数量限制 (默认: 不限制)')
    parser.add_argument('--val-count', type=int, help='验证集抽样数量限制 (默认: 不限制)')
    args = parser.parse_args()
    
    # 更新配置
    global DEEPSEEK_API_KEY, DEEPSEEK_API_URL, INPUT_DIR_BAD, OUTPUT_DIR_BAD
    if args.api_key:
        DEEPSEEK_API_KEY = args.api_key
    if args.api_url:
        DEEPSEEK_API_URL = args.api_url
    if args.input_dir_bad:
        INPUT_DIR_BAD = args.input_dir_bad
    if args.output_dir_bad:
        OUTPUT_DIR_BAD = args.output_dir_bad
    
    # 确保输出目录存在
    ensure_dir(OUTPUT_DIR_BAD)
    
    # 处理指定数据集
    if args.dataset == 'sampled':
        # 使用基于评分的抽样方法
        logging.info("使用基于评分的抽样方法处理[坏样本]文件 (评分 >= 7)")
        filtered_files = load_analysis_files()
        sampled_files = sample_files(
            filtered_files, 
            train_count=args.train_count, 
            test_count=args.test_count, 
            val_count=args.val_count
        )
        process_sampled_files(sampled_files)
    elif args.dataset == 'all':
        # 处理所有数据集
        total_success = 0
        total_files = 0
        
        for dataset in ['train', 'val', 'test']:
            if os.path.exists(os.path.join(INPUT_DIR_BAD, dataset)):
                success, total = process_directory(dataset, INPUT_DIR_BAD, OUTPUT_DIR_BAD)
                total_success += success
                total_files += total
        
        logging.info(f"bad 数据集处理完成: 成功 {total_success}/{total_files}")
    else:
        # 处理特定数据集
        process_directory(args.dataset, INPUT_DIR_BAD, OUTPUT_DIR_BAD)

if __name__ == "__main__":
    main() 