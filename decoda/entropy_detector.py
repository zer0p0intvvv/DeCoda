import os
import math
import re
import json
import argparse
from collections import Counter
from tqdm import tqdm

def calculate_entropy(text):
    """
    计算字符串的熵值
    熵越高，表示字符串随机性越高
    """
    if not text:
        return 0
    
    # 统计字符频率
    counter = Counter(text)
    total_chars = len(text)
    
    # 计算香农熵
    entropy = 0
    for count in counter.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
    
    return entropy

def calculate_average_line_length(code):
    """计算代码的平均行长度"""
    lines = [line for line in code.split('\n') if line.strip()]
    if not lines:
        return 0
    return sum(len(line) for line in lines) / len(lines)

def calculate_var_name_features(code):
    """分析代码中的变量名特征"""
    # 提取可能的变量名（简化版，实际中可能需要更复杂的解析）
    var_pattern = r'var\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
    let_pattern = r'let\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
    const_pattern = r'const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
    param_pattern = r'function\s*(?:[a-zA-Z_$][a-zA-Z0-9_$]*)?\s*\(([^)]*)\)'
    
    var_names = re.findall(var_pattern, code)
    var_names.extend(re.findall(let_pattern, code))
    var_names.extend(re.findall(const_pattern, code))
    
    # 处理函数参数
    param_lists = re.findall(param_pattern, code)
    for param_list in param_lists:
        params = [p.strip() for p in param_list.split(',') if p.strip()]
        var_names.extend(params)
    
    if not var_names:
        return 0, 0
    
    # 计算变量名平均长度
    avg_name_length = sum(len(name) for name in var_names) / len(var_names)
    
    # 计算短变量名(<=2字符)的比例
    short_var_ratio = sum(1 for name in var_names if len(name) <= 2) / len(var_names)
    
    return avg_name_length, short_var_ratio

def count_suspicious_patterns(code):
    """计算可疑模式的出现次数"""
    suspicious_patterns = [
        r'eval\s*\(', 
        r'Function\s*\(', 
        r'\\x[0-9a-fA-F]{2}',  # 十六进制转义
        r'\\u[0-9a-fA-F]{4}',  # Unicode转义
        r'fromCharCode',
        r'decodeURIComponent',
        r'atob\s*\(',
        r'String\.fromCharCode',
        r'unescape\s*\(',
        r'parseInt\s*\(.+,.+\)'  # 特定基数的parseInt
    ]
    
    count = 0
    for pattern in suspicious_patterns:
        count += len(re.findall(pattern, code))
    
    return count

def detect_minification(code):
    """检测代码是否被最小化（压缩）"""
    # 最小化代码通常有很长的行和很少的换行符
    lines = code.split('\n')
    if len(lines) <= 1:
        return True
    
    long_line_ratio = sum(1 for line in lines if len(line) > 100) / len(lines)
    if long_line_ratio > 0.5:
        return True
    
    # 检测分号后面立即跟随其他字符的比例
    semicolons = re.findall(r';[^\s\n]', code)
    if len(semicolons) > 10:
        return True
    
    return False

def analyze_file(file_path):
    """分析文件并返回各种特征"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()
        
        # 计算熵值
        entropy = calculate_entropy(code)
        
        # 计算平均行长度
        avg_line_length = calculate_average_line_length(code)
        
        # 变量名分析
        avg_var_length, short_var_ratio = calculate_var_name_features(code)
        
        # 可疑模式计数
        suspicious_count = count_suspicious_patterns(code)
        
        # 检测最小化
        is_minified = detect_minification(code)
        
        return {
            "file_path": file_path,
            "entropy": entropy,
            "avg_line_length": avg_line_length,
            "avg_var_length": avg_var_length,
            "short_var_ratio": short_var_ratio,
            "suspicious_count": suspicious_count,
            "is_minified": is_minified
        }
    except Exception as e:
        return {
            "file_path": file_path,
            "error": str(e)
        }

def is_obfuscated(features, entropy_threshold=4.5):
    """
    根据特征判断代码是否被混淆
    
    判断标准:
    1. 熵值超过阈值
    2. 代码被最小化
    3. 短变量名比例高
    4. 存在可疑模式
    """
    score = 0
    
    # 熵值判断 (0-2分)
    if features["entropy"] > entropy_threshold:
        score += 2
    elif features["entropy"] > entropy_threshold - 0.5:
        score += 1
    
    # 最小化判断 (0-1分)
    if features["is_minified"]:
        score += 1
    
    # 变量名判断 (0-2分)
    if features["short_var_ratio"] > 0.6:
        score += 2
    elif features["short_var_ratio"] > 0.4:
        score += 1
    
    # 平均行长度判断 (0-2分)
    if features["avg_line_length"] > 200:
        score += 2
    elif features["avg_line_length"] > 100:
        score += 1
    
    # 可疑模式判断 (0-2分)
    if features["suspicious_count"] > 5:
        score += 2
    elif features["suspicious_count"] > 2:
        score += 1
    
    # 总分超过4分判定为混淆代码
    return score >= 7, score, {
        "min_score": 0,
        "max_score": 9,
        "obfuscation_threshold": 4,
    }

def process_directory(directory, output_file=None, verbose=False):
    """处理目录中的所有JavaScript文件"""
    results = []
    js_files = []
    
    # 查找所有JavaScript文件
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.js') or file.endswith('.txt'):
                js_files.append(os.path.join(root, file))
    
    print(f"在 {directory} 中找到 {len(js_files)} 个文件")
    
    # 分析每个文件
    obfuscated_count = 0
    for file_path in tqdm(js_files, desc=f"分析文件"):
        features = analyze_file(file_path)
        
        if "error" in features:
            if verbose:
                print(f"处理文件出错: {file_path} - {features['error']}")
            continue
        
        obfuscated, score, _ = is_obfuscated(features)
        features["obfuscated"] = obfuscated
        features["score"] = score
        
        if obfuscated:
            obfuscated_count += 1
            if verbose:
                print(f"检测到混淆代码: {file_path} (得分: {score}/9)")
        
        results.append(features)
    
    # 输出摘要
    print(f"\n分析结果摘要:")
    print(f"总文件数: {len(results)}")
    print(f"混淆文件数: {obfuscated_count} ({obfuscated_count/len(results)*100:.2f}%)")
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_files": len(results),
                    "obfuscated_files": obfuscated_count,
                    "obfuscation_ratio": obfuscated_count/len(results) if results else 0
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"详细结果已保存到: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='使用熵值和其他特征检测JavaScript代码是否被混淆')
    parser.add_argument('directory', help='要分析的目录路径')
    parser.add_argument('--output', '-o', help='输出结果的JSON文件路径')
    parser.add_argument('--entropy', '-e', type=float, default=4.5, help='熵值阈值 (默认: 4.5)')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    process_directory(args.directory, args.output, args.verbose)

if __name__ == "__main__":
    main() 