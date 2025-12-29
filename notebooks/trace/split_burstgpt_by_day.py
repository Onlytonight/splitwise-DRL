import csv
import os
from collections import defaultdict

# 配置
input_file = 'data/BurstGPT_without_fails_1.csv'
output_dir = 'traces/burst'
SECONDS_PER_DAY = 86400  # 一天86400秒
MAPPED_DAY_LENGTH = 60  # 映射后一天的长度（秒）

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 存储按天分组的数据
daily_requests = defaultdict(list)

print("正在读取数据...")
# 读取并分组数据
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        timestamp = float(row['Timestamp'])
        request_tokens = int(row['Request tokens'])
        response_tokens = int(row['Response tokens'])
        
        # 计算属于第几天（从0开始）
        day = int(timestamp // SECONDS_PER_DAY)
        # 计算该天内的相对时间（秒）
        relative_time = timestamp % SECONDS_PER_DAY
        
        daily_requests[day].append({
            'relative_time': relative_time,
            'request_tokens': request_tokens,
            'response_tokens': response_tokens
        })

print(f"共找到 {len(daily_requests)} 天的数据")
print(f"总请求数: {sum(len(requests) for requests in daily_requests.values())}")

# 处理每一天的数据
for day in sorted(daily_requests.keys()):
    requests = daily_requests[day]
    
    if len(requests) == 0:
        continue
    
    # 按相对时间排序
    requests.sort(key=lambda x: x['relative_time'])
    
    # 准备输出数据
    output_data = []
    request_id = 0
    
    # 获取当天的最小和最大时间（用于线性映射）
    min_time = requests[0]['relative_time']
    max_time = requests[-1]['relative_time']
    
    # 如果只有一条记录，时间范围设为1秒
    if max_time == min_time:
        time_range = 1.0
    else:
        time_range = max_time - min_time
    
    # 处理该天的所有请求
    for req in requests:
        relative_time = req['relative_time']
        
        # 线性映射到0-300秒
        if time_range > 0:
            # 归一化到0-1，然后映射到0-300
            normalized_time = (relative_time - min_time) / time_range
            mapped_timestamp = normalized_time * MAPPED_DAY_LENGTH
        else:
            mapped_timestamp = 0.0
        
        output_data.append({
            'request_id': request_id,
            'request_type': 2,
            'application_id': 0,
            'arrival_timestamp': mapped_timestamp,
            'batch_size': 1,
            'prompt_size': req['request_tokens'],
            'token_size': req['response_tokens']
        })
        request_id += 1
    
    # 生成输出文件名（day_0.csv, day_1.csv, ...）
    output_file = os.path.join(output_dir, f'day_{day}.csv')
    
    # 写入输出文件
    print(f"正在写入第 {day} 天的数据到 {output_file} (共 {len(output_data)} 条记录)")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['request_id', 'request_type', 'application_id', 'arrival_timestamp', 
                      'batch_size', 'prompt_size', 'token_size']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)

print(f"\n处理完成！共生成 {len(daily_requests)} 个文件")
print(f"输出目录: {output_dir}")

