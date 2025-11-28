import os
import datetime
import json


def get_today_date():
    # 获取当前日期和时间
    today = datetime.datetime.now()
    
    # 格式化日期为字符串，格式为 "YYYY-MM-DD"
    date_string = today.strftime("%Y%m%d")
    return date_string

 
def get_directory_size(directory):
    """递归计算文件夹的总大小（字节）"""
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # 忽略无效链接（可选）
            if not os.path.exists(file_path):
                continue
            total_size += os.path.getsize(file_path)
    return total_size

def file_size(path, n):
    has_directory = False
    has_file = False
    file_size_bytes = 0

    pre_entries = os.listdir(path)
    if not pre_entries:  # 空目录情况
        return 0

    # 检查路径下是否有文件或目录
    for entry in pre_entries:
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path):
            has_directory = True
        elif os.path.isfile(entry_path):
            has_file = True
        break  # 只需要检查第一个条目

    if has_file:
        for file_name in pre_entries:
            base, ext = os.path.splitext(file_name)
            if not ext:
                continue
            if "_" not in base:
                continue
            prefix, old_num = base.rsplit("_", 1)
            num_digits = len(old_num)
            new_num = str(n).zfill(num_digits)
            new_file_name = f"{prefix}_{new_num}{ext}"
            file_path = os.path.join(path, new_file_name)
            if os.path.isfile(file_path):
                file_size_bytes += os.path.getsize(file_path)
                break
        return file_size_bytes

    elif has_directory:
        for subdir in pre_entries:
            subdir_path = os.path.join(path, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            # 先检查子目录中的文件
            found = False
            for file_name in os.listdir(subdir_path):
                base, ext = os.path.splitext(file_name)
                if "_" not in base:
                    continue
                prefix, old_num = base.rsplit("_", 1)
                num_digits = len(old_num)
                new_num = str(n).zfill(num_digits)
                new_file_name = f"{prefix}_{new_num}{ext}"
                file_path = os.path.join(subdir_path, new_file_name)
                if os.path.isfile(file_path):
                    file_size_bytes += os.path.getsize(file_path)
                    found = True
                    break
                # 如果没找到文件，递归检查子目录
                if not found:
                    file_size_bytes += get_directory_size(file_path)
                break
                
        return file_size_bytes

    return 0

def get_data_size(fold_path, data): # 文件大小单位(MB)
    try:
        size_bytes = 0

        # directory_path = os.path.join(fold_path, get_today_date())
        # print(directory_path)
        # #directory_path = os.path.join(fold_path1, "2025701")
        # if not os.path.exists(directory_path):
        #     return 400

        # task_path = os.path.join(directory_path,f"{str(data['task_name'])}_{str(data['task_id'])}")

        task_path=fold_path
        opdata_path = os.path.join(task_path,"meta","op_dataid.jsonl")
        with open(opdata_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object_data = json.loads(line.strip())
                    if json_object_data["dataid"] == str(data["task_data_id"]):
                        episode_index = json_object_data["episode_index"]
                        break
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
        
        entries_1 = os.listdir(task_path) 
        for entry in entries_1:
            if entry == "meta":
                continue
            if entry == "videos":
                if "images" in entries_1:
                    continue
                data_path = os.path.join(task_path,entry,"chunk-000")
                size_bytes += file_size(data_path,episode_index)
            if entry == "images":
                data_path = os.path.join(task_path,entry)
                size_bytes += file_size(data_path,episode_index)
            if entry == "data":
                data_path = os.path.join(task_path,entry,"chunk-000")
                size_bytes += file_size(data_path,episode_index)
            if entry == "audio":
                data_path = os.path.join(task_path,entry,"chunk-000")
                size_bytes += file_size(data_path,episode_index)
        size_mb = round(size_bytes / (1024 * 1024), 2)
        return size_mb


    except Exception as e:
        print(str(e))
        return 500
    




def get_data_duration(fold_path,data):  # 文件时长单位(s)
    try:
        # directory_path = os.path.join(fold_path, get_today_date())
        # print(directory_path)
        # #directory_path = os.path.join(fold_path1, "2025701")
        # if not os.path.exists(directory_path):
        #     return 30
        
        # task_path = os.path.join(directory_path,f"{str(data['task_name'])}_{str(data['task_id'])}")
        task_path = fold_path
        info_path = os.path.join(task_path,"meta","info.json")
        opdata_path = os.path.join(task_path,"meta","op_dataid.jsonl")
        episodes_path = os.path.join(task_path,"meta","episodes.jsonl")
        with open(info_path,"r",encoding="utf-8") as f:
            info_data = json.load(f)
            fps = info_data["fps"] # 
        with open(opdata_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object_data = json.loads(line.strip())
                    if json_object_data["dataid"] == str(data["task_data_id"]):
                        episode_index = json_object_data["episode_index"]
                        break
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
        with open(episodes_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # 去除行末的换行符，并解析为 JSON 对象
                    json_object_data = json.loads(line.strip())
                    if json_object_data["episode_index"] == episode_index:
                        length= json_object_data["length"]
                        break
                except json.JSONDecodeError as e:
                    print(f"解析 JSON 失败，行内容: {line.strip()}, 错误信息: {e}")
        duration = round(length/fps,2)
        return duration        
    except Exception as e:
        print(str(e))
        return 30

def update_dataid_json(path, episode_index, data):
    opdata_path = os.path.join(path, "meta", "op_dataid.jsonl")

    append_data = {
        "episode_index": episode_index,
        "dataid": str(data["task_data_id"]),
    }
    
    # 以追加模式打开文件（如果不存在则创建）
    with open(opdata_path, 'a', encoding='utf-8') as f:
        # 写入一行 JSON 数据（每行一个 JSON 对象）
        f.write(json.dumps(append_data, ensure_ascii=False) + '\n')

def find_epindex_from_dataid_json(path: str, task_data_id: str) -> int:
    """
    根据 task_data_id 从 op_dataid.jsonl 文件中查询对应的 episode_index
    
    Args:
        path: 数据根目录路径（包含 meta 子目录）
        task_data_id: 需要查询的任务数据ID
    
    Returns:
        int: 对应的 episode_index 值
        
    Raises:
        FileNotFoundError: 当 op_dataid.jsonl 文件不存在时
        ValueError: 当指定 task_data_id 未找到时
    """
    opdata_path = os.path.join(path, "meta", "op_dataid.jsonl")
    
    if not os.path.exists(opdata_path):
        raise FileNotFoundError(f"元数据文件不存在: {opdata_path}")
    
    # 规范化 task_data_id 类型（确保字符串比较）
    target_id = str(task_data_id).strip()
    
    with open(opdata_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                # 严格匹配 dataid 字段（考虑大小写和空格）
                if str(record.get("dataid", "")).strip() == target_id:
                    return int(record["episode_index"])
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # 跳过无效行但记录警告（实际项目中可添加日志）
                continue
    
    raise ValueError(f"未找到 task_data_id={task_data_id} 对应的 episode_index")

def delete_dataid_json(path, episode_index, data):
    opdata_path = os.path.join(path, "meta", "op_dataid.jsonl")
    
    # 构建要删除的匹配条件
    target_episode = episode_index
    target_dataid = str(data["task_data_id"])
    
    # 如果文件不存在，直接返回（无内容可删除）
    if not os.path.exists(opdata_path):
        return
    
    # 临时存储过滤后的数据
    filtered_data = []
    
    # 读取并过滤文件内容
    with open(opdata_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # 定义匹配条件（同时匹配episode_index和dataid）
                if (entry.get("episode_index") == target_episode and 
                    entry.get("dataid") == target_dataid):
                    continue  # 跳过匹配的条目（即删除）
                filtered_data.append(entry)
            except json.JSONDecodeError:
                continue  # 跳过无效JSON行
    
    # 覆盖写回文件（不含匹配条目）
    with open(opdata_path, 'w', encoding='utf-8') as f:
        for entry in filtered_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def update_common_record_json(path, data):
    opdata_path = os.path.join(path, "meta", "common_record.json")

    overwrite_data = {
        "task_id": str(data["task_id"]),
        "task_name": str(data["task_name"]),
        # "machine_id": str(data["machine_id"]),
    }
    
    # 以追加模式打开文件（如果不存在则创建）
    with open(opdata_path, 'w', encoding='utf-8') as f:
        # 写入一行 JSON 数据（每行一个 JSON 对象）
        f.write(json.dumps(overwrite_data, ensure_ascii=False) + '\n')


# if __name__ == '__main__':
#     fold_path = '/home/liuyou/Documents'
#     data = {
#         "task_id": "187",
#         "task_name": "刀具安全取放",
#         "task_data_id": "2043",
#         "collector_id":"001",
#         "task_steps": [
#             {
#                 "doruation": "10",
#                 "instruction": "put"
#             },
#             {
#                 "doruation": "2",
#                 "instruction": "close"
#             },
#             {
#                 "doruation": "5",
#                 "instruction": "clean"
#             }
#         ]
#     } # 之后作为参数传递
#     print(data_size(fold_path,data))
#     print(data_duration(fold_path,data))
        









