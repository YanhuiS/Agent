import argparse
import json
import random
import re
import numpy as np
import paddle

def set_seed(seed):
    """
    设置随机种子以确保结果的可重复性。
    """
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_txt(file_path):
    """
    加载文本文件，将每一行文本存入列表中。
    Args:
        file_path (str): 文本文件路径。
    Returns:
        texts (list): 文本列表，每个元素为一行文本。
    """
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            texts.append(line.strip())
    return texts

def load_json_file(path):
    """
    加载JSON文件，将每一行JSON数据存入列表中。
    Args:
        path (str): JSON文件路径。
    Returns:
        examples (list): JSON对象列表，每个元素为一行JSON数据。
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            example = json.loads(line)
            examples.append(example)
    return examples

def write_json_file(examples, save_path):
    """
    将JSON对象列表写入文件，每行一个JSON对象。
    Args:
        examples (list): JSON对象列表。
        save_path (str): 保存路径。
    """
    with open(save_path, "w", encoding="utf-8") as f:
        for example in examples:
            line = json.dumps(example, ensure_ascii=False)
            f.write(line + "\n")

def str2bool(v):
    """
    支持argparse模块的bool类型转换。
    Args:
        v (str): 字符串形式的布尔值。
    Returns:
        bool值。
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def create_data_loader(dataset, mode="train", batch_size=1, trans_fn=None):
    """
    创建数据加载器。
    Args:
        dataset (paddle.io.Dataset): 数据集实例。
        mode (str, optional): 模式，可选 'train' 或 'test'，默认为 'train'。
        batch_size (int, optional): 批大小，默认为1。
        trans_fn (callable, optional): 数据转换函数，默认为None。
    Returns:
        dataloader (paddle.io.DataLoader): 数据加载器。
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, return_list=True)
    return dataloader

def convert_example(example, tokenizer, max_seq_len):
    """
    转换示例数据，生成模型输入格式。
    Args:
        example (dict): 输入示例，包含 'title'、'prompt'、'content' 和 'result_list'。
        tokenizer (obj): 分词器实例。
        max_seq_len (int): 最大序列长度。
    Returns:
        tokenized_output (tuple): 模型输入数据，包括input_ids、token_type_ids、position_ids、attention_mask、start_ids和end_ids。
    """
    encoded_inputs = tokenizer(
        text=[example["prompt"]],
        text_pair=[example["content"]],
        truncation=True,
        max_seq_len=max_seq_len,
        pad_to_max_seq_len=True,
        return_attention_mask=True,
        return_position_ids=True,
        return_dict=False,
        return_offsets_mapping=True,
    )
    encoded_inputs = encoded_inputs[0]
    offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]
    bias = 0
    for index in range(1, len(offset_mapping)):
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
            bias = offset_mapping[index - 1][1] + 1  # 包括 [SEP] 标记
        if mapping[0] == 0 and mapping[1] == 0:
            continue
        offset_mapping[index][0] += bias
        offset_mapping[index][1] += bias
    start_ids = [0 for x in range(max_seq_len)]
    end_ids = [0 for x in range(max_seq_len)]
    for item in example["result_list"]:
        start = map_offset(item["start"] + bias, offset_mapping)
        end = map_offset(item["end"] - 1 + bias, offset_mapping)
        start_ids[start] = 1.0
        end_ids[end] = 1.0

    tokenized_output = [
        encoded_inputs["input_ids"],
        encoded_inputs["token_type_ids"],
        encoded_inputs["position_ids"],
        encoded_inputs["attention_mask"],
        start_ids,
        end_ids,
    ]
    tokenized_output = [np.array(x, dtype="int64") for x in tokenized_output]
    return tuple(tokenized_output)

def map_offset(ori_offset, offset_mapping):
    """
    将原始偏移映射到标记偏移。
    Args:
        ori_offset (int): 原始偏移。
        offset_mapping (list): 偏移映射列表。
    Returns:
        index (int): 标记偏移索引。
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1

def reader(data_path, max_seq_len=512):
    """
    读取JSON文件，生成模型输入格式数据。
    Args:
        data_path (str): JSON文件路径。
        max_seq_len (int, optional): 最大序列长度，默认为512。
    Yields:
        json_line (dict): JSON行数据，包含 'content'、'result_list' 和 'prompt'。
    """
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line["content"].strip()
            prompt = json_line["prompt"]
            if max_seq_len <= len(prompt) + 3:
                raise ValueError("The value of max_seq_len is too small, please set a larger value")
            max_content_len = max_seq_len - len(prompt) - 3
            if len(content) <= max_content_len:
                yield json_line
            else:
                result_list = json_line["result_list"]
                json_lines = []
                accumulate = 0
                while True:
                    cur_result_list = []

                    for result in result_list:
                        if result["start"] + 1 <= max_content_len < result["end"]:
                            max_content_len = result["start"]
                            break

                    cur_content = content[:max_content_len]
                    res_content = content[max_content_len:]

                    while True:
                        if len(result_list) == 0:
                            break
                        elif result_list[0]["end"] <= max_content_len:
                            if result_list[0]["end"] > 0:
                                cur_result = result_list.pop(0)
                                cur_result_list.append(cur_result)
                            else:
                                cur_result_list = [result for result in result_list]
                                break
                        else:
                            break

                    json_line = {"content": cur_content, "result_list": cur_result_list, "prompt": prompt}
                    json_lines.append(json_line)

                    for result in result_list:
                        if result["end"] <= 0:
                            break
                        result["start"] -= max_content_len
                        result["end"] -= max_content_len
                    accumulate += max_content_len
                    max_content_len = max_seq_len - len(prompt) - 3
                    if len(res_content) == 0:
                        break
                    elif len(res_content) < max_content_len:
                        json_line = {"content": res_content, "result_list": result_list, "prompt": prompt}
                        json_lines.append(json_line)
                        break
                    else:
                        content = res_content

                for json_line in json_lines:
                    yield json_line

def unify_prompt_name(prompt):
    """
    在评估期间统一分类标签。
    Args:
        prompt (str): 输入提示。
    Returns:
        prompt (str): 统一后的提示。
    """
    if re.search(r"\[.*?\]$", prompt):
        prompt_prefix = prompt[: prompt.find("[", 1)]
        cls_options = re.search(r"\[.*?\]$", prompt).group()[1:-1].split(",")
        cls_options = sorted(list(set(cls_options)))
        cls_options = ",".join(cls_options)
        prompt = prompt_prefix + "[" + cls_options + "]"
        return prompt
    return prompt

def get_relation_type_dict(relation_data):
    """
    获取关系类型字典。
    Args:
        relation_data (list): 关系数据列表。
    Returns:
        relation_type_dict (dict): 关系类型字典。
    """
    def compare(a, b):
        a = a[::-1]
        b = b[::-1]
        res = ""
        for i in range(min(len(a), len(b))):
            if a[i] == b[i]:
                res += a[i]
            else:
                break
        if res == "":
            return res
        elif res[::-1][0] == "的":
            return res[::-1][1:]
        return ""

    relation_type_dict = {}
    added_list = []
    for i in range(len(relation_data)):
        added = False
        if relation_data[i][0] not in added_list:
            for j in range(i + 1, len(relation_data)):
                match = compare(relation_data[i][0], relation_data[j][0])
                if match != "":
                    match = unify_prompt_name(match)
                    if relation_data[i][0] not in added_list:
                        added_list.append(relation_data[i][0])
                        relation_type_dict.setdefault(match, []).append(relation_data[i][1])
                    added_list.append(relation_data[j][0])
                    relation_type_dict.setdefault(match, []).append(relation_data[j][1])
                    added = True
            if not added:
                added_list.append(relation_data[i][0])
                suffix = relation_data[i][0].rsplit("的", 1)[1]
                suffix = unify_prompt_name(suffix)
                relation_type_dict[suffix] = relation_data[i][1]
    return relation_type_dict
