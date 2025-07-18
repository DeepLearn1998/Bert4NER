import numpy as np
import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, data, args, tokenizer):
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # 获取样本（text）和标签（label）
        text = self.data[item]["text"]
        labels = self.data[item]["labels"]
        # 统一文本长度
        if len(text) > self.max_seq_len - 2:  # 保留 CLS 和 SEP 的位置
            text = text[:self.max_seq_len - 2]
            labels = labels[:self.max_seq_len - 2]
        # 样本转 id
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = torch.tensor(np.array(input_ids))
        # 注意力掩码
        attention_mask = [1] * len(tmp_input_ids)
        attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        # 标签转 id
        labels = [self.label2id[label] for label in labels]
        labels = [0] + labels + [0] + [0] * (self.max_seq_len - len(tmp_input_ids))
        labels = torch.tensor(np.array(labels))
        # 输出
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data
