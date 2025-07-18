import os

import numpy as np
import torch
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer

from llm.bert_bilstm_crf import BertBiLSTMCRF
from utils.ner_config import NERConfig


class Predictor:
    """根据 list 进行 NER，生成预测结果"""
    def __init__(self, args: NERConfig):
        self.ner_id2label = args.id2label  # id2label
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)  # 分词器
        self.max_seq_len = args.max_seq_len  # 最大序列长度
        self.device = args.device  # 运行设备
        # 加载模型
        self.model = BertBiLSTMCRF(args).to(self.device)  # Bert-BiLSTM-CRF 模型
        self.model.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch_model_ner.bin"),
                                              map_location=self.device))

    def ner_tokenizer(self, text):
        """将文本转化为 token ids、attention mask"""
        assert len(text) <= self.max_seq_len - 2, f"文本长度需小于：{self.max_seq_len}"
        text = text[:self.max_seq_len - 2]
        text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)  # 将字符串转化为 id
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))  # padding
        input_ids = torch.tensor(np.array([input_ids])).to(self.device)  # 转换为 tensor
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))  # attention mask
        attention_mask = torch.tensor(np.array([attention_mask])).to(self.device)
        return input_ids, attention_mask

    def ner_predict(self, text):
        """将预测结果转化为实体信息"""
        # 将文本转换为 token ids、attention mask
        input_ids, attention_mask = self.ner_tokenizer(text)
        # Bert-BiLSTM-CRF 模型预测
        output = self.model(input_ids, attention_mask)
        # 预测结果转化为标签：logits -> BIO
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])  # 计算有效序列长度
        logits = output.logits  # 模型原始输出
        logits = logits[0][1:length - 1]  # 过滤 [CLS]、[SEP]
        logits = [self.ner_id2label[i] for i in logits]  # 将 logits 转化为 BIO 标签
        # 标签转换为实体：BIO -> entity
        entities = get_entities(logits)  # 获取实体三元组 [(类型, 起始位置, 结束位置),...]
        result = {}  # 记录结果
        for ent in entities:
            ent_name = ent[0]  # 实体类型
            ent_start = ent[1]  # 起始位置
            ent_end = ent[2]  # 结束位置
            if ent_name not in result:
                result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
            else:
                result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
        return result


if __name__ == "__main__":
    # 加载参数
    ner_args = NERConfig()
    predictor = Predictor(ner_args)
    events = [
        "测试文本1",
        "测试文本2",
        "测试文本3",
        "测试文本4",
        "测试文本5",
    ]
    for event in events:
        ner_result = predictor.ner_predict(event)
        print(f"输入文本\n{event}\n实体信息\n{ner_result}\n")
