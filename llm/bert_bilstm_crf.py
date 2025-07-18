import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertConfig

from utils.ner_config import NERConfig


class ModelOutput:
    """模型输出"""
    def __init__(self, logits, labels, loss=None):
        self.logits = logits
        self.labels = labels
        self.loss = loss


class BertBiLSTMCRF(nn.Module):
    """用于处理 NER 任务的 Bert-BiLSTM-CRF 模型"""
    def __init__(self, ner_args: NERConfig):
        super(BertBiLSTMCRF, self).__init__()
        # BERT层：加载预训练模型
        self.bert = BertModel.from_pretrained(ner_args.model_id)
        self.bert_config = BertConfig.from_pretrained(ner_args.model_id)  # Bert 配置

        # BiLSTM层：双向LSTM捕捉上下文特征
        hidden_size = self.bert_config.hidden_size  # Bert 隐藏层大小
        self.lstm_hidden = 128  # Bi-LSTM 隐藏层大小
        self.max_seq_len = ner_args.max_seq_len  # 最大序列长度
        self.bi_lstm = nn.LSTM(hidden_size, self.lstm_hidden, 1, bidirectional=True, batch_first=True,
                               dropout=0.1)  # Bi-LSTM

        # CRF层：建模标签转移概率
        self.linear = nn.Linear(self.lstm_hidden * 2, ner_args.num_bio_labels)  # 线性层
        self.crf = CRF(ner_args.num_bio_labels, batch_first=True)  # CRF 层

    def forward(self, input_ids, attention_mask, labels=None):
        # Bert 特征提取
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # Bert 模型输出
        seq_out = bert_output[0]  # 获取 Bert 序列输出 [batch_size, max_seq_len, bert_hidden_size]
        batch_size = seq_out.size(0)  # 获取批次大小

        # Bi-LSTM 上下文编码
        seq_out, _ = self.bi_lstm(seq_out)  # Bi-LSTM 对 Bert 的序列输出进行编码
        seq_out = seq_out.contiguous().view(-1, self.lstm_hidden * 2)  # 展平 Bi-LSTM 输出
        seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)  # 恢复为批次大小
        seq_out = self.linear(seq_out)  # 线性变换

        # CRF 解码
        logits = self.crf.decode(seq_out, mask=attention_mask.bool())

        # 计算损失
        loss = None
        if labels is not None:
            loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')  # 负对数似然损失

        # 返回模型输出
        model_output = ModelOutput(logits, labels, loss)
        return model_output
