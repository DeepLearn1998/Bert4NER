import json
import os

import torch
from seqeval.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, BertTokenizer

from utils.data_loader import NERDataset
from llm.bert_bilstm_crf import BertBiLSTMCRF
from utils.ner_config import NERConfig


class Trainer:
    def __init__(self,
                 model,
                 train_loader,
                 dev_loader,
                 optimizer,
                 schedule,
                 args: NERConfig):
        self.output_dir = args.output_dir
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.epochs = args.epochs
        self.device = args.device
        self.optimizer = optimizer
        self.schedule = schedule
        self.id2label = args.id2label
        self.save_step = args.save_step
        self.total_step = len(self.train_loader) * self.epochs

    def data_to_device(self, batch_data):
        """将数据加载至设备"""
        for key, value in batch_data.items():
            batch_data[key] = value.to(self.device)  # 将数据加载至设备 CPU/GPU
        input_ids = batch_data["input_ids"]  # 输入 ids
        attention_mask = batch_data["attention_mask"]  # 注意力 mask
        labels = batch_data["labels"]  # 标签
        output = self.model(input_ids, attention_mask, labels)  # 前向传播
        return input_ids, attention_mask, labels, output

    def train(self):
        # 全局步数
        global_step = 1
        # 训练进度条
        epoch_pbar = tqdm(total=self.epochs, desc="训练进度", unit="epoch")
        for epoch in range(1, self.epochs + 1):  # 遍历每个 epoch
            # 训练阶段
            self.model.train()
            for step, batch_data in enumerate(self.train_loader):  # 遍历每个 batch
                _, _, _, output = self.data_to_device(batch_data)
                # 计算损失
                loss = output.loss
                # 反向传播
                self.optimizer.zero_grad()  # 梯度归零
                loss.backward()  # 反向传播计算梯度
                self.optimizer.step()  # 更新模型参数
                self.schedule.step()  # 更新学习率调度器
                global_step += 1
                # 定期保存模型，默认500步
                if global_step % self.save_step == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_ner.bin"))
            epoch_pbar.update(1)
        epoch_pbar.close()
        # 保存模型
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_ner.bin"))

    def test(self):
        tqdm.write("加载最佳模型进行预测...")
        # 加载模型并切换至测试模式
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "pytorch_model_ner.bin")))
        self.model.eval()
        # 记录测试集预测结果
        pred = []  # 预测结果
        true = []  # 真实结果
        test_pbar = tqdm(self.dev_loader, desc="测试集预测", unit="batch")
        for step, batch_data in enumerate(test_pbar):  # 遍历每个 batch
            # 数据准备
            input_ids, attention_mask, labels, output = self.data_to_device(batch_data)
            # 处理模型输出
            logits = output.logits  # Bert-BiLSTM-CRF 的输出
            attention_mask = attention_mask.detach().cpu().numpy()  # .detach() 阻断反向传播
            labels = labels.detach().cpu().numpy()  # 获取标签
            batch_size = input_ids.size(0)  # 获取 batch_size
            for i in range(batch_size):  # 遍历每个样本
                length = sum(attention_mask[i])  # 计算注意力 mask 的有效长度（排除 padding 部分）
                # 处理预测结果
                logit = logits[i][1:length]  # 过滤 [CLS] 标签
                logit = [self.id2label[i] for i in logit]
                pred.append(logit)
                # 处理真实结果
                label = labels[i][1:length]
                label = [self.id2label[i] for i in label]
                true.append(label)
        test_pbar.close()
        # 生成评估结果
        report = classification_report(true, pred)
        return report


def build_optimizer_and_scheduler(args, model, t_total):
    """按照差分学习率，构建优化器和学习率调度器"""
    # 获取模型
    module = model.module if hasattr(model, "module") else model  # 多卡并行时，模型在 model.module 中

    # 差分学习率：预训练层（如Bert）设置较小的学习率，新增层（如CRF或BiLSTM）设置较大的学习率
    no_decay = ["bias", "LayerNorm.weight"]  # 无需权重衰减的参数
    model_param = list(module.named_parameters())  # 模型参数

    # 获取参数
    bert_param_optimizer = []  # 预训练层（Bert）参数
    other_param_optimizer = []  # 新增层（CRF或BiLSTM）参数
    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module' or space[0] == "bert":
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    # 根据模型参数，分组设置优化器
    optimizer_grouped_parameters = [
        # 预训练层参数
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.bert_learning_rate},  # 带权重衰减
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.bert_learning_rate},  # 无权重衰减

        # 新增层参数
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_learning_rate},  # 带权重衰减
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.crf_learning_rate},  # 无权重衰减
    ]

    # 创建 AdamW 优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_learning_rate, eps=args.adam_epsilon)
    # 创建具有 warmup 的线性学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * t_total),
                                                num_training_steps=t_total)
    return optimizer, scheduler


def dataset_to_dataloader(data_type, ner_args, tokenizer):
    """将数据集加载至 DataLoader"""
    with open(os.path.join(ner_args.ner_data_dir, f"{data_type}.jsonl"), "r", encoding="utf-8") as fp:
        data = fp.read().split("\n")
    data = [json.loads(d) for d in data]
    dataset = NERDataset(data, ner_args, tokenizer)
    return DataLoader(dataset, shuffle=True, batch_size=ner_args.batch_size, num_workers=2)


def main():
    # 加载参数
    ner_args = NERConfig()

    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(ner_args.model_id, cache_dir=ner_args.bert_dir)

    # 加载数据集
    with tqdm(total=3, desc="数据集加载") as pbar:
        train_loader = dataset_to_dataloader('train', ner_args, tokenizer)
        pbar.update(1)
        dev_loader = dataset_to_dataloader('dev', ner_args, tokenizer)
        pbar.update(1)

    # 创建 Bert 模型，处理 NER 任务
    model = BertBiLSTMCRF(ner_args).to(ner_args.device)
    # # 查看模型所有参数
    # for name, _ in model.named_parameters():
        # print(name)
    total_step = len(train_loader) * ner_args.epochs  # 训练总步数
    optimizer, schedule = build_optimizer_and_scheduler(ner_args, model, total_step)  # 创建优化器及调度器

    # 模型训练
    train = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        schedule=schedule,
        args=ner_args,
    )
    train.train()

    # 模型预测
    report = train.test()
    print(report)


if __name__ == "__main__":
    main()
