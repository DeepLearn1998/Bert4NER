import configparser
import os

import torch

from utils.project_path import get_project_path

# 加载配置文件
config = configparser.ConfigParser()
with open(os.path.join(get_project_path(), 'config.ini'), 'r', encoding='utf-8') as config_file:
    config.read_file(config_file)


class NERConfig:
    def __init__(self):
        self.model_id = config['model']['model_id']
        path_list = []
        # 预训练模型路径
        self.bert_dir = os.path.join(get_project_path(), 'models', self.model_id)
        path_list.append(self.bert_dir)
        # 模型保存路径
        self.output_dir = os.path.join(get_project_path(), 'models', 'checkpoint', config['data']['data_name'])
        path_list.append(self.output_dir)
        # 数据存放路径
        self.data_dir = os.path.join(get_project_path(), 'data')
        path_list.append(self.data_dir)
        # 未处理的数据路径
        self.ori_data_dir = os.path.join(self.data_dir, config['data']['data_name'], 'ori_data')
        path_list.append(self.ori_data_dir)
        # 处理后的数据路径
        self.ner_data_dir = os.path.join(self.data_dir, config['data']['data_name'], 'ner_data')
        self.re_data_dir = os.path.join(self.data_dir, config['data']['data_name'], 're_data')
        path_list.append(self.ner_data_dir)
        path_list.append(self.re_data_dir)
        # 检查路径是否存在
        self.path_check(path_list)

        # 读取标签
        self.labels = [config['labels'][label] for label in config['labels']]
        # 构造 bio 标签
        self.bio_labels = ['O']
        for label in self.labels:
            self.bio_labels.append('B-{}'.format(label))
            self.bio_labels.append('I-{}'.format(label))
        self.num_bio_labels = len(self.bio_labels)  # 标签数量
        # 构造 label2id、id2label
        self.label2id = {label: i for i, label in enumerate(self.bio_labels)}
        self.id2label = {i: label for i, label in enumerate(self.bio_labels)}

        # 模型训练参数
        self.max_seq_len = int(config['model']['max_seq_len'])
        self.epochs = int(config['model']['epochs'])
        self.batch_size = int(config['model']['batch_size'])
        self.bert_learning_rate = float(config['model']['bert_learning_rate'])
        self.crf_learning_rate = float(config['model']['crf_learning_rate'])
        self.adam_epsilon = float(config['model']['adam_epsilon'])
        self.weight_decay = float(config['model']['weight_decay'])
        self.warmup_proportion = float(config['model']['warmup_proportion'])
        self.save_step = int(config['model']['save_step'])

        # 运行设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def path_check(path_list):
        """检查路径是否存在"""
        for path in path_list:
            # 标准化路径（处理多余的斜杠等）
            normalized_path = os.path.normpath(path)
            if not os.path.exists(normalized_path):
                try:
                    # 创建所有不存在的目录（包括中间目录）
                    os.makedirs(normalized_path)
                    print(f"创建目录: {normalized_path}")
                except OSError as e:
                    print(f"创建目录失败 [{normalized_path}]: {str(e)}")
