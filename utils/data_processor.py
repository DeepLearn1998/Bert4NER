import json
import os.path
import random

from tqdm import tqdm

from utils.ner_config import NERConfig


class DataProcessor:
    def __init__(self, args: NERConfig, train_ratio=0.92):
        # 数据集路径
        self.ori_path = args.ori_data_dir  # 未处理的数据路径
        self.ner_path = args.ner_data_dir  # 处理后的数据路径
        self.re_path = args.re_data_dir
        # 加载训练集
        with open(os.path.join(self.ori_path, 'train.jsonl'), 'r', encoding='utf-8', errors='replace') as fp:
            self.train_data = fp.readlines()
        # 数据集划分比例
        self.train_ratio = train_ratio
        # 标签
        self.h_label = args.labels[0]
        self.t_label = args.labels[1]

    @staticmethod
    def save_labels(data_dir, labels):
        """保存标签"""
        with open(os.path.join(data_dir, 'labels.txt'), 'w', encoding='utf-8') as fp:
            fp.write("\n".join(labels))

    @staticmethod
    def save_jsonl(data_dir, data_type, data):
        """保存数据集"""
        with open(os.path.join(data_dir, f'{data_type}.jsonl'), 'w', encoding='utf-8') as f:
            json_lines = [json.dumps(item, ensure_ascii=False) for item in data]
            f.write('\n'.join(json_lines))

    def split_dataset(self, dataset, data_dir):
        """划分，并保存数据集"""
        # 划分数据集
        train_num = int(len(dataset) * self.train_ratio)
        train_data = dataset[:train_num]
        dev_data = dataset[train_num:]

        # 保存数据集
        self.save_jsonl(data_dir, 'train', train_data)
        self.save_jsonl(data_dir, 'dev', dev_data)

    def get_ner_data(self):
        """将训练集转换为 NER 数据的 sample-label 对"""
        # 存储 sample-label 对
        dataset = []
        # 遍历所有样本
        for did, d in enumerate(tqdm(self.train_data, desc="提取实体信息")):
            d = json.loads(str(d))
            tmp = {}
            text = d.get('text', '')
            tmp['id'] = d.get('ID', '')
            tmp['text'] = [i for i in text]
            tmp['labels'] = ['O'] * len(tmp['text'])
            # 根据 text 标注 BIO 实体
            for rel_id, spo in enumerate(d['spo_list']):
                h = spo['h']
                t = spo['t']
                h_start = h['pos'][0]
                h_end = h['pos'][1]
                t_start = t['pos'][0]
                t_end = t['pos'][1]
                tmp['labels'][h_start] = f'B-{self.h_label}'
                for i in range(h_start + 1, h_end):
                    tmp['labels'][i] = f'I-{self.h_label}'
                tmp['labels'][t_start] = f'B-{self.t_label}'
                for i in range(t_start + 1, t_end):
                    tmp['labels'][i] = f'I-{self.t_label}'
            dataset.append(tmp)

        # 划分并保存数据集
        self.split_dataset(dataset, self.ner_path)

        # 保存标签
        self.save_labels(self.ner_path, [self.h_label, self.t_label])

    def get_re_data(self):
        """将训练集转换为关系型数据"""
        dataset = []  # 存储关系样本对
        re_labels = set()  # 存储关系类型
        # 遍历所有样本
        for did, d in enumerate(tqdm(self.train_data, desc="提取关联条件")):
            d = json.loads(str(d))
            text = d['text']
            sbjs = set()  # 主体
            objs = set()  # 客体
            sbj_obj = []  # 主体-客体
            # 处理主客体，并构建关系
            for rel_id, spo in enumerate(d['spo_list']):
                tmp = {'id': str(did) + '_' + str(rel_id), 'text': text}
                h_name = spo['h']['name']
                t_name = spo['t']['name']
                relation = spo['relation']
                tmp['labels'] = [h_name, t_name, relation]
                re_labels.add(relation)
                dataset.append(tmp)
                sbjs.add(h_name)  # 主体名称
                objs.add(t_name)  # 客体名称
                sbj_obj.append((h_name, t_name))  # 主体-客体
            sbjs = list(sbjs)
            objs = list(objs)

            # 构造负样本：若不在 sbj_obj 中，则视为没有关系
            if len(sbjs) > 1 and len(objs) > 1:
                neg_total = 3  # 负样本最大数量
                neg_cur = 0
                # 遍历所有主体
                for sbj in sbjs:
                    random.shuffle(objs)
                    print(objs)
                    # 遍历所有客体
                    for obj in objs:
                        if (sbj, obj) not in sbj_obj:
                            tmp = {
                                'text': text,
                                'labels': [sbj, obj, "没关系"],
                                'id': str(did) + '_' + 'norel'
                            }
                            dataset.append(tmp)
                            neg_cur += 1
                        if neg_cur >= neg_total:  # 达到数量后跳出
                            break
                    if neg_cur >= neg_total:
                        break

        # 划分并保存数据集
        self.split_dataset(dataset, self.re_path)

        # 保存标签
        self.save_labels(self.re_path, list(re_labels) + ["没关系"])


if __name__ == "__main__":
    ner_args = NERConfig()
    dp = DataProcessor(ner_args)
    dp.get_ner_data()
    dp.get_re_data()
