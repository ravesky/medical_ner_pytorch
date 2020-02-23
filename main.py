import pickle
import sys

import yaml
import numpy as np
import torch
import torch.optim as optim
import time
from data_manager import DataManager
from model import BiLSTMCRF
from utils import f1_score, get_tags, format_result
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./tensorboard/5epoch',comment='_BiLstm_CRF_5epcoh')

class ChineseNER(object):
    
    def __init__(self, entry="train"):
        self.load_config()
        self.__init_model(entry)

    def __init_model(self, entry):
        if entry == "train":
            self.train_manager = DataManager(batch_size=self.batch_size, tags=self.tags)
            self.total_size = len(self.train_manager.batch_data)
            data = {
                "batch_size": self.train_manager.batch_size,
                "input_size": self.train_manager.input_size,
                "vocab": self.train_manager.vocab,
                "tag_map": self.train_manager.tag_map,
            }
            self.save_params(data)
            self.dev_manager = DataManager(batch_size=60, data_type="dev")
            # 验证集
            self.dev_batch = self.dev_manager.iteration()

            self.model = BiLSTMCRF(
                tag_map=self.train_manager.tag_map,
                batch_size=self.batch_size,
                vocab_size=len(self.train_manager.vocab),
                dropout=self.dropout,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size,
            )
            self.model = self.model.cuda()
            self.restore_model()
        elif entry == "predict":
            # python main.py predict
            data_map = self.load_params()
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")
            print('input_size',input_size)
            print('tag_map',self.tag_map)
            print('vocab',self.vocab)
            self.model = BiLSTMCRF(
                tag_map=self.tag_map,
                vocab_size=input_size,
                embedding_dim=self.embedding_size,
                hidden_dim=self.hidden_size
            )
            self.model = self.model.cuda()
            self.restore_model()
    # 加载配置项
    def load_config(self):
        try:
            fopen = open("models/config.yml")
            config = yaml.load(fopen)
            fopen.close()
        except Exception as error:
            print("Load config failed, using default config {}".format(error))
            fopen = open("models/config.yml", "w")
            config = {
                "embedding_size": 300,
                "hidden_size": 128,
                "batch_size": 30,
                "dropout":0.5,
                "model_path": "models/",
                "tags": ["TREATMENT", "BODY","SIGNS","CHECK","DISEASE"]
            }
            yaml.dump(config, fopen)
            fopen.close()
        self.embedding_size = config.get("embedding_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.model_path = config.get("model_path")
        self.tags = config.get("tags")
        self.dropout = config.get("dropout")

    # 保存模型各种训练参数
    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params_5all.pkl"))
            print("model restore success!")
        except Exception as error:
            print("model restore faild! {}".format(error))

    # 保存模型超参数
    def save_params(self, data):
        with open("models/data_5all.pkl", "wb") as fopen:
            pickle.dump(data, fopen)

    # 加载模型超参数
    def load_params(self):
        with open("models/data_5all.pkl", "rb") as fopen:
            data_map = pickle.load(fopen)
        return data_map

    def train(self):
        optimizer = optim.Adam(self.model.parameters(),weight_decay=1e-2,lr=0.00001)
        # optimizer = optim.SGD(self.model.parameters(), lr=0.001,weight_decay=0.01,momentum=0.9)
        scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
        for epoch in range(74,400):
            losses = []
            index = 0
            startTime = time.process_time()
            for batch in self.train_manager.get_batch():
                start = time.process_time()
                index += 1
                self.model.zero_grad()

                sentences, tags, length = zip(*batch)
                # lenght 是句子的原本长度
                # shape (batch_size,max.len(sentence) (20,332) batch_size 和 每个batch最长句子的长度
                sentences_tensor = torch.tensor(sentences, dtype=torch.long).cuda()
                tags_tensor = torch.tensor(tags, dtype=torch.long).cuda()
                length_tensor = torch.tensor(length, dtype=torch.long).cuda()

                loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)
                losses.append(loss.cpu().item())
                progress = ("█"*int(index * 25 / self.total_size)).ljust(25)
                # self.evaluate(index)
                loss.backward()
                optimizer.step()
                torch.save(self.model.state_dict(), self.model_path + 'params_5all.pkl')
                end = time.process_time()
                dur = end - start
                print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}\t\ttime {}""".format(
                    epoch, progress, index, self.total_size, loss.cpu().tolist()[0],str(dur)
                    )
                )
                print("-" * 50)
            endTime = time.process_time()
            totalTime = endTime - startTime
            avg_loss = np.mean(losses)
            writer.add_scalar('BiLstm_CRF:avg_loss-epoch', avg_loss, epoch)
            print('epoch ',epoch,'avg_loss ', avg_loss,'total_time ',totalTime)
            if epoch % 5 == 0:
                self.calculaterF1(epoch/5)
            print("-"*100)
            scheduler_lr.step()
        writer.close()
    # train: BODY 7507, SIGNS 6355, CHECK 6965, DISEASE 474, TREATMENT 805
    # test:
    def calculaterF1(self,epoch):
        all_origins = all_founds = all_rights = 0
        for tag in self.tags:
            origins = founds = rights = 0
            for batch in self.dev_manager.get_batch():
                sentences, labels, length = zip(*batch)
                _, paths = self.model(sentences)
                origin, found, right = f1_score(labels, paths, tag, self.model.tag_map)
                origins += origin
                founds += found
                rights += right
            all_origins += origins
            all_founds += founds
            all_rights += rights
            recall = 0. if origins == 0 else (rights / origins)
            precision = 0. if founds == 0 else (rights / founds)
            f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
            print("\t{}\torgins:{}\tfounds:{}\trights:{}".format(tag, origins, founds, rights))
            print("\t  \trecall:{}\tprecision:{}\tf1:{}".format(recall, precision, f1))
            tag_epoch = tag + '-5epoch'
            writer.add_scalars(tag_epoch,{
                'recall':recall,
                'precision':precision,
                'f1':f1
            }, epoch)
        all_recall = 0. if all_origins == 0 else (all_rights / all_origins)
        all_precision = 0. if all_founds == 0 else (all_rights / all_founds)
        all_f1 = 0. if all_recall + all_precision == 0 else (2 * all_precision * all_recall) / (all_precision + all_recall)
        print("\tall_orgins:{}\tall_founds:{}\tall_rights:{}".format(all_origins, all_founds, all_rights))
        print("\tall_recall:{}\tall_precision:{}\tall_f1:{}".format(all_recall, all_precision, all_f1))
        writer.add_scalars("ALL-5epoch", {
            'all_recall': all_recall,
            'all_precision': all_precision,
            'all_f1': all_f1
        }, epoch)
        return all_recall, all_precision, all_f1

    def evaluate(self,index):
        # batch_size 个句子
        sentences, labels, length = zip(*self.dev_batch.__next__())
        _, paths = self.model(sentences)
        print("\t评估验证集(batch[{}])".format(index))
        for tag in self.tags:
            f1_score(labels, paths, tag, self.model.tag_map)
    # 预测方法
    def predict(self, input_str=""):
        if not input_str:
            input_str = input("请输入文本: ")
        # 获取输入句子所有汉字的在vocab的索引
        input_vec = [self.vocab.get(i, 0) for i in input_str]
        # convert to tensor
        sentences = torch.tensor(input_vec,dtype=torch.long).view(1, -1)
        sentences = sentences.cuda()
        # paths 预测出来的标签索引 shape 为 [1,1]
        _, paths = self.model(sentences)

        entities = []
        # "tags": ["ORG", "PER"]
        for tag in self.tags:
            tags = get_tags(paths[0], tag, self.tag_map)
            entities += format_result(tags, input_str, tag)
        return entities

if __name__ == "__main__":
    entry = input('请输入train or predict: ')
    if entry == 'train':
        cn = ChineseNER("train")
        cn.train()
    elif entry == 'predict':
        cn = ChineseNER("predict")
        print(cn.predict())
    else:
        pass
    # if len(sys.argv) < 2:
    #     print("menu:\n\ttrain\n\tpredict")
    #     exit()
    # if sys.argv[1] == "train" or entry == 'train':
    #     cn = ChineseNER("train")
    #     cn.train()
    # elif sys.argv[1] == "predict" or entry == 'predict':
    #     cn = ChineseNER("predict")
    #     print(cn.predict())
