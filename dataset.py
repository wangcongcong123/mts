from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from collections import Counter
import pandas as pd
import torch
from tqdm import tqdm
import os
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVDataset(Dataset):
    def __init__(self, filepath, cate_vocab_save_path=None, numeric_feature_names=[], cate_feature_names=[], nrows=-1,
                 data_cache_path=None,is_reg=False,
                 label_name=None):
        '''
        :param filepath: csv file path
        :param feature_names: columns of the csv file
        '''

        if nrows > 0:
            self.df = pd.read_csv(filepath, nrows=nrows)
        else:
            self.df = pd.read_csv(filepath)

        self.numeric_feature_names = numeric_feature_names
        self.catefn2vocab = {}
        if cate_feature_names != []:
            assert cate_vocab_save_path is not None
            self.cate_feature_names = cate_feature_names

            if not os.path.isdir(cate_vocab_save_path):
                os.makedirs(cate_vocab_save_path)

            if os.path.isfile(os.path.join(cate_vocab_save_path, "catefn2vocab.p")):
                self.catefn2vocab = pickle.load(open(os.path.join(cate_vocab_save_path, "catefn2vocab.p"), "rb"))
            else:
                self.catefn2vocab = self.build_catefn2vocab(self.df, cate_vocab_save_path)
                pickle.dump(self.catefn2vocab, open(os.path.join(cate_vocab_save_path, "catefn2vocab.p"), "wb"))

        label = []
        if label_name is not None:
            label=list(self.df[label_name])
        else:
            label = [-1] * self.df.shape[0]
        if not is_reg:
            unique_label = list(set(label))
            self.num_label = len(unique_label)
            self.label2index = {}
            for i in range(len(unique_label)):
                self.label2index[unique_label[i]] = i

            self.label = torch.tensor([self.label2index[l] for l in label])
            logger.info(f"label2index: {self.label2index}")
            logger.info(f"num_label: {self.num_label}")
        else:
            self.label=torch.tensor(label)

        self.cate_data = []
        self.numeric_data = []
        self.numeric_feature_names = numeric_feature_names
        self.cate_feature_names = cate_feature_names

        if data_cache_path is not None:
            if not os.path.exists(os.path.join(".cache", data_cache_path)):
                os.makedirs(os.path.join(".cache", data_cache_path))
            if os.path.isfile(os.path.join(".cache", data_cache_path, "data.p")):
                logging.info(
                    f'found {os.path.join(".cache", data_cache_path, "data.p")}, so load cate_data and numeric_data')
                data = pickle.load(open(os.path.join(".cache", data_cache_path, "data.p"), "rb"))
                self.cate_data = data["cate_data"]
                self.numeric_data = data["numeric_data"]
            else:
                logging.info(f'not found {os.path.join(".cache", data_cache_path, "data.p")}')
                self.build_data()
                logger.info(f'save cate_data and numeric_data to {os.path.join(".cache", data_cache_path, "data.p")}')
                pickle.dump({"cate_data": self.cate_data, "numeric_data": self.numeric_data},
                            open(os.path.join(".cache", data_cache_path, "data.p"), "wb"))
        else:
            self.build_data()

    def build_data(self):
        for column_name in tqdm(self.df.columns, desc="converting features"):
            if column_name in self.cate_feature_names:
                feature_values = []
                for feature_value in self.df[column_name]:
                    feature_values.append(self.catefn2vocab[column_name].stoi[feature_value])
                self.cate_data.append(torch.tensor(feature_values))
            if column_name in self.numeric_feature_names:
                self.numeric_data.append(torch.tensor(list(self.df[column_name])))

    def save_vocab(self, save_path):
        if len(self.catefn2vocab) != 0:
            pickle.dump(self.catefn2vocab, open(os.path.join(save_path, "catefn2vocab.p"), "wb"))

    def build_catefn2vocab(self, df, cate_feature_names):
        cate_fn2vocab = {}
        for cate_feature_name in tqdm(cate_feature_names, desc='constructing vocabulary for categorical features'):
            series = df[cate_feature_name]
            counter = Counter(series)
            vocab = Vocab(counter)
            cate_fn2vocab[cate_feature_name] = vocab
        return cate_fn2vocab

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        batch_data = ()
        for feature_list in self.cate_data:
            batch_data += (feature_list[item],)
        for feature_list in self.numeric_data:
            batch_data += (feature_list[item],)
        return batch_data, self.label[item]
