import os.path
import time
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
import json
import pandas as pd


device = torch.device("cuda:0")
model_path = 'ksic_model'

try:
    # bertmodel = torch.load(model_path)
    bertmodel = torch.load(os.path.join(model_path, 'KSIC_KoBERT.pt'))  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    bertmodel.load_state_dict(
        torch.load(os.path.join(model_path, 'KSIC_model_state_dict.pt')))  # state_dict를 불러 온 후, 모델에 저장
    _, vocab = get_pytorch_kobert_model(cachedir=".cache")
    print('Using saved model')
except Exception as e:
    print(e)
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")


def load_train_dataset():
    try:
        train_ds = nlp.data.TSVDataset('.cache/ksic_train_ds.tsv', encoding='utf-8',
                                       field_indices=[0, 1], num_discard_samples=1)
        val_ds = nlp.data.TSVDataset('.cache/ksic_val_ds.tsv', encoding='utf-8',
                                     field_indices=[0, 1], num_discard_samples=1)
        test_ds = nlp.data.TSVDataset('.cache/ksic_test_ds.tsv', encoding='utf-8',
                                      field_indices=[0, 1], num_discard_samples=1)
        print(time.strftime('%l:%M%p %Z on %b %d, %Y'))  # ' 1:36PM EDT on Oct 18, 2010'
        print('Loading saved text-label pair dataset completed')
        with open('.cache/label_ksic.pickle', 'rb') as f:
            ksic_index_dict = pickle.load(f)
        with open('.cache/ksic_label.pickle', 'rb') as f:
            ksic_label_dict = pickle.load(f)
    except:
        def load_datasets():
            try:
                with open('.cache/label_ksic.pickle', 'rb') as f:
                    ksic_index_dict = pickle.load(f)
                with open('.cache/ksic_label.pickle', 'rb') as f:
                    ksic_label_dict = pickle.load(f)
                train_input = pd.read_csv('.cache/train_input.csv', encoding='utf-8', low_memory=False)
                val_input = pd.read_csv('.cache/val_input.csv', encoding='utf-8', low_memory=False)
                test_input = pd.read_csv('.cache/test_input.csv', encoding='utf-8', low_memory=False)
                print(time.strftime('%l:%M%p %Z on %b %d, %Y'))  # ' 1:36PM EDT on Oct 18, 2010'
                print('Loading saved train_input completed')
            except:
                fname = ".cache/ksic00.json"
                f_list = [".cache/ksic00.json", ".cache/ksic01.json", ".cache/ksic02.json"]

                with open(fname, encoding='utf-8') as f:
                    for line in tqdm(f):
                        try:
                            # print('line: ', line)
                            temp = json.loads(line)
                            print(temp['an'])
                        #             text = re.sub('[-=.#/?:$}(){,]', ' ', patent['title'] + patent['ab'] + patent['cl'])
                        #             token = text.split()
                        except:
                            pass

                def read_column_names(fn):
                    with open(fn, encoding='utf-8') as json_f:
                        json_line = json_f.readline()
                    temp = json.loads(json_line)
                    return temp.keys()

                col_name = read_column_names(f_list[0])

                temp = []
                error = []
                for fn in f_list[1:]:
                    with open(fn, encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            try:
                                temp.append(json.loads(line.replace('\\\\"', '\\"')))
                            except Exception as e:
                                error.append([e, line])
                raw_df = pd.DataFrame(data=temp, columns=col_name)

                class_count = raw_df['ksic'].value_counts()
                class_count2 = class_count[class_count >= 500]

                raw_df2 = raw_df.loc[raw_df['ksic'].isin(class_count2.keys())].copy()

                ksic_label = raw_df2['ksic'].unique()
                ksic_index_dict = {i: label for i, label in enumerate(ksic_label)}
                ksic_label_dict = {ksic_index_dict[key]: key for key in ksic_index_dict.keys()}

                # ksic_index_dict
                with open('.cache/label_ksic.pickle', 'wb') as f:
                    pickle.dump(ksic_index_dict, f)
                with open('.cache/ksic_label.pickle', 'wb') as f:
                    pickle.dump(ksic_label_dict, f)

                raw_df2['label'] = raw_df2['ksic'].map(ksic_label_dict)
                train_input, test_input = train_test_split(raw_df2, random_state=15, test_size=0.2,
                                                           stratify=raw_df2['ksic'],
                                                           shuffle=True)
                train_input, val_input = train_test_split(train_input, random_state=15, test_size=0.15,
                                                          stratify=train_input['ksic'], shuffle=True)

                train_input.to_csv('.cache/train_input.csv', encoding='utf-8', mode='w', index=False)
                val_input.to_csv('.cache/val_input.csv', encoding='utf-8', mode='w', index=False)
                test_input.to_csv('.cache/test_input.csv', encoding='utf-8', mode='w', index=False)
                print(time.strftime('%l:%M%p %Z on %b %d, %Y'))  # ' 1:36PM EDT on Oct 18, 2010'
                print('Loading json files and saving "train_input.csv" completed')
            return ksic_index_dict, ksic_label_dict, train_input, val_input, test_input

        ksic_index_dict, ksic_label_dict, train_input, val_input, test_input = load_datasets()

        def make_input_text(df):
            input_tl = df[['title', 'label']].copy()
            input_tl.rename(columns={'title': 'text'}, inplace=True)
            input_ab = df[['ab', 'label']].copy()
            input_ab.rename(columns={'ab': 'text'}, inplace=True)
            input_cl = df[['cl', 'label']].copy()
            input_cl.rename(columns={'cl': 'text'}, inplace=True)
            input_text = pd.concat([input_tl, input_ab, input_cl]).copy()
            input_text['text_len'] = input_text['text'].str.len()
            input_text2 = input_text.loc[
                input_text['text_len'] > 3, ['text', 'label']].copy()  # 60813 rows × 3 columns 제거
            return input_text2

        train_ds = make_input_text(train_input)
        val_ds = make_input_text(val_input)
        test_ds = make_input_text(test_input)

        train_ds.to_csv('.cache/ksic_train_ds.tsv', encoding='utf-8', mode='w', index=False, sep='\t')
        val_ds.to_csv('.cache/ksic_val_ds.tsv', encoding='utf-8', mode='w', index=False, sep='\t')
        test_ds.to_csv('.cache/ksic_test_ds.tsv', encoding='utf-8', mode='w', index=False, sep='\t')
        print(time.strftime('%l:%M%p %Z on %b %d, %Y'))  # ' 1:36PM EDT on Oct 18, 2010'
        print('Saving text-label pair dataset completed')
    return train_ds, val_ds, test_ds


tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in tqdm(dataset)]
        self.labels = [np.int32(i[label_idx]) for i in tqdm(dataset)]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


max_len = 256
batch_size = 16
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 10000
learning_rate = 5e-5

print(time.strftime('%l:%M%p %Z on %b %d, %Y'))  # 8:37AM KST on Jun 16, 2022
print('Starting to make BERTDataset')
try:
    # data_train_id = np.load('.cache/data_train_id.npy', mmap_mode='r')
    # data_train_id = np.load('.cache/ksic_data_train_id.npy')
    # data_test_id = np.load('.cache/ksic_data_test_id.npy')
    with open(".cache/ksic_data_train_id.pickle", "rb") as fr:
        data_train_id = pickle.load(fr)
    with open(".cache/ksic_data_test_id.pickle", "rb") as fr:
        ksic_data_test_id = pickle.load(fr)
    # Array can't be memory-mapped: Python objects in dtype. -> mmap_mode='r' 주석처리
    # Object arrays cannot be loaded when allow_pickle=False -> np.dave에 allow_pickle=True 문구 추가
    print(time.strftime('%l:%M%p %Z on %b %d, %Y'))  # ' 1:36PM EDT on Oct 18, 2010'
    print('Completed to load BERTDataset')
except Exception as e:
    print(e)
    train_ds, val_ds, test_ds = load_train_dataset()
    data_train_id = BERTDataset(train_ds, 0, 1, tok, max_len, True, False)
    # np.save('.cache/ksic_data_train_id.npy', data_train_id, allow_pickle=True)
    with open('.cache/ksic_data_train_id.pickle', "wb") as fw:
        pickle.dump(data_train_id, fw)
    # 출처: https: // korbillgates.tistory.com / 173[생물정보학자의 블로그: 티스토리]
    data_test_id = BERTDataset(test_ds, 0, 1, tok, max_len, True, False)
    # np.save('.cache/ksic_data_test_id.npy', data_test_id, allow_pickle=True)
    with open('.cache/ksic_data_test_id.pickle', "wb") as fw:
        pickle.dump(data_test_id, fw)
    # data_train_id = BERTDataset(train_ds.to_numpy(), 0, 1, tok, max_len, True, False)
    # data_test_id = BERTDataset(test_ds.to_numpy(), 0, 1, tok, max_len, True, False)
    print(time.strftime('%l:%M%p %Z on %b %d, %Y'))  # ' 1:36PM EDT on Oct 18, 2010'
    print('Completed to make BERTDataset')
