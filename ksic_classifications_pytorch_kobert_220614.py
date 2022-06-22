import os.path
import time
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from transformers import AdamW, BertModel
from transformers.optimization import get_cosine_schedule_with_warmup
import json
import pandas as pd

# device = torch.device("cpu")
device = torch.device("cuda:0")
model_path = 'ksic_model'

try:
    # bertmodel = BertModel.from_pretrained(model_path, return_dict=False)
    bertmodel = torch.load(os.path.join(model_path, 'KSIC_KoBERT.pt'))  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
    bertmodel.load_state_dict(
        torch.load(os.path.join(model_path, 'KSIC_model_state_dict.pt')))  # state_dict를 불러 온 후, 모델에 저장
    _, vocab = get_pytorch_kobert_model(cachedir=".cache")
    print('Using saved model')
except Exception as e:
    print(e)
    bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")


# def load_datasets():
#     try:
#         with open('.cache/label_ksic.pickle', 'rb') as f:
#             ksic_index_dict = pickle.load(f)
#         with open('.cache/ksic_label.pickle', 'rb') as f:
#             ksic_label_dict = pickle.load(f)
#         train_input = pd.read_csv('.cache/train_input.csv', encoding='utf-8', low_memory=False)
#         val_input = pd.read_csv('.cache/val_input.csv', encoding='utf-8', low_memory=False)
#         test_input = pd.read_csv('.cache/test_input.csv', encoding='utf-8', low_memory=False)
#         print(time.strftime('%l:%M%p %Z on %b %d, %Y'))  # ' 1:36PM EDT on Oct 18, 2010'
#         print('Loading saved train_input completed')
#     except:
#         fname = ".cache/ksic00.json"
#         f_list = [".cache/ksic00.json", ".cache/ksic01.json", ".cache/ksic02.json"]
#
#         with open(fname, encoding='utf-8') as f:
#             for line in tqdm(f):
#                 try:
#                     # print('line: ', line)
#                     temp = json.loads(line)
#                     print(temp['an'])
#                 #             text = re.sub('[-=.#/?:$}(){,]', ' ', patent['title'] + patent['ab'] + patent['cl'])
#                 #             token = text.split()
#                 except:
#                     pass
#
#         def read_column_names(fn):
#             with open(fn, encoding='utf-8') as json_f:
#                 json_line = json_f.readline()
#             temp = json.loads(json_line)
#             return temp.keys()
#
#         col_name = read_column_names(f_list[0])
#
#         temp = []
#         error = []
#         for fn in f_list[1:]:
#             with open(fn, encoding='utf-8') as f:
#                 for i, line in enumerate(f):
#                     try:
#                         temp.append(json.loads(line.replace('\\\\"', '\\"')))
#                     except Exception as e:
#                         error.append([e, line])
#         raw_df = pd.DataFrame(data=temp, columns=col_name)
#
#         class_count = raw_df['ksic'].value_counts()
#         class_count2 = class_count[class_count >= 500]
#
#         raw_df2 = raw_df.loc[raw_df['ksic'].isin(class_count2.keys())].copy()
#
#         ksic_label = raw_df2['ksic'].unique()
#         ksic_index_dict = {i: label for i, label in enumerate(ksic_label)}
#         ksic_label_dict = {ksic_index_dict[key]: key for key in ksic_index_dict.keys()}
#
#         # ksic_index_dict
#         with open('.cache/label_ksic.pickle', 'wb') as f:
#             pickle.dump(ksic_index_dict, f)
#         with open('.cache/ksic_label.pickle', 'wb') as f:
#             pickle.dump(ksic_label_dict, f)
#
#         raw_df2['label'] = raw_df2['ksic'].map(ksic_label_dict)
#         train_input, test_input = train_test_split(raw_df2, random_state=15, test_size=0.2, stratify=raw_df2['ksic'],
#                                                    shuffle=True)
#         train_input, val_input = train_test_split(train_input, random_state=15, test_size=0.15,
#                                                   stratify=train_input['ksic'], shuffle=True)
#
#         train_input.to_csv('.cache/train_input.csv', encoding='utf-8', mode='w', index=False)
#         val_input.to_csv('.cache/val_input.csv', encoding='utf-8', mode='w', index=False)
#         test_input.to_csv('.cache/test_input.csv', encoding='utf-8', mode='w', index=False)
#         print(time.strftime('%l:%M%p %Z on %b %d, %Y'))  # ' 1:36PM EDT on Oct 18, 2010'
#         print('Loading json files and saving "train_input.csv" completed')
#     return ksic_index_dict, ksic_label_dict, train_input, val_input, test_input
#
#
# ksic_index_dict, ksic_label_dict, train_input, val_input, test_input = load_datasets()


def load_train_dataset():
    try:
        train_ds = nlp.data.TSVDataset('.cache/train_ds.tsv', encoding='utf-8',
                                       field_indices=[0, 1], num_discard_samples=1)
        val_ds = nlp.data.TSVDataset('.cache/val_ds.tsv', encoding='utf-8',
                                     field_indices=[0, 1], num_discard_samples=1)
        test_ds = nlp.data.TSVDataset('.cache/test_ds.tsv', encoding='utf-8',
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

        train_ds.to_csv('.cache/train_ds.tsv', encoding='utf-8', mode='w', index=False, sep='\t')
        val_ds.to_csv('.cache/val_ds.tsv', encoding='utf-8', mode='w', index=False, sep='\t')
        test_ds.to_csv('.cache/test_ds.tsv', encoding='utf-8', mode='w', index=False, sep='\t')
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
    # data_train_id = np.load('.cache/ksic_data_train_id.npy', allow_pickle=True)
    # data_test_id = np.load('.cache/ksic_data_test_id.npy', allow_pickle=True)
    with open(".cache/ksic_data_train_id.pickle", "rb") as fr:
        data_train_id = pickle.load(fr)
    with open(".cache/ksic_data_test_id.pickle", "rb") as fr:
        data_test_id = pickle.load(fr)
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
    data_test_id = BERTDataset(test_ds, 0, 1, tok, max_len, True, False)
    # np.save('.cache/ksic_data_test_id.npy', data_test_id, allow_pickle=True)
    with open('.cache/ksic_data_test_id.pickle', "wb") as fw:
        pickle.dump(data_test_id, fw)
    # data_train_id = BERTDataset(train_ds.to_numpy(), 0, 1, tok, max_len, True, False)
    # data_test_id = BERTDataset(test_ds.to_numpy(), 0, 1, tok, max_len, True, False)
    print(time.strftime('%l:%M%p %Z on %b %d, %Y'))  # ' 1:36PM EDT on Oct 18, 2010'
    print('Completed to make BERTDataset')

train_dataloader = torch.utils.data.DataLoader(data_train_id, batch_size=batch_size, num_workers=8, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(data_test_id, batch_size=batch_size, num_workers=8, shuffle=True)


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)


model = BERTClassifier(bertmodel, num_classes=500, dr_rate=0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


def calc_accuracy(X, y):
    max_vals, max_indices = torch.max(X, 1)
    acc = (max_indices == y).sum().data.cpu().numpy()/max_indices.size()[0]
    return acc


for e in range(num_epochs):
    print(time.strftime('%l:%M%p %Z on %b %d, %Y'), ': starting epoch ', e)
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader),
                                                                        total=len(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("\ntime: {}, epoch {} batch id {}\n loss {} train acc {}".format(time.strftime('%l:%M%p'),
                                                                                   e+1, batch_id+1,
                                                                                   loss.data.cpu().numpy(),
                                                                                   train_acc / (batch_id+1)))
            # torch.save(model, model_path)  # 전체 모델 저장
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader),
                                                                        total=len(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    print(time.strftime('%l:%M%p %Z on %b %d, %Y'))
    # torch.save(model, model_path)  # 전체 모델 저장
    torch.save(model, os.path.join(model_path, 'KSIC_KoBERT.pt'))  # 전체 모델 저장
    torch.save(model.state_dict(), os.path.join(model_path, 'KSIC_model_state_dict.pt'))  # 모델 객체의 state_dict 저장
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(model_path, 'all.tar'))
    # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능,
    # https://velog.io/@dev-junku/KoBERT-%EB%AA%A8%EB%8D%B8%EC%97%90-%EB%8C%80%ED%95%B4

print(time.strftime('%l:%M%p %Z on %b %d, %Y'))

