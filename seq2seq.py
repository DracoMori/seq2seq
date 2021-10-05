# -*- encoding: utf-8 -*-
'''
@Time    :   2021/09/15 09:44:42
@Author  :   流氓兔233333 
@Version :   1.0
@Contact :   中英翻译尝试   中文(BERT)->英文    batch_size 未解决, 不同batch_sizes下outputs存储问题
'''

data_path = './raw_data/'
save_path = './temp_results/'
cache_dir = 'D:/NLP/tokenizer_cache'


import numpy as np
import os, pickle
from tqdm import tqdm
import random

from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim


with open(data_path+'train.en', 'r', encoding='utf-8') as f:   
    data_en = f.read()

with open(data_path+'train.zh', 'r', encoding='utf-8') as f:   
    data_zh = f.read()

def data_pair_create(data_zh, data_en, num=1000):
    zh_list = data_zh.split('\n')
    en_list = data_en.split('\n')
    
    sample_list = random.sample(range(len(zh_list)), num)
    
    data_pair = []
    for i in tqdm(sample_list):
        data_pair.append((zh_list[i], en_list[i]))
    
    return data_pair

data_pair = data_pair_create(data_zh, data_en, num=10000)
data_length = [len(x) for x,_ in data_pair]
max(data_length)

# set seed
def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed = 23333
set_seed(seed)

# train test spilt
def train_test_spilt(data, test_size=0.4):
    random.shuffle(data)
    data_trn = data[:int(len(data)*(1-0.4))]
    data_val = data[int(len(data)*(1-0.4)):]
    return data_trn, data_val

data_trn, data_val = train_test_spilt(data_pair, test_size=0.2)


# tokenizer 中bert-base-chinese 英bert-base-uncased   maxlen=64
class CustomDataset(Data.Dataset):
    def __init__(self, data, maxlen, model_zh='bert-base-chinese', model_en='bert-base-uncased'):
        self.data = data  # pandas dataframe

        #Initialize the tokenizer
        self.tokenizer_zh = AutoTokenizer.from_pretrained(model_zh, cache_dir=cache_dir) 
        self.tokenizer_en = AutoTokenizer.from_pretrained(model_en, cache_dir=cache_dir) 

        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        # 根据自己输入data的格式修改
        text_zh = self.data[index][0]
        text_eh = self.data[index][1]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_zh = self.tokenizer_zh(text_zh,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,       # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        encoded_en = self.tokenizer_en(text_eh,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,       # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids_zh = encoded_zh['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks_zh = encoded_zh['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids_zh = encoded_zh['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        token_ids_en = encoded_en['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks_en = encoded_en['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids_en = encoded_en['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        return token_ids_zh, attn_masks_zh, token_type_ids_zh, token_ids_en, attn_masks_en, token_type_ids_en 


tokenizer_en = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir) 
tokenizer_en.vocab_size

batch_size = 16
dataset_trn = CustomDataset(data_trn, 32)
loader_trn = Data.DataLoader(dataset_trn, batch_size, False)

dataset_val = CustomDataset(data_val, 32)
loader_val = Data.DataLoader(dataset_val, batch_size, False)





# seq2seq model    input.dim=len(SRC.vocab)
class Encoder(nn.Module):
    def __init__(self, dec_hid_dim, model_name='bert-base-chinese'):
        super(Encoder, self).__init__()
        self.bert_embedding = BertModel.from_pretrained(model_name, output_hidden_states=True, 
                    return_dict=True, cache_dir=cache_dir)
        self.fc = nn.Linear(768, dec_hid_dim)
    
    def forward(self, input_id, attn_masks, token_type):
        outputs = self.bert_embedding(input_ids=input_id, token_type_ids=token_type, attention_mask=attn_masks) 
        enc_output = outputs.last_hidden_state # [bs, seq_len, 768]
        # 我只要 [CLS]上的 hidden，表示整个句子的信息  [bs, 768]
        # s 包含整个句子的信息, fc作用是将s的维度转换成dec_hid_dim 的维度
        s = torch.tanh(self.fc(enc_output[:,0,:]))
        return enc_output, s

# encoder = Encoder(dec_hid_dim=768)

# enc_output, s = encoder(batch[0], batch[1], batch[2])
# enc_output.shape, s.shape

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hid_dim+dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        '''
        s [bs, dec_hid_dim]    
        enc_output [bs, seq_len, enc_hid_dim=768]
        '''
        bs = enc_output.shape[0]
        seq_len = enc_output.shape[1]

        # 把 s 扩充到和 enc_output 相同的维度
        s = s.unsqueeze(1).repeat(1, seq_len, 1)

        # s he enc_output 拼接, 并送入 atten 层计算 注意力得分
        # torch.cat((s, enc_output) [bs, 1, dec_hid_dim + enc_hid_dim]
        # energy [bs, seq_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2))) 

        # attention [bs, src_len] 
        # self.v的作用是dim=2上的维度转为1
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    # src, src_mask, src_typeid = batch[0], batch[1], batch[2]
    # trg = batch[3]
    def forward(self, src, src_mask, src_typeid, trg, teacher_forcing_ratio=0.5):
        '''
        src = [bs, src_len]  trg = [bs, trg_len]
        '''
        bs = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # teacher to store decoder output
        # [trg_len, bs, trg_vocab_size]
        outputs = torch.zeros(bs, trg_len, trg_vocab_size).to(device)

        # s 为 decoder rnn 的 hidden state
        enc_output, s = self.encoder(src, src_mask, src_typeid)
        # first input decoder is the <CLS> tokens
        dec_input = trg[:, 0]    # [bs]
        
        # <CLS> 101 <SEP> 102
        dec_output = torch.zeros(bs, trg_vocab_size).to(device)
        dec_output[:, 101] =  torch.ones(bs)
        outputs[:,0,:] = dec_output

        for t in range(1, trg_len):

            dec_output, s = self.decoder(dec_input, s, enc_output)

            outputs[:,t,:] = dec_output  # 记录 decoder的 output  [bs, ]

            # 判断 是否进行 teacher forcing, 若否，则使用decoder 的输出作为下一个时刻的输入
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(1)  # decoder 的输出

            dec_input = trg[:,t] if teacher_force else top1

        return outputs


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim    # trg_vocan_size
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # rnn 的输入是 enc_out 和 decoder的embedding
        self.rnn = nn.GRU(enc_hid_dim+emb_dim, dec_hid_dim, batch_first=True)
        # enc_doutput + 本身 rnn输出 + 自身embedding -> out_dim
        # self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.fc_out = nn.Linear(ec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, s, enc_output):
        '''
        dec_input = [bs]
        s = [bs, 768]
        enc_output = [bs, src_len, 768]
        '''

        dec_input = dec_input.unsqueeze(1)  # [bs, 1]
        embedded = self.dropout(self.embedding(dec_input)) # [bs, 1, dec_emb_dim]

        # attention 的 input 是 上一个时刻的 s 和 enc_output
        # s 包含了 decoder 上一时刻输入 和 encoder 各token之间的关系信息
        a = self.attention(s, enc_output).unsqueeze(1)# [bs, 1, src_len]

        # a=[bs, 1, src_len] x b=[bs, src_len, 768]  
        # c 包含了 整个输入的信息( enc_output 的 attention 的 weighted sum)
        c = torch.bmm(a, enc_output) # [bs, 1, 768]

        # rnn 的输入是 c 和 decoder的embedding 计算输入rnn， c是包含了注意力机制的 encoder 全部的信息
        # c是由 s 作为上一时刻 hidden_state 计算attention 后 和 enc_output 计算 weighted sum得到
        rnn_input = torch.cat((embedded, c), dim=2) # [bs, 1, dec_emb_dim+768]

        # dec_output = [bs, 1, dec_hid_dim]  
        # dec_hidden = [n_layers*num_direstiopns, bs, dec_hid_dim]
        # dec_hidden.squeeze(0) 就是下一时刻的 s 

        # rnn hidden state 的输入维度是 [n_layer*direction, bs, hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(1).transpose(0,1))    

        embedded = embedded.squeeze(1) # [bs, enc_embed_dim]
        dec_output = dec_output.squeeze(1) # [bs, dec_hid_dim]
        c = c.squeeze(1) # [bs, 768]

        # pred 预测下一个词  
        # 输入为 decoder rnn的输出, c, decoder 的 embedding 
        # pred = self.fc_out(torch.cat((dec_output, c, embedded), dim=1))
        pred = self.fc_out(dec_output)

        return pred, dec_hidden.squeeze(0)







output_dim = tokenizer_en.vocab_size
dec_embed_dim = 256
enc_hid_dim = 768
dec_hid_dim = 768
dec_dropout = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

attn = Attention(enc_hid_dim, dec_hid_dim)
encoder = Encoder(dec_hid_dim)
decoder = Decoder(output_dim, dec_embed_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn)

model = seq2seq(encoder, decoder, device)

# src, src_mask, src_typeid, trg = batch[0], batch[1], batch[2], batch[3]

batch = next(iter(loader_trn))
batch[0].shape, batch[3].shape
outputs = model(batch[0], batch[1], batch[2], batch[3])
outputs.shape  # [bs, seq_len, output_dim]

w = outputs.argmax(2)
s = tokenizer_en.convert_ids_to_tokens(w[4, :])


# training model
output, target, mask, num_labels = outputs, batch[3], batch[4], outputs.shape[2]
def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    output.shape
    active_logits = output.contiguous().view(-1, num_labels)
    active_labels = torch.where(active_loss, target.view(-1), 
                               torch.tensor(lfn.ignore_index).type_as(target))
    loss = lfn(active_logits, active_labels)

    return loss


def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    },'./seq2seq.pth')
    print('The best model has been saved')

def train_eval(model, criterion, optimizer, loader_trn, loader_val, epochs=2, continue_=True):
    if continue_:
        try:
            checkpoint = torch.load('./seq2seq.pth', map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('-----Continue Training-----')
        except:
            print('No Pretrained model!')
            print('-----Training-----')
    else:
        print('-----Training-----')
    
    model = model.to(device)
    loss_his = []
    for epoch in range(epochs):
        model.train()
        print('eopch: %d/%d'% (epoch, epochs))
        for _, batch in enumerate(tqdm(loader_trn)):
            batch = [x.to(device) for x in batch]
            
            optimizer.zero_grad()

            output = model(batch[0], batch[1], batch[2], batch[3])
            loss = loss_fn(output, batch[3], batch[4], output.shape[2])
            loss_his.append(loss.item())

            loss.backward()
            optimizer.step()
            
        if epoch % 1 == 0:
            print(loss.item())
            eval(model, optimizer, loader_val)
    
    return loss_his


best_score = 0.0
def eval(model, optimizer, loader_val):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader_trn)):
            batch = [x.to(device) for x in batch]
            output = model(batch[0], batch[1], batch[2], batch[3])
            output = output.detach().cpu().numpy()

            pred = np.argmax(output, axis=2)
            
            label_ids = batch[3].detach().cpu().numpy()
            masked = batch[4].detach().cpu().numpy()
           
            scores = []
            for p, r, m in zip(pred, label_ids, masked):
                idx_ = np.where(m==1)[0]
                scores.append(1.0-(len(set(r[idx_]).difference(set(p[idx_]))) / len(r[idx_])))
    
    print("Validation Accuracy: {}".format(np.mean(scores)))
    global best_score
    if best_score < np.mean(scores):
        best_score = np.mean(scores)
        save(model, optimizer)




# ============================================================
# Encoder 不适用 bert
class Encoder(nn.Module):
    def __init__(self, input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.5):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, enc_emb_dim)
        self.rnn = nn.GRU(enc_emb_dim, enc_hid_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hid_dim*2, dec_hid_dim)
    
    def forward(self, input_id, attn_masks, token_type):
        '''
        input_id [bs, src_len]
        '''
        embedd = self.dropout(self.embedding(input_id))  # [bs, src_len, enc_emb_dim]
        # enc_output = [bs, src, enc_hid_dim*2]
        # enc_hidden = [n_layers*direction, bs, enc_hid_dim]
        enc_output, enc_hidden = self.rnn(embedd) 

        s = torch.tanh(self.fc(torch.cat((enc_hidden[-2,:,:], enc_hidden[-1,:,:]), dim=1)))
        return enc_output, s 


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hid_dim*2+dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        '''
        s [bs, dec_hid_dim]    
        enc_output [bs, seq_len, enc_hid_dim=768]
        '''
        bs = enc_output.shape[0]
        seq_len = enc_output.shape[1]

        # 把 s 扩充到和 enc_output 相同的维度
        s = s.unsqueeze(1).repeat(1, seq_len, 1)

        # s he enc_output 拼接, 并送入 atten 层计算 注意力得分
        # torch.cat((s, enc_output) [bs, 1, dec_hid_dim + enc_hid_dim]
        # energy [bs, seq_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2))) 

        # attention [bs, src_len] 
        # self.v的作用是dim=2上的维度转为1
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    # src, src_mask, src_typeid = batch[0], batch[1], batch[2]
    # trg = batch[3]
    def forward(self, src, src_mask, src_typeid, trg, teacher_forcing_ratio=0.5):
        '''
        src = [bs, src_len]  trg = [bs, trg_len]
        '''
        bs = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # teacher to store decoder output
        # [trg_len, bs, trg_vocab_size]
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)

        # s 为 decoder rnn 的 hidden state
        enc_output, s = self.encoder(src, src_mask, src_typeid)
        
        # first input decoder is the <CLS> tokens
        dec_input = trg[:, 0]
        for t in range(1, trg_len):

            dec_output, s = self.decoder(dec_input, s, enc_output)
            outputs[t] = dec_output  # 记录 decoder的 output  [bs, ]

            # 判断 是否进行 teacher forcing, 若否，则使用decoder 的输出作为下一个时刻的输入
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(1)  # decoder 的输出

            dec_input = trg[t] if teacher_force else top1

            return outputs.transpose(0,1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim    # trg_vocan_size
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # rnn 的输入是 enc_out 和 decoder的embedding
        self.rnn = nn.GRU(enc_hid_dim*2+emb_dim, dec_hid_dim, batch_first=True)
        # enc_doutput + 本身 rnn输出 + 自身embedding -> out_dim
        self.fc_out = nn.Linear(enc_hid_dim*2 + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, s, enc_output):
        '''
        dec_input = [bs]
        s = [bs, 768]
        enc_output = [bs, src_len, 768]
        '''

        dec_input = dec_input.unsqueeze(1)  # [bs, 1]
        embedded = self.dropout(self.embedding(dec_input)) # [bs, 1, dec_emb_dim]

        # attention 的 input 是 上一个时刻的 s 和 enc_output
        # s 包含了 decoder 上一时刻输入 和 encoder 各token之间的关系信息
        a = self.attention(s, enc_output).unsqueeze(1) # [bs, 1, src_len]

        # a=[bs, 1, src_len] x b=[bs, src_len, 768]  
        # c 包含了 整个输入的信息( enc_output 的 attention 的 weighted sum)
        c = torch.bmm(a, enc_output) # [bs, 1, 768]

        # rnn 的输入是 c 和 decoder的embedding 计算输入rnn， c是包含了注意力机制的 encoder 全部的信息
        # c是由 s 作为上一时刻 hidden_state 计算attention 后 和 enc_output 计算 weighted sum得到
        rnn_input = torch.cat((embedded, c), dim=2) # [bs, 1, dec_emb_dim+768]

        # dec_output = [bs, 1, dec_hid_dim]  
        # dec_hidden = [n_layers*num_direstiopns, bs, dec_hid_dim]
        # dec_hidden.squeeze(0) 就是下一时刻的 s 

        # rnn hidden state 的输入维度是 [n_layer*direction, bs, hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(1).transpose(0,1))    

        embedded = embedded.squeeze(1) # [bs, enc_embed_dim]
        dec_output = dec_output.squeeze(1) # [bs, dec_hid_dim]
        c = c.squeeze(1) # [bs, 768]

        # pred 预测下一个词  
        # 输入为 decoder rnn的输出, c, decoder 的 embedding 
        pred = self.fc_out(torch.cat((dec_output, c, embedded), dim=1))

        return pred, dec_hidden.squeeze(0)





tokenizer_zh = AutoTokenizer.from_pretrained('bert-base-chinese', cache_dir=cache_dir) 
tokenizer_zh.vocab_size

input_dim = tokenizer_zh.vocab_size
output_dim = tokenizer_en.vocab_size
enc_embed_dim = 256
dec_embed_dim = 256
enc_hid_dim = 512
dec_hid_dim = 512
enc_dropout = 0.5
dec_dropout = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

attn = Attention(enc_hid_dim, dec_hid_dim)
encoder = Encoder(input_dim, enc_embed_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
decoder = Decoder(output_dim, dec_embed_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn)
model = seq2seq(encoder, decoder, device)


batch = next(iter(loader_trn))
outputs = model(batch[0], batch[1], batch[2], batch[3])
outputs.shape  # [bs, seq_len, output_dim]










