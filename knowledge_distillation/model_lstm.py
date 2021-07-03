import time
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchtext
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

from knowledge_distillation.model_evaluation import plot_confusion_matrix_heatmap, plot_roc_auc


class LSTMModel(torch.nn.Module):
    # https://www.kaggle.com/swarnabha/pytorch-text-classification-torchtext-lstm
    # https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
    # https://qiita.com/m__k/items/841950a57a0d7ff05506
    # https://www.programmersought.com/article/47054517678/
    
    #Define all layers
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):
        
        super().__init__()          
        # embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        #lstm layer
        self.lstm = torch.nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        #Fully connected layer
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
                
    def forward(self, text, text_lengths):
        # print("text", text.size, text)
        #text = [batch size,sent_length]
        embedded = self.embedding(text)
        #embedded = [batch size, sent_len, emb dim]
        # print("embedded", embedded)

        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(
          embedded, text_lengths.cpu(), batch_first=True)
        # print("packed_embedded",packed_embedded)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # print("packed_output", packed_output)

        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]
        
        #Connect the final forward and reverse hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        #hidden = [batch size, hid dim * num directions]

        dense_outputs=self.fc(hidden)

        # outputs = F.log_softmax(dense_outputs, dim=1)
        return dense_outputs
 
class LstmModelling():

    def process_text(filename_train, filename_test, BATCH_SIZE=32):
        TEXT = torchtext.legacy.data.Field(tokenize = 'spacy', include_lengths = True)
        LABEL = torchtext.legacy.data.LabelField(dtype = torch.float)
        # read csv file and create dataset
        train_dataset = torchtext.legacy.data.TabularDataset(path=filename_train, format='csv', skip_header=True,
                                    fields=[('text', TEXT), ('label', LABEL)])
        test_dataset = torchtext.legacy.data.TabularDataset(path=filename_test, format='csv', skip_header=True,
                                    fields=[('text', TEXT), ('label', LABEL)])

        # pretrained: https://torchtext.readthedocs.io/en/latest/vocab.html
        TEXT.build_vocab(train_dataset, 
                 min_freq=3,
                 vectors = 'glove.6B.200d')
        # TEXT.build_vocab(train_dataset, test_dataset, test_dataset, vectors='glove.6B.200d')
        LABEL.build_vocab(train_dataset)
        print("Size of TEXT vocabulary:",len(TEXT.vocab))
        print("Size of LABEL vocabulary:",len(LABEL.vocab))
        print(TEXT.vocab.freqs.most_common(10))  
        print(TEXT.vocab.stoi) 

        # seperate batches
        train_iter, test_iter= torchtext.legacy.data.BucketIterator.splits(
            (train_dataset, test_dataset), 
            batch_size = BATCH_SIZE,
            sort_key = lambda x: len(x.text),
            sort_within_batch=True)

        # train_iter, test_iter = torchtext.legacy.data.Iterator.splits((train_dataset, test_dataset), 
        # batch_sizes=(BATCH_SIZE, BATCH_SIZE), repeat=False, sort=False)
        return train_iter, test_iter

    def generate_model(num_classes):
        # # set up device 
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

        #Define hyperparameters
        size_of_vocab = 30000 #20457 #1110 # 20000 # 1110 #len(TEXT.vocab)
        embedding_dim = 128 # 100
        num_hidden_nodes = 16 # 32
        num_output_nodes = num_classes # 1
        num_layers = 2
        bidirection = False # True
        dropout = 0.0 # 0.2
        
        #Instance model
        lstm_model = LSTMModel(size_of_vocab, embedding_dim, 
        num_hidden_nodes,num_output_nodes, num_layers, 
        bidirectional = bidirection, dropout = dropout)#.to(device)
        print(lstm_model)
        return lstm_model

    def train(lstm_model, device, train_iter, num_classes, num_epochs=20, token_flag="lstm"):
        # # set up device 
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        # print("device", device)
        # Pass model to GPU
        lstm_model = lstm_model.to(device)  

        # training
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
        start = time.time()
        for epoch in range(num_epochs):
            all_loss = 0
            for idx, data in enumerate(train_iter):
                label = data.label.to(device)
                label = torch.tensor(label, dtype=torch.long)
                if token_flag=="distilbert":
                    text = data.text[0].to(device)
                else:
                    text = data.text[0].T.to(device)
                text_lengths = data.text[1].to(device)
                text_lengths[text_lengths<=0] = 1

                batch_loss = 0
                lstm_model.zero_grad()
                predited_label = lstm_model(text, text_lengths)
                predited_label = F.log_softmax(predited_label, dim=1)

                batch_loss = criterion(predited_label, label)
                batch_loss.backward()
                optimizer.step()
                all_loss += batch_loss.item()
            print("epoch: ", epoch, "\t" , "loss: ", all_loss)
        end = time.time()
        print ("time : ", end - start)

        # # release GPU memory
        # torch.save(lstm_model.state_dict(), PATH)
        # torch.cuda.empty_cache()        

        return lstm_model, device

    def predict(model, device, train_iter, token_flag="lstm", title_name=None, num_classes=2):
        # # Set up GPU
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      

        answer = []
        prediction = []
        start = time.time()
        with torch.no_grad():
            for data in (train_iter):
                label_tensor = data.label.to(device)
                if token_flag=="distilbert":
                    text = data.text[0].to(device)
                else:
                    text = data.text[0].T.to(device)
                text_lengths = data.text[1].to(device)
                text_lengths[text_lengths<=0] = 1
                score = model(text, text_lengths)
                _, pred = torch.max(score, 1)

                prediction += list(pred.cpu().numpy())
                answer += list(label_tensor.cpu().numpy())
        
        end = time.time()
        print ("time : ", end - start)

        # print classification report
        print(classification_report(prediction, answer))
        print("predicted label: ", set(prediction))

        # model evaluation
        plot_confusion_matrix_heatmap(answer, prediction, "confusion matrix {}".format(title_name))
        plot_roc_auc(answer, prediction, title_name, num_classes)

        # # release GPU memory
        # torch.cuda.empty_cache()        
        return 