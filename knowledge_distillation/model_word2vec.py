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


# modified from the tutorial: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
class EmbeddingBagModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(EmbeddingBagModel, self).__init__()
        # self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim)
        self.linear = torch.nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, text, text_lengths=None):
        embedded = self.embedding(text)
        output = self.linear(embedded)
        return output

class Word2vecModelling():
    def process_text(filename_train, filename_test, BATCH_SIZE=32):
        # create field object
        TEXT = torchtext.legacy.data.Field(sequential=True,
            init_token='(sos)',  # start of sequence
            eos_token='(eos)',   # replace parens with less, greater
            lower=True,
            tokenize=torchtext.legacy.data.utils.get_tokenizer("basic_english"),)
        LABEL = torchtext.legacy.data.Field(sequential=False,
            use_vocab=False,
            unk_token=None,
            is_target=True)

        # read csv file and create dataset
        train_dataset = torchtext.legacy.data.TabularDataset(path=filename_train, format='csv', skip_header=True,
                                    fields=[('text', TEXT), ('label', LABEL)])
        test_dataset = torchtext.legacy.data.TabularDataset(path=filename_test, format='csv', skip_header=True,
                                    fields=[('text', TEXT), ('label', LABEL)])

        # pretrained: https://torchtext.readthedocs.io/en/latest/vocab.html
        TEXT.build_vocab(train_dataset, vectors='glove.6B.50d')
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
        # train_iter, test_iter = torchtext.legacy.data.Iterator.splits((train_dataset, test_dataset), batch_sizes=(BATCH_SIZE, BATCH_SIZE), repeat=False, sort=False)

        return train_iter, test_iter

    def generate_model(num_classes):
        # # set up device 
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

        vocab_size = 30000 # 20000 # len(vocab)
        emsize = 128 # 64
        print (num_classes, vocab_size)
        embedding_model = EmbeddingBagModel(vocab_size, emsize, num_classes)#.to(device)
        print(embedding_model)
        return embedding_model

    def train_embedding(embedding_model, device, train_iter, num_classes, num_epochs=20):
        # # set up device 
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        print("device", device)
        # Pass model to GPU
        embedding_model = embedding_model.to(device) 
        
        torch.autograd.set_detect_anomaly(True)

        # training
        criterion = torch.nn.CrossEntropyLoss()
        LR = 5 
        optimizer = torch.optim.SGD(embedding_model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        start = time.time()
        for epoch in range(num_epochs):
            all_loss = 0
            for idx, data in enumerate(train_iter):
                label = data.label.to(device)
                if len(data.text)==2:   
                    text = data.text[0].to(device)
                else:
                    text = data.text.T.to(device)

                batch_loss = 0
                embedding_model.zero_grad()
                predited_label = embedding_model(text)
                predited_label = F.log_softmax(predited_label, dim=1) 
                
                batch_loss = criterion(predited_label, label)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), 0.1)
                optimizer.step()
                all_loss += batch_loss.item()
            print("epoch: ", epoch, "\t" , "loss: ", all_loss)
        end = time.time()
        print ("time : ", end - start)

        # # release GPU memory
        # torch.cuda.empty_cache()        

        return embedding_model, device

    def predict_embedding(model, device, train_iter, title_name=None, num_classes=2):
        # # Set up GPU
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      

        answer = []
        prediction = []
        start = time.time()
        with torch.no_grad():
            for data in (train_iter):
                label_tensor = data.label.to(device)
                if len(data.text)==2:      
                    text_tensor = data.text[0].to(device)
                else:
                    text_tensor = data.text.T.to(device)

                score = model(text_tensor)
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