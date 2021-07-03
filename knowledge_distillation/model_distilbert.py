import time 

from sklearn.metrics import classification_report

import torch
from torch import nn
import torch.nn.functional as F
import torchtext
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim

from knowledge_distillation.model_evaluation import plot_confusion_matrix_heatmap, plot_roc_auc

# Creating the customised model, by adding a dense layer on top of distilbert to get the final output for the model. 
class DistilBERTClassifierNetwork(torch.nn.Module):
    def __init__(self, num_classes):
        super(DistilBERTClassifierNetwork, self).__init__()
        self.distil_bert = AutoModel.from_pretrained("distilbert-base-uncased")
        # self.pre_classifier = torch.nn.Linear(768, 768)
        # self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, num_classes)

        # weight initialisation
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.normal_(self.classifier.bias, 0)

    def forward(self, input_ids, text_lengths=None):
        output_1 = self.distil_bert(input_ids=input_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        # pooler = self.pre_classifier(pooler)
        # pooler = torch.nn.Tanh()(pooler)
        # pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        # output = F.log_softmax(output, dim=1)
        return  output

class DistilBERTModelling():
    def get_tokenizer(select_pretrained_model):
        # #### English
        # select_pretrained_model = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(select_pretrained_model)
        # distilbert_model = AutoModel.from_pretrained(select_pretrained_model)

        # #### Japanese
        # tokenizer_jap = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        # distilbert_jap_model = AutoModel.from_pretrained("bandainamco-mirai/distilbert-base-japanese")

        # print(distilbert_model)
        return tokenizer#, distilbert_model

    def process_text(tokenizer, filename_train, filename_test, BATCH_SIZE=256): 
        # create Field object
        def text_tokenizer(text):
            return tokenizer.encode(text, return_tensors='pt')[0]

        TEXT = torchtext.legacy.data.Field(sequential=True, tokenize=text_tokenizer, use_vocab=False, lower=False,
                                    include_lengths=True, batch_first=True, pad_token=0, unk_token=0, eos_token=0)
        LABEL = torchtext.legacy.data.Field(sequential=False, use_vocab=False)

        # read csv file and create dataset
        train_dataset = torchtext.legacy.data.TabularDataset(path=filename_train, format='csv', skip_header=True,
                                    fields=[('text', TEXT), ('label', LABEL)])
        test_dataset = torchtext.legacy.data.TabularDataset(path=filename_test, format='csv', skip_header=True,
                                    fields=[('text', TEXT), ('label', LABEL)])

        # # check data
        # for train in train_dataset:
        #     print (train.text, train.label)
        #     break
        TEXT.build_vocab(train_dataset)
        LABEL.build_vocab(train_dataset)
        print("Size of TEXT vocabulary:",len(TEXT.vocab))
        print("Size of LABEL vocabulary:",len(LABEL.vocab))

        # seperate batches
        train_iter, test_iter= torchtext.legacy.data.BucketIterator.splits(
            (train_dataset, test_dataset), 
            batch_size = BATCH_SIZE,
            sort_key = lambda x: len(x.text),
            sort_within_batch=True)
        # train_iter, test_iter = torchtext.legacy.data.Iterator.splits((train_dataset, test_dataset), batch_sizes=(BATCH_SIZE, BATCH_SIZE), repeat=False, sort=False)
        # print (len(train_iter))

        # # check data
        # for train in train_iter:
        #     print (train.text, train.label)
        #     break
        return train_iter, test_iter

    def generate_model(num_classes):
        # get classification model
        distil_classifier = DistilBERTClassifierNetwork(num_classes)

        # #### fine-tuning
        # Turn OFF all paramters
        for param in distil_classifier.parameters():
            param.requires_grad = False

        # Turn ON the last layer parameter
        # .transfomer.layer[-1] for DistilBERT or .encoder.layer[-1] for BERT-base
        for param in distil_classifier.distil_bert.transformer.layer[-1].parameters():
            param.requires_grad = True

        # Turn ON the classification part
        for param in distil_classifier.classifier.parameters():
            param.requires_grad = True

        return distil_classifier

    def train_distilbert(distil_classifier, device, train_iter, num_classes, num_epochs=10):
        # # Set up GPU
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print("device", device)
        # Pass model to GPU
        distil_classifier.to(device)

        # Small lr value for pretrained layer and bigger value for the last layer
        optimizer = optim.Adam([
            {'params': distil_classifier.distil_bert.transformer.layer[-1].parameters(), 'lr': 5e-5},
            {'params': distil_classifier.classifier.parameters(), 'lr': 1e-4}
        ])

        loss_function = nn.NLLLoss()
        losses = []

        start = time.time()
        # Set up epoch 
        for epoch in range(num_epochs):
            all_loss = 0
            for idx, batch in enumerate(train_iter):
                # print ("iter: ", idx)
                batch_loss = 0
                distil_classifier.zero_grad()
                input_ids = batch.text[0].to(device)
                label_ids = batch.label.to(device)
                out = distil_classifier(input_ids)
                out = F.log_softmax(out, dim=1)

                batch_loss = loss_function(out, label_ids)
                batch_loss.backward()
                optimizer.step()
                all_loss += batch_loss.item()
            print("epoch: ", epoch, "\t" , "loss: ", all_loss)

        end = time.time()
        print ("time : ", end - start)
        # # release GPU memory
        # torch.cuda.empty_cache()        
        return distil_classifier, device

    def predict_distilbert(distil_classifier, device, batch_iter, title_name=None, num_classes=2):
        # # Set up GPU
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      

        answer = []
        prediction = []
        start = time.time()
        with torch.no_grad():
            for batch in batch_iter:

                text_tensor = batch.text[0].to(device)
                label_tensor = batch.label.to(device)

                score = distil_classifier(text_tensor)
                _, pred = torch.max(score, 1)

                prediction += list(pred.cpu().numpy())
                answer += list(label_tensor.cpu().numpy())
        
        end = time.time()
        print ("time : ", end - start)

        # print classification report
        print(classification_report(prediction, answer))
        print("predicted label: ", set(prediction))

        # model evaluation
        plot_confusion_matrix_heatmap(answer, prediction, "{}".format(title_name))
        plot_roc_auc(answer, prediction, title_name, num_classes)
            
        # # release GPU memory
        # torch.cuda.empty_cache()        
        return    