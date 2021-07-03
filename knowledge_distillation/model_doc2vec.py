import time
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn import utils
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchtext
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

from knowledge_distillation.model_evaluation import plot_confusion_matrix_heatmap, plot_roc_auc


class SimpleNeuralNetwork(torch.nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.linear = torch.nn.Linear(vocab_size, num_labels)

    def forward(self, vector):
        output = self.linear(vector)
        # output = F.log_softmax(output, dim=1)
        return output

class Doc2vecModelling:

    def select_data(select_data_df_train, select_data_df_test):
        # get data
        train_text = select_data_df_train["text"].values.tolist()
        train_label = select_data_df_train["label"].values

        test_text = select_data_df_test["text"].values.tolist()
        test_label = select_data_df_test["label"].values.tolist()
        return train_text, train_label, test_text, test_label

    def generate_doc2vec_model(data_path_train, train_text, num_epochs):
        # tag text
        train_text_tag = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_text)]
        train_text_tag[0:5]

        # %time
        # #### model
        # dm=0: distributed bag of words (PV-DBOW)
        # dm=1: distributed memory (PV-DM)
        # min_count: ignores all words with total frequency lower than this
        # negative: specifies how many “noise words” should be drawn.
        # hs: and negative is non-zero, negative sampling will be used.
        # sample: the threshold for configuring which higher-frequency words are randomly down sampled.
        # workers: use these many worker threads to train the model (=faster training with multicore machines)

        model = Doc2Vec(train_text_tag, dm=0, vector_size=24, window=5, min_count=1, workers=4, epochs=100)
        # model = Doc2Vec(train_text_tag, dm=0, vector_size=24, window=5, min_count=1, workers=4, epochs=num_epochs)
        filename = data_path_train + "doc2vec_model.model"
        # filename = "trained_models/" + "doc2vec_model.model"
        model.save(filename)
        return model

    def process_data(model, train_text, train_label, test_text, test_label, batch_size=32):
        X = []
        for i in range(len(train_text)):
            X.append(model.infer_vector(train_text[i]))
        train_x = np.asarray(X)
        print(train_x.shape)

        X = []
        for i in range(len(test_text)):
            X.append(model.infer_vector(test_text[i]))
        test_x = np.asarray(X)
        print(test_x.shape)

        Y = np.asarray(train_label)
        le = preprocessing.LabelEncoder()
        le.fit(Y)
        train_y = le.transform(Y)
        print(train_y.shape)

        Y = np.asarray(test_label)
        le = preprocessing.LabelEncoder()
        le.fit(Y)
        test_y = le.transform(Y)
        print(test_y.shape)

        train_data = []
        for i in range(len(train_x)):
            train_data.append([train_x[i], train_y[i]])

        test_data = []
        for i in range(len(test_x)):
            test_data.append([test_x[i], test_y[i]])

        train_loader = DataLoader(train_data, batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size, num_workers=2, pin_memory=True)

        return train_x, train_y, test_x, test_y, train_loader, test_loader

    def classify_model_logistic_regression(train_x, train_y):
        # fit logistic regression model
        logreg = linear_model.LogisticRegression()
        logreg.fit(train_x, train_y)        
        return logreg

    def classfiy_model_neural_network(train_loader, num_labels, vocab_size, num_epochs=20):
        # Set up GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Get model
        nn_model = SimpleNeuralNetwork(num_labels, vocab_size)
        # print(nn_model)

        # Pass model to GPU
        nn_model.to(device)
        losses = []

        loss_function = torch.nn.NLLLoss()
        optimizer = optim.SGD(nn_model.parameters(), lr=0.1)

        start = time.time()
        # Set up epoch 
        for epoch in range(num_epochs):
            all_loss = 0
            for idx, batch in enumerate(train_loader):
                # print ("iter: ", idx)
                batch_loss = 0
                nn_model.zero_grad()
                input_ids = batch[0].to(device)
                label_ids = batch[1].to(device)
                out = nn_model(input_ids)
                out = F.log_softmax(out, dim=1)
                batch_loss = loss_function(out, label_ids)
                batch_loss.backward()
                optimizer.step()
                all_loss += batch_loss.item()
            print("epoch: ", epoch, "\t" , "loss: ", all_loss)

        end = time.time()
        print ("time : ", end - start)        
        return nn_model, train_loader


class Doc2vecModelPrediction:
    def predict_texts_logreg(logreg, test_text, test_label, title_name=None):
        test_list = []
        for i in range(len(test_text)):
            test_list.append(model.infer_vector(test_text[i]))
        test_x = np.asarray(test_list)
        test_Y = np.asarray(test_label)
        test_y = le.transform(test_Y)
        preds = logreg.predict(test_x)

        # print classification report
        print(classification_report(preds, test_y))
        print("predicted label: ", set(preds))

        # plot confusion matrix
        plot_confusion_matrix_heatmap(test_y, preds, "confusion matrix {}".format(title_name))
        try:
            plot_roc_auc(test_y, preds, title_name)
        except:
            pass        
        return

    def predict_texts_nn(nn_model, train_loader, title_name=None, num_classes=2):
        # Set up GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      
        answer = []
        prediction = []
        start = time.time()
        with torch.no_grad():
            for batch in train_loader:
                text_tensor = torch.as_tensor(batch[0]).to(device)
                label_tensor = torch.as_tensor(batch[1]).to(device)

                score = nn_model(text_tensor)
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
        return    

