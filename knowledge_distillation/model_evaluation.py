import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import torch

import warnings
warnings.filterwarnings('ignore')


def plot_confusion_matrix_heatmap(true_labels, predicted_labels, title_name):
    # get_metrics
    print('Accuracy:', metrics.accuracy_score(true_labels, predicted_labels))
    print('Precision:', metrics.precision_score(true_labels, predicted_labels, average='weighted'))
    print('Recall:', metrics.recall_score(true_labels, predicted_labels, average='weighted'))
    print('F1 Score:', metrics.f1_score(true_labels, predicted_labels,average='weighted'))

    # confusion matrix
    labels = list(set(true_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    cm_labeled = pd.DataFrame(cm, columns=labels, index=labels)
    fig = plt.figure(figsize=(6,4))
    sns.heatmap(cm_labeled, annot=True, cmap='Greens', fmt='g')
    title_name = "Confusion Matrix " + str(title_name)
    plt.title(title_name)
    plt.show()
    plt.close(fig)
    return

def plot_roc_auc(true_labels, predicted_labels, title_name, num_class=2):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    if num_class==2:
        auc = roc_auc_score(true_labels, predicted_labels, average=None)
        precision_score = average_precision_score(true_labels, predicted_labels)
        print ('precision_score', precision_score)
    else:
        labels = list(range(0,num_class))
        ypreds = label_binarize(predicted_labels, classes=labels)
        auc = roc_auc_score(true_labels, ypreds, average='macro',multi_class='ovo')
    print ('roc_auc_score-auc:', auc)
    
    if num_class==2:
        fpr, tpr, _ = roc_curve(true_labels,  predicted_labels)
        print ('roc_curve-fpr:', fpr)
        print ('roc_curve-tpr:', tpr)

        fig = plt.figure(figsize=(6,4))
        plt.plot(fpr,tpr,label="auc="+str(auc))
        plt.legend(loc=4)
        title_name = "ROC and AUC " + str(title_name)
        plt.title(title_name)
        plt.show()
        plt.close(fig)
    return
