import time 

from sklearn.metrics import classification_report

import torch
from torch import nn
import torch.nn.functional as F
import torchtext
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim

from knowledge_distillation.model_evaluation import plot_confusion_matrix_heatmap, plot_roc_auc


# Awesome Knowledge-Distillation: https://github.com/FLHonker/Awesome-Knowledge-Distillation

# avg logits
def avg_logits(te_scores_Tensor):
    mean_Tensor = torch.mean(te_scores_Tensor, dim=1)
    return mean_Tensor

# distillation loss
def distillation_loss(out_student, labels, logits_teachers, temp, alpha=0.7):
    distil_loss = nn.KLDivLoss()(F.log_softmax(out_student/temp), logits_teachers) * (
        temp*temp * 2.0 * alpha) + F.cross_entropy(out_student, labels) * (1. - alpha)
    return distil_loss

def distillation_kd_loss(out_student, labels, logits_teachers, temp, alpha=0.5):
    # https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/train_kd.py
    distil_loss = nn.KLDivLoss()(F.log_softmax(out_student/temp), logits_teachers) * (
        temp*temp * 2.0 * alpha)
    return distil_loss


class KnowledgeDistillationModelling():
    def train(model, device, train_iter, model_name, token_name, num_classes, num_epochs=10, model_path="/"):
        print("model_name: ", model_name)
        print("token_name: ", token_name)
        # # Set up GPU
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print("device", device)
        # Pass model to GPU
        model.to(device)

        # Small lr value for pretrained layer and bigger value for the last layer
        if model_name=="distilbert":
            optimizer = optim.Adam([
                {'params': model.distil_bert.transformer.layer[-1].parameters(), 'lr': 5e-5},
                {'params': model.classifier.parameters(), 'lr': 1e-4}
            ])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss_function = nn.NLLLoss()
        losses = []

        start = time.time()
        # Set up epoch 
        for epoch in range(num_epochs):
            all_loss = 0
            for idx, batch in enumerate(train_iter):
                batch_loss = 0
                model.zero_grad()

                # process batch data
                label_ids = batch.label.to(device)
                label_ids = torch.tensor(label_ids, dtype=torch.long)
                if token_name=="distilbert":
                    text = batch.text[0].to(device)
                else:
                    text = batch.text[0].T.to(device)
                text_lengths = batch.text[1].to(device)

                # pass data to model
                out = model(text, text_lengths)
                out = F.log_softmax(out, dim=1)

                # calculate loss
                batch_loss = loss_function(out, label_ids)
                batch_loss.backward()
                optimizer.step()
                all_loss += batch_loss.item()
            print("epoch: ", epoch, "\t" , "loss: ", all_loss)

        end = time.time()
        print ("time : ", end - start)

        # save model
        model_filename = model_path + model_name + ".pth"
        torch.save(model.state_dict(), model_filename)

        # # release GPU memory
        # torch.cuda.empty_cache()        
        return model, device

    # train with multi-teacher
    def train_multi_teachers_kd(st_model, teacher_models, device, train_iter, num_epochs=10, model_path="/", model_name="kd"):
        # https://github.com/FLHonker/AMTML-KD-code/blob/master/multi_teacher_avg_distill.ipynb
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        temp = 20.0
        lr = 0.01
        optimizer = optim.Adam(st_model.parameters(), lr=lr)
        optimizer_sgd = optim.SGD(st_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_sgd, milestones=[100, 150])

        start = time.time()
        for epoch in range(num_epochs):
            lr_scheduler.step()

            # switch to train mode
            st_model = st_model.to(device)
            st_model.train()
            all_loss = 0
            for i, batch in enumerate(train_iter):
                target = batch.label.to(device)
                target = torch.tensor(target, dtype=torch.long)
                input = batch.text[0].to(device)
                input_lengths = batch.text[1].to(device)

                # compute student outputs
                output = st_model(input, input_lengths)
                # print("st_output", output)
                te_scores_list = []
                for j, te in enumerate(teacher_models):
                    te.to(device)
                    te.eval()
                    with torch.no_grad():
                        t_output = te(input, input_lengths)
                        t_output = t_output.float()
                    t_output = F.softmax(t_output/temp) # softmax with temperature
                    te_scores_list.append(t_output)
                te_scores_Tensor = torch.stack(te_scores_list, dim=1)
                mean_logits = avg_logits(te_scores_Tensor)
                
                kd_loss = 0
                st_model.zero_grad() # optimizer_sgd.zero_grad() # st_model.zero_grad()

                # compute gradient and do SGD step
                kd_loss = distillation_loss(output, target, mean_logits, temp=temp, alpha=0.7)     
                batch_loss = kd_loss

                batch_loss.backward(retain_graph=True)
                optimizer_sgd.step()

                output = output.float()
                # print("output", output)
                batch_loss = batch_loss.float()
                # print("batch_loss_float", batch_loss)
                all_loss += batch_loss.item()
            print("epoch: ", epoch, "\t" , "loss: ", all_loss)

        end = time.time()
        print ("time : ", end - start)

        # save model
        model_filename = model_path + model_name + ".pth"
        torch.save(st_model.state_dict(), model_filename)

        # release GPU memory
        torch.cuda.empty_cache()
        return st_model, device

    # train with multi-teacher with adjusted parameters
    def train_multi_teachers_kd_param(st_model, teacher_models, device, train_iter, 
                                num_epochs=10, model_path="/", model_name="kd", 
                                temp=20.0, lr=0.01):
        # https://github.com/FLHonker/AMTML-KD-code/blob/master/multi_teacher_avg_distill.ipynb

        print("temp:{}, lr:{}".format(temp, lr))
        optimizer = optim.Adam(st_model.parameters(), lr=lr)
        optimizer_sgd = optim.SGD(st_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_sgd, milestones=[100, 150])

        start = time.time()
        for epoch in range(num_epochs):
            lr_scheduler.step()

            # switch to train mode
            st_model = st_model.to(device)
            st_model.train()
            all_loss = 0
            for i, batch in enumerate(train_iter):
                target = batch.label.to(device)
                target = torch.tensor(target, dtype=torch.long)
                input = batch.text[0].to(device)
                input_lengths = batch.text[1].to(device)

                # compute student outputs
                output = st_model(input, input_lengths)
                # print("st_output", output)
                te_scores_list = []
                for j, te in enumerate(teacher_models):
                    te.to(device)
                    te.eval()
                    with torch.no_grad():
                        t_output = te(input, input_lengths)
                        t_output = t_output.float()
                    t_output = F.softmax(t_output/temp) # softmax with temperature
                    te_scores_list.append(t_output)
                te_scores_Tensor = torch.stack(te_scores_list, dim=1)
                mean_logits = avg_logits(te_scores_Tensor)
                
                kd_loss = 0
                st_model.zero_grad() # optimizer_sgd.zero_grad() # st_model.zero_grad()

                # compute gradient and do SGD step
                kd_loss = distillation_loss(output, target, mean_logits, temp=temp, alpha=0.7)     
                batch_loss = kd_loss

                batch_loss.backward(retain_graph=True)
                optimizer_sgd.step()

                output = output.float()
                # print("output", output)
                batch_loss = batch_loss.float()
                # print("batch_loss_float", batch_loss)
                all_loss += batch_loss.item()
            print("epoch: ", epoch, "\t" , "loss: ", all_loss)

        end = time.time()
        print ("time : ", end - start)

        # save model
        model_filename = model_path + model_name + ".pth"
        torch.save(st_model.state_dict(), model_filename)

        # release GPU memory
        torch.cuda.empty_cache()
        return st_model, device
                
    def predict(model, device, batch_iter, token_name="distilbert", title_name=None, num_classes=2):
        # # Set up GPU
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      
        answer = []
        prediction = []
        
        start = time.time()
        with torch.no_grad():
            for batch in (batch_iter):
                label = batch.label.to(device)
                if token_name=="distilbert":
                    text = batch.text[0].to(device)
                else:
                    text = batch.text[0].T.to(device)
                text_lengths = batch.text[1].to(device)

                score = model(text, text_lengths)
                _, pred = torch.max(score, 1)

                prediction += list(pred.cpu().numpy())
                answer += list(label.cpu().numpy())
        
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