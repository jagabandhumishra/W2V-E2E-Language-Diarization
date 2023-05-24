import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from csv import DictWriter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from data_load import *
from model import *
from model_evaluation import *
#%%
# python WaveAttention.py --savedir " " \
# --train " " \
# --test " " \
# --seed 0 --device 0 --batch 32 --epochs 60 --lang 3 --model my_base_model --lr 0.0001 --maxlength 666 --lmbda 0.5 --fll ""

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False # Do not set True here, otherwise the code will be 10 times slower
    torch.backends.cudnn.benchmark = True

def get_output(outputs, seq_len):
    output_ = 0
    for i in range(len(seq_len)):
        length = seq_len[i]
        output = outputs[i, :length, :]
        if i == 0:
            output_ = output
        else:
            output_ = torch.cat((output_, output), dim=0)
    return output_

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
"""
def change(MyList, fll):
    empty = []
    for ele in MyList:
        if ele==0:
            empty.append(fll)
        elif ele==1:
            empty.append('S')
    return empty
"""

def change(MyList, fll):
    empty = []
    for ele in MyList:
        if ele==0:
            empty.append('B')
        elif ele==1:
            empty.append(fll)
        elif ele==2:
            empty.append('S')
    return empty

"""
def evaluation_metric(labels,predicted,fll):
    parameters = metrics.classification_report(labels,predicted,zero_division=0)
    cm = metrics.confusion_matrix(labels, predicted)
    
    if fll == 'P':
        cm = [ cm[1][1], cm[1][0], cm[0][1], cm[0][0] ]
    elif fll == 'P':
        cm = [ cm[1][1], cm[1][0], cm[0][1], cm[0][0] ]

    s1 = sum(cm[0:2])
    s2 = sum(cm[2:4])
    #s3 = sum(cm[6:])
    cm1 = [round(cm[0]*100/s1,2),round( cm[1]*100/s1,2), round(cm[2]*100/s1,2), round(cm[3]*100/s2,2) ]
    values = parameters.split()

    if fll == 'P':
        mydata = [labels, predicted, values[21], values[13], values[8], values[18], values[20],values[11],values[6],values[16]]
    elif fll == 'P':
        mydata = [labels, predicted, values[21], values[18], values[8], values[13], values[20],values[16],values[6],values[11]]
    
    print(cm)
    print(cm1)
    return mydata, cm1, cm
"""
def evaluation_metric(labels,predicted,fll):
    parameters = metrics.classification_report(labels,predicted,zero_division=0)
    cm = metrics.confusion_matrix(labels, predicted)

    if fll == 'P':
        cm = [ cm[1][1], cm[1][0], cm[1][2], cm[0][1], cm[0][0], cm[0][2], cm[2][1], cm[2][0], cm[2][2] ]
    elif fll == 'P':
        cm = [ cm[1][1], cm[1][0], cm[1][2], cm[0][1], cm[0][0], cm[0][2], cm[2][1], cm[2][0], cm[2][2] ]

    s1 = sum(cm[0:3])
    s2 = sum(cm[3:6])
    s3 = sum(cm[6:])
    cm1 = [round(cm[0]*100/s1,2),round( cm[1]*100/s1,2), round(cm[2]*100/s1,2), round(cm[3]*100/s2,2), round(cm[4]*100/s2,2), round(cm[5]*100/s2,2), round(cm[6]*100/s3,2), round(cm[7]*100/s3,2), round(cm[8]*100/s3,2) ]
    values = parameters.split()

    if fll == 'P':
        mydata = [labels, predicted, values[21], values[13], values[8], values[18], values[20],values[11],values[6],values[16]]
    elif fll == 'P':
        mydata = [labels, predicted, values[21], values[18], values[8], values[13], values[20],values[16],values[6],values[11]]

    print(cm)
    print(cm1)
    return mydata, cm1, cm

def main():
    global best_eer
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--savedir', type=str, help='dir in which the trained model is saved')
    parser.add_argument('--train', type=str, help='training data, in .txt')
    parser.add_argument('--test', type=str, help='testing data, in .txt')
    parser.add_argument('--seed', type=int, help='seed', default=0)
    parser.add_argument('--device', type=int, help='Device ID', default=0)
    parser.add_argument('--batch', type=int, help='batch size', default=32)
    parser.add_argument('--epochs', type=int, help='num of epochs', default=30)
    parser.add_argument('--lang', type=int, help='num of language classes', default=3)
    parser.add_argument('--model', type=str, help='model name', default='Transformer')
    parser.add_argument('--lr', type=float, help='initial learning rate', default=0.0001)
    parser.add_argument('--maxlength', type=int, help='Max sequence length for positional enc', default=666)
    parser.add_argument('--lmbda', type=float, help='hyperparameter for joint training, default 0.5', default=0.5)
    parser.add_argument('--fll', type=str, help='first language literal, ex : Gujarati->"G", Tamil/Telugu->T',required=True)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    print('Current device: {}'.format(device))

    #load model
    model = X_Attention_E2E_LD(n_lang=args.lang,
                                    dropout=0.1,
                                    feat_dim=256,
                                    n_heads=4,
                                    d_k=256,
                                    d_v=256,
                                    d_ff=2048,
                                    max_seq_len=args.maxlength,
                                    device=device)
                                    
    model.to(device)
    #class_weights = torch.tensor([1.7558, 0.4868, 2.6658])
    #                  [     S,    G/T,      E]
    # Gujarati-English [1.8027, 0.4905, 2.4586]
    # Telagu-English   [1.7473, 0.4849, 2.7338]
    # Tamil-English    [1.7176, 0.4851, 2.8051]
    # Average          [1.7558, 0.4868, 2.6658]
    #loss_func_CRE = nn.CrossEntropyLoss(weight=class_weights,reduction='mean', label_smoothing=0.1).to(device)
    #loss_func_xv = nn.CrossEntropyLoss(weight=class_weights,reduction='mean',ignore_index=255, label_smoothing=0.1).to(device) # this is important since 255 is for zero paddings
    loss_func_CRE = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1).to(device)
    loss_func_xv = nn.CrossEntropyLoss(reduction='mean',ignore_index=255, label_smoothing=0.1).to(device) # this is important since 255 is for zero paddings
    train_txt = args.train
    valid_txt = args.test
    train_set = RawFeatures(train_txt)
    valid_set = RawFeatures(valid_txt)

    train_data = DataLoader(dataset=train_set,
                                batch_size=args.batch,
                                pin_memory=True,
                                num_workers=16,
                                shuffle=True,
                                collate_fn=collate_fn_cnn_atten)

    valid_data = DataLoader(dataset=valid_set,
                                batch_size=args.batch,
                                pin_memory=True,
                                shuffle=False,
                                collate_fn=collate_fn_cnn_atten) 

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    total_step = len(train_data)
    best_acc = 0

    Epoch = []
    Loss = []
    Curr_LR = []
    Curr_Acc = []

    EER = []
    eer0 = []
    eer1 = []
    eer2 = []

    ACTUAL = []
    PREDICTED = []
    N_TOTAL = []
    N_L1 = []
    N_L2 = []
    N_S = []
    ACCURACY = []
    RECL_L1 = []
    RECL_L2 = []
    RECL_S = []

    CMP = []
    CMA = []
    
    for epoch in tqdm(range(args.epochs)):
        MYACTUAL = []
        MYPRED = []
        model.train()
        for step, (utt, labels, cnn_labels, seq_len, filename) in enumerate(train_data):
            utt_ = utt.to(device=device, dtype=torch.float)
            atten_mask = get_atten_mask(seq_len, utt_.size(0))
            atten_mask = atten_mask.to(device=device)
            labels = labels.to(device=device, dtype=torch.long)
            cnn_labels = cnn_labels.to(device=device, dtype=torch.long)

            # Forward pass
            outputs, cnn_outputs = model(utt_, seq_len, atten_mask)
            outputs = get_output(outputs, seq_len)
            loss_trans = loss_func_CRE(outputs, labels)
            loss_xv = loss_func_xv(cnn_outputs,cnn_labels)
            loss = args.lmbda*loss_trans + (1-args.lmbda)*loss_xv

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 25 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Trans: {:.4f} XV: {:.4f}".
                    format(epoch + 1, args.epochs, step + 1, total_step,
                            loss.item(), loss_trans.item(), loss_xv.item()))
        scheduler.step()
        print('Current LR: {}'.format(get_lr(optimizer)))

        model.eval()
        eer = 0
        correct = 0
        total = 0
        FAR_list = torch.zeros(args.lang)
        FRR_list = torch.zeros(args.lang)
        with torch.no_grad():
            for step, (utt, labels, cnn_labels, seq_len, filename) in enumerate(valid_data):
                utt_ = utt.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)
                outputs, cnn_outputs = model(x=utt_, seq_len=seq_len, atten_mask=None)
                outputs = get_output(outputs, seq_len)
                predicted = torch.argmax(outputs, -1)
                total += labels.size(-1)
                print(labels.shape)
                print(predicted.shape)
                correct += (predicted == labels).sum().item()
                FAR, FRR = compute_far_frr(args.lang, predicted, labels)
                FAR_list += FAR
                FRR_list += FRR

                if step%200 ==0:
                    print(step)
                L1 = labels.cpu().numpy()
                P1 = predicted.cpu().numpy()

                L1 = change(L1, args.fll)
                P1 = change(P1, args.fll)

                MYACTUAL.extend(L1)
                MYPRED.extend(P1)

            acc = correct / total

        my_data, cmpp, cma = evaluation_metric(MYACTUAL, MYPRED, args.fll)

        print('Current Acc.: {:.4f} %'.format(100 * acc))
        for i in range(args.lang):
            eer_ = (FAR_list[i] / total + FRR_list[i] / total) / 2
            eer += eer_
            print("EER for label {}: {:.4f}%".format(i, eer_ * 100))

        print('EER: {:.4f} %'.format(100 * eer / args.lang))
        if acc > best_acc:
            print('New best Acc.: {:.4f}%, EER: {:.4f} %, model saved!'.format(100 * acc, 100 * eer / args.lang))
            best_acc = acc
            best_eer = eer / args.lang
            torch.save(model.state_dict(), args.savedir +'/'+ '{}.ckpt'.format(args.model))

        eer0.append(((FAR_list[0] / total + FRR_list[0] / total) / 2).item())
        eer1.append(((FAR_list[1] / total + FRR_list[1] / total) / 2).item())
        #eer2.append(((FAR_list[2] / total + FRR_list[2] / total) / 2).item())


        Epoch.append(epoch + 1)
        Loss.append(loss.item())

        Curr_LR.append(get_lr(optimizer))
        Curr_Acc.append(100 * acc)
        EER.append((100 * eer / args.lang).item())

        ACTUAL.append(my_data[0])
        PREDICTED.append(my_data[1])
        N_TOTAL.append(my_data[2])
        N_L1.append(my_data[3])
        N_L2.append(my_data[4])
        N_S.append(my_data[5])
        ACCURACY.append(my_data[6])
        RECL_L1.append(my_data[7])
        RECL_L2.append(my_data[8])
        RECL_S.append(my_data[9])
        CMP.append(cmpp)
        CMA.append(cma)
        
        pd.DataFrame(data={"Epoch": Epoch, "Curr_LR":Curr_LR, "Curr_Acc": Curr_Acc,"LOSS":Loss,"EER":EER,'eer0':eer0,'eer1':eer1, 'N_Total':N_TOTAL,'N_L1':N_L1, 'N_L2':N_L2, 'N_S':N_S, 'ACCURACY':ACCURACY, 'RECL_L1':RECL_L1,'RECL_L2':RECL_L2,'RECL_S':RECL_S,'CMP':CMP, 'CMA':CMA}).to_csv(args.savedir+"/Model_Details_wavvec.csv")

    print('Final Acc: {:.4f}%, EER: {:.4f}%'.format(100 * best_acc, 100 * best_eer))
    model_name = args.savedir + '{}.ckpt'.format(args.model)
    final_name = args.savedir + '{}_{:.4f}_{:.4f}.ckpt'.format(args.model, best_acc * 100, best_eer * 100)
    os.rename(model_name, final_name)

if __name__ == "__main__":
    main()
    
# python WaveAttention.py --savedir "/data/KLESLD/Final_checkpoints/WavAttention/Telagu_pre" \
# --train "/data/KLESLD/Dataset_WV2/Telugu/Train/WAVData.txt" \
# --test "/data/KLESLD/Dataset_WV2/Telugu/Dev/WAVData.txt" \
# --seed 0 --device 0 --batch 32 --epochs 60 --lang 3 --model my_base_model --lr 0.0001 --maxlength 666 --lmbda 0.5 --fll "T"
# %%
