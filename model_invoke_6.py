from models import DrugBAN
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import argparse
import warnings, os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
import torch.nn.functional as F
from openpyxl import Workbook

#Parameters setting: --cfg "configs/MiDSFN.yaml" --data "drug_microbe" --split "aBiofilm_to_MDAD"
#Decoder setting in './configs/MiDSFN.yaml': "KAN" or "MLP"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="MiDSFN for Drug-Microbe prediction")
parser.add_argument('--cfg', required=True, type=str, help="path to config file", metavar='CFG')
parser.add_argument('--data', required=True, type=str, metavar='TASK', help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task",
                    choices=['MDAD', 'aBiofilm', 'MDAD_aBiofilm', 'MDAD_aBiofilm_discard', 'aBiofilm_MDAD',
                             'MDAD+aBiofilm', 'MDAD+aBiofilm_discard', 'aBiofilm_to_MDAD', 'Case_prediction',
                             'Ethnic_interaction'])
args = parser.parse_args()

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

cfg = get_cfg_defaults()
cfg.merge_from_file(args.cfg)
n_class = cfg["DECODER"]["BINARY"]
decoder = cfg["DECODER"]["NAME"]
params = {'batch_size': 1, 'shuffle': False, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}
dataFolder = f'./datasets/{args.data}'
dataFolder = os.path.join(dataFolder, str(args.split))
result_path = os.path.join('./result/' + str(args.split), decoder)
test_path = os.path.join(dataFolder, "test.csv")
df_test = pd.read_csv(test_path)
test_dataset = DTIDataset(df_test.index.values, df_test)
test_generator = DataLoader(test_dataset, **params)
num_batches = len(test_generator)

MyModel = DrugBAN(**cfg).to(device)
MyModel.eval()
MyModel.load_state_dict(torch.load(result_path + '/best_model_epoch_20.pth')) #This place need to be confirm every time.


test_loss = 0
y_label_, y_pred_ = [], []
with torch.no_grad():
    for i, (v_d, v_p, a_m, d_i, labels) in enumerate(test_generator):
        a_m = torch.stack(a_m, 0)
        v_d, v_p, a_m, d_i, labels = v_d.to(device), v_p.to(device), a_m.to(device), d_i, labels.float().to(device)
        v_d, v_p, f, score = MyModel(v_d, v_p, a_m, d_i)
        if n_class == 1:
            n, loss = binary_cross_entropy(score, labels)
        else:
            n, loss = cross_entropy_logits(score, labels)
        test_loss += loss.item()
        y_label_ = y_label_ + labels.to("cpu").tolist()
        y_pred_ = y_pred_ + n.to("cpu").tolist()

a = 3968    #aB_only_length
r = 15047  #repeated_length
m = 13946   #MD_only_length
print(a+r+m)
eval_list = [[i for i in range(0, a)], [i for i in range(a, a+r)], [i for i in range(a+r, a+r+m)],
             [i for i in range(0, a+r)], [i for i in range(a, a+r+m)], [i for i in range(0, a+r+m)]]
name_list = ['aBiofilm_only', 'repeated_data', 'MDAD_only', 'aBiofilm_all', 'MDAD_all', 'All_data']

res = [["Test Mode", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy", "test_loss", "thred_optim", "precision1"]]
for i in range(6):
    y_pred = []
    y_label = []
    fra = eval_list[i]
    name = name_list[i]
    for j in fra:
        y_label.append(y_label_[j])
        y_pred.append(y_pred_[j])
    auroc = roc_auc_score(y_label, y_pred)
    auprc = average_precision_score(y_label, y_pred)
    test_loss = test_loss / num_batches
    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    prec, recall, _ = precision_recall_curve(y_label, y_pred)
    precision = tpr / (tpr + fpr + 1e-12)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]
    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
    cm1 = confusion_matrix(y_label, y_pred_s)
    accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
    sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    precision1 = precision_score(y_label, y_pred_s)


    print('Test '+ name + ' part at Model' + ' with test loss ' + str(test_loss),
      " AUROC " + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity "
      + str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))
    res.append([name, auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1])

print(res)
workbook = Workbook()
worksheet = workbook.active
for p in range(len(res)):
    for q in range(len(res[0])):
        worksheet.cell(row=1 + p, column=1 + q).value = res[p][q]
workbook.save(result_path + "/Test_on_different_part.xlsx")
