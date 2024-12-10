import seaborn as sns
import matplotlib.pyplot as plt
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
import torch.nn.functional as F
import math
from matplotlib.lines import Line2D

#Parameters setting: --cfg "configs/MiDSFN.yaml" --data "drug_microbe" --split "Ethnic_interaction"
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
ex_split = args.split
n_class = cfg["DECODER"]["BINARY"]
decoder = cfg["DECODER"]["NAME"]
params = {'batch_size': 1, 'shuffle': False, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}
dataFolder = f'./datasets/{args.data}/'
dataFolder = os.path.join(dataFolder, ex_split)
result_path = os.path.join('./result/' + ex_split, decoder)
test_path = os.path.join(dataFolder, "test.csv")
df_test = pd.read_csv(test_path)
test_dataset = DTIDataset(df_test.index.values, df_test)
test_generator = DataLoader(test_dataset, **params)
num_batches = len(test_generator)

MyModel = DrugBAN(**cfg).to(device)
MyModel.eval()
if decoder == "MLP":
    MyModel.load_state_dict(torch.load(result_path + '/best_model_epoch_17.pth')) #This place need to be confirm every time.
if decoder == "KAN":
    MyModel.load_state_dict(torch.load(result_path + '/best_model_epoch_5.pth')) #This place need to be confirm every time.

def invoke(MyModel, test_generator):
    test_loss = 0
    y_label, y_pred = [], []
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
            y_label = y_label + labels.to("cpu").tolist()
            y_pred = y_pred + n.to("cpu").tolist()
    return y_pred

scores = invoke(MyModel, test_generator)
print(scores)

a = 0.004
for i in range(len(scores)):
    scores[i] = scores[i]/(a+scores[i])

data_len = [0, 156, 299, 369, 397, 446, 495]
every_len = [[[0, 36, 72, 108, 156], [0, 33, 66, 99, 143]], [[0, 40, 60, 70], [0, 16, 24, 28]],
             [[0, 7, 14, 35, 49], [0, 7, 14, 35, 49]]]
p_d = ['Antibiotics', 'Antifungal', 'Antiviral']
p_m = [['Gram positive bacteria', 'Gram negative bacteria'], ['Yeast', 'Mold'], ['DNA virus', 'RNA virus']]
colors = ['lightblue', 'pink', 'lightgreen']
microbe_categories = [['Penicillins', 'Cephalosporins', 'Macrolides', 'Quinolones'],
                      ['Azoles', 'Polyene class'],
                      ['Nucleoside analogues (Acyclovir)', 'Nucleoside analogues (Lamivudine)',
                       'Protease inhibitors', 'Neuraminidase inhibitor']]
#Polyene class (Amphotericin B) Polyene class (Caspofungin)
color_d = ['r', 'b', 'y', 'g']
alpha_d = [0.7, 0.6, 0.9, 0.75]

t = 0
for i in range(3):
    drug_category = p_d[i]
    microbe_category = p_m[i]
    datas = []
    data_sp = []
    data_cut = []
    c = colors[i]
    m_c = microbe_categories[i]
    e_l = every_len[i]
    for j in range(2):
        data = scores[data_len[t]: data_len[t+1]]
        sorted_data = np.sort(data)
        min_value = sorted_data[0]
        q1 = np.percentile(sorted_data, 25)
        median = np.percentile(sorted_data, 50)
        q3 = np.percentile(sorted_data, 75)
        max_value = sorted_data[-1]
        data_5 = [min_value, q1, median, q3, max_value]
        data_cut.append((float(q1-1.5*(q3-q1)), float(q3+1.5*(q3-q1))))
        data_sp.append(data)
        datas.append(data_5)
        t += 1
    plt.figure(figsize=(960, 600))
    sns.violinplot(data_sp, positions=[0, 1], inner=None, color=c, cut=0)
    plt.boxplot(datas[0], positions=[0], patch_artist=True, boxprops=dict(facecolor='white'),
                medianprops=dict(color='black'), whis=[0, 100])
    plt.boxplot(datas[1], positions=[1], patch_artist=True, boxprops=dict(facecolor='white'),
                medianprops=dict(color='black'), whis=[0, 100])
    plt.xticks([0, 1], [microbe_category[0], microbe_category[1]])
    legend_elements = []
    for k in range(len(m_c)):
        a_d = alpha_d[k]
        c_d = color_d[k]
        l1 = e_l[0][k+1]-e_l[0][k]
        l2 = e_l[1][k+1]-e_l[1][k]
        plt.scatter([-0.05 + 0.1/len(m_c)*p/l1 + 0.1*k/len(m_c) for p in range(l1)], data_sp[0][e_l[0][k]:e_l[0][k+1]],
                    edgecolors=c_d, facecolors=c_d, label=m_c[k], zorder=5, s=15, alpha=a_d, linestyle='None')
        plt.scatter([0.95 + 0.1/len(m_c)*p/l2 + 0.1*k/len(m_c) for p in range(l2)], data_sp[1][e_l[1][k]:e_l[1][k+1]],
                    edgecolors=c_d, facecolors=c_d, label=m_c[k], zorder=6, s=15, alpha=a_d, linestyle='None')
        legend_elements.append(Line2D([0], [0], marker='o', markerfacecolor='none', markeredgecolor=c_d,
                                      markerfacecoloralt=c_d, alpha=a_d, label=m_c[k], markersize=6, linewidth=0))
    plt.legend(handles=legend_elements, loc='upper center', title='Types of Drugs')
    plt.title(p_d[i] + '    a=' + str(a))
    plt.ylabel('Scores', color='black')
    plt.show()