import csv
import random
import numpy as np
import openpyxl

st = ['SMILES', 'Microbe', 'Y']

aBiofilm_drug_smiles = []
workbook = openpyxl.load_workbook('./source_dealed_data/aBiofilm_data/drug_microbe_matrix.xlsx')
sheet = workbook.active
dataa = []
for row in sheet.rows:
    row_data = []
    for cell in row:
        row_data.append(cell.value)
    aBiofilm_drug_smiles.append(row_data[1])
    dataa.append(row_data)

aBiofilm_microbe_name = dataa[0]

workbook = openpyxl.load_workbook('./source_dealed_data/MDAD_data/drug_microbe_matrix.xlsx')
sheet = workbook.active
data = []
for row in sheet.rows:
    row_data = []
    for cell in row:
        row_data.append(cell.value)
    data.append(row_data)
kn = np.loadtxt('./data_index/MDAD_index/known.txt')
uk = np.loadtxt('./data_index/MDAD_index/unknown.txt')
kn = kn.tolist()
uk = uk.tolist()

new_MDAD_connect = []
new_MDAD_uncertain = []

for i in range(len(kn)):
    if data[int(kn[i][0])][1] not in aBiofilm_drug_smiles or data[0][int(kn[i][1]+1)] not in aBiofilm_microbe_name:
        new_MDAD_connect.append([data[int(kn[i][0])][1],data[0][int(kn[i][1]+1)], data[int(kn[i][0])][int(kn[i][1]+1)]])
for i in range(len(uk)):
    if data[int(uk[i][0])][1] not in aBiofilm_drug_smiles or data[0][int(uk[i][1]+1)] not in aBiofilm_microbe_name:
        new_MDAD_uncertain.append([data[int(uk[i][0])][1],data[0][int(uk[i][1]+1)], data[int(uk[i][0])][int(uk[i][1]+1)]])

kn = np.loadtxt('./data_index/aBiofilm_index/known.txt')
uk = np.loadtxt('./data_index/aBiofilm_index/unknown.txt')
kn = kn.tolist()
uk = uk.tolist()

label_1 = []
label_0 = []

for i in range(len(kn)):
    label_1.append([dataa[int(kn[i][0])][1], dataa[0][int(kn[i][1]+1)], dataa[int(kn[i][0])][int(kn[i][1]+1)]])

for i in range(len(uk)):
    label_0.append([dataa[int(uk[i][0])][1], dataa[0][int(uk[i][1]+1)], dataa[int(uk[i][0])][int(uk[i][1]+1)]])

random.shuffle(label_1)
random.shuffle(label_0)

kn_va = label_1[0:len(label_1)//8]
uk_va = label_0[0:len(label_0)//8]
kn_tr = label_1[len(label_1)//8:len(label_1)]
uk_tr = label_0[len(label_0)//8:len(label_0)]
kn_te = new_MDAD_connect
uk_te = new_MDAD_uncertain
val = kn_va + uk_va
train = kn_tr + uk_tr
test = kn_te + uk_te
random.shuffle(val)
random.shuffle(train)

with open('./datasets/drug_microbe/aBiofilm_MDAD/val.csv', 'w', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(st)
    for row in val:
        writer.writerow(row)
f.close()

with open('./datasets/drug_microbe/aBiofilm_MDAD/train.csv', 'w', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(st)
    for row in train:
        writer.writerow(row)
f.close()

with open('./datasets/drug_microbe/aBiofilm_MDAD/test.csv', 'w', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(st)
    for row in test:
        writer.writerow(row)
f.close()