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

MDAD_drug_smiles = []
workbook = openpyxl.load_workbook('./source_dealed_data/MDAD_data/drug_microbe_matrix.xlsx')
sheet = workbook.active
data = []
for row in sheet.rows:
    row_data = []
    for cell in row:
        row_data.append(cell.value)
    MDAD_drug_smiles.append(row_data[1])
    data.append(row_data)
MDAD_microbe_name = data[0]

kn = np.loadtxt('./data_index/MDAD_index/known.txt')
uk = np.loadtxt('./data_index/MDAD_index/unknown.txt')
kn = kn.tolist()
uk = uk.tolist()

aB_only_1 = []
aB_only_0 = []
rep_1 = []
rep_0 = []
MD_only_1 = []
MD_only_0 = []

for i in range(len(kn)):
    if data[int(kn[i][0])][1] not in aBiofilm_drug_smiles or data[0][int(kn[i][1]+1)] not in aBiofilm_microbe_name:
        MD_only_1.append([data[int(kn[i][0])][1], data[0][int(kn[i][1]+1)], data[int(kn[i][0])][int(kn[i][1]+1)]])
    else:
        rep_1.append([data[int(kn[i][0])][1], data[0][int(kn[i][1]+1)], data[int(kn[i][0])][int(kn[i][1]+1)]])
for i in range(len(uk)):
    if data[int(uk[i][0])][1] not in aBiofilm_drug_smiles or data[0][int(uk[i][1]+1)] not in aBiofilm_microbe_name:
        MD_only_0.append([data[int(uk[i][0])][1], data[0][int(uk[i][1]+1)], data[int(uk[i][0])][int(uk[i][1]+1)]])
    else:
        rep_0.append([data[int(uk[i][0])][1], data[0][int(uk[i][1]+1)], data[int(uk[i][0])][int(uk[i][1]+1)]])

kn = np.loadtxt('./data_index/aBiofilm_index/known.txt')
uk = np.loadtxt('./data_index/aBiofilm_index/unknown.txt')
kn = kn.tolist()
uk = uk.tolist()

for i in range(len(kn)):
    if dataa[int(kn[i][0])][1] not in MDAD_drug_smiles or dataa[0][int(kn[i][1]+1)] not in MDAD_microbe_name:
        aB_only_1.append([dataa[int(kn[i][0])][1], dataa[0][int(kn[i][1]+1)], dataa[int(kn[i][0])][int(kn[i][1]+1)]])
for i in range(len(uk)):
    if dataa[int(uk[i][0])][1] not in MDAD_drug_smiles or dataa[0][int(uk[i][1]+1)] not in MDAD_microbe_name:
        aB_only_0.append([dataa[int(uk[i][0])][1], dataa[0][int(uk[i][1]+1)], dataa[int(uk[i][0])][int(uk[i][1]+1)]])

random.shuffle(aB_only_1)
random.shuffle(aB_only_0)
random.shuffle(rep_1)
random.shuffle(rep_0)
random.shuffle(MD_only_1)
random.shuffle(MD_only_0)
val_1 = aB_only_1[0:len(aB_only_1)//10] + aB_only_0[0:len(aB_only_0)//10]
val_2 = rep_1[0:len(rep_1)//10] + rep_0[0:len(rep_0)//10]
val_3 = MD_only_1[0:len(MD_only_1)//10] + MD_only_0[0:len(MD_only_0)//10]
train_1 = aB_only_1[len(aB_only_1)//10:len(aB_only_1)-len(aB_only_1)//5] + aB_only_0[len(aB_only_0)//10:len(aB_only_0)-len(aB_only_0)//5]
train_2 = rep_1[len(rep_1)//10:len(rep_1)-len(rep_1)//5] + rep_0[len(rep_0)//10:len(rep_0)-len(rep_0)//5]
train_3 = MD_only_1[len(MD_only_1)//10:len(MD_only_1)-len(MD_only_1)//5] + MD_only_0[len(MD_only_0)//10:len(MD_only_0)-len(MD_only_0)//5]
test_1 = aB_only_1[len(aB_only_1)-len(aB_only_1)//5:len(aB_only_1)] + aB_only_0[len(aB_only_0)-len(aB_only_0)//5:len(aB_only_0)]
test_2 = rep_1[len(rep_1)-len(rep_1)//5:len(rep_1)] + rep_0[len(rep_0)-len(rep_0)//5:len(rep_0)]
test_3 = MD_only_1[len(MD_only_1)-len(MD_only_1)//5:len(MD_only_1)] + MD_only_0[len(MD_only_0)-len(MD_only_0)//5:len(MD_only_0)]
random.shuffle(val_1)
random.shuffle(val_2)
random.shuffle(val_3)
random.shuffle(train_1)
random.shuffle(train_2)
random.shuffle(train_3)
print(len(train_1))
print(len(train_2))
print(len(train_3))

train = train_1 + train_2 + train_3
val = val_1 + val_2 + val_3
test = test_1 + test_2 + test_3
print('aB_only_length:'+str(len(test_1)))
print('repeated_length:'+str(len(test_2)))
print('MD_only_length:'+str(len(test_3)))

with open('./datasets/drug_microbe/aBiofilm_to_MDAD/val.csv', 'w', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(st)
    for row in val:
        writer.writerow(row)
f.close()

with open('./datasets/drug_microbe/aBiofilm_to_MDAD/train.csv', 'w', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(st)
    for row in train:
        writer.writerow(row)
f.close()

with open('./datasets/drug_microbe/aBiofilm_to_MDAD/test.csv', 'w', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(st)
    for row in test:
        writer.writerow(row)
f.close()