import csv
import random
import numpy as np
import openpyxl

st = ['SMILES', 'Microbe', 'Y']

workbook = openpyxl.load_workbook('./source_dealed_data/aBiofilm_data/drug_microbe_matrix.xlsx')
sheet = workbook.active
dataa = []
for row in sheet.rows:
    row_data = []
    for cell in row:
        row_data.append(cell.value)
    dataa.append(row_data)

kn = np.loadtxt('./data_index/aBiofilm_index/known.txt')
uk = np.loadtxt('./data_index/aBiofilm_index/unknown.txt')
kn = kn.tolist()
uk = uk.tolist()

kn_va = kn[0:len(kn)//10]
uk_va = uk[0:len(uk)//10]
kn_tr = kn[len(kn)//10:len(kn)-len(kn)//5]
uk_tr = uk[len(uk)//10:len(uk)-len(uk)//5]
kn_te = kn[len(kn)-len(kn)//5:len(kn)]
uk_te = uk[len(uk)-len(uk)//5:len(uk)]
val = kn_va + uk_va
train = kn_tr + uk_tr
test = kn_te + uk_te
random.shuffle(val)
random.shuffle(train)
#np.savetxt('./datasets/drug_microbe/aBiofilm/learning_part.txt', kn_tr + kn_va)
#np.savetxt('./datasets/drug_microbe/aBiofilm/testing_part.txt', kn_te)

for i in range(len(val)):
    val[i] = [dataa[int(val[i][0])][1], dataa[0][int(val[i][1]+1)], dataa[int(val[i][0])][int(val[i][1]+1)]]

for i in range(len(train)):
    train[i] = [dataa[int(train[i][0])][1], dataa[0][int(train[i][1]+1)], dataa[int(train[i][0])][int(train[i][1]+1)]]

for i in range(len(test)):
    test[i] = [dataa[int(test[i][0])][1], dataa[0][int(test[i][1]+1)], dataa[int(test[i][0])][int(test[i][1]+1)]]

with open('./datasets/drug_microbe/aBiofilm/val.csv', 'w', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(st)
    for row in val:
        writer.writerow(row)
f.close()

with open('./datasets/drug_microbe/aBiofilm/train.csv', 'w', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(st)
    for row in train:
        writer.writerow(row)
f.close()

with open('./datasets/drug_microbe/aBiofilm/test.csv', 'w', newline='',encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(st)
    for row in test:
        writer.writerow(row)
f.close()