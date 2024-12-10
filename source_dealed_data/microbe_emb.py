import numpy as np
import openpyxl
import json

workbook = openpyxl.load_workbook("./aBiofilm_data/microbes.xlsx")
sheet = workbook.active
aB = []
aB_name = []
for row in sheet.rows:
    row_data = []
    for cell in row:
        row_data.append(cell.value)
    aB.append(row_data)
    aB_name.append(row_data[1])
del aB[0]
del aB_name[0]
aB = aB[:140]
aB_name = aB_name[:140]

workbook = openpyxl.load_workbook("./MDAD_data/microbes.xlsx")
sheet = workbook.active
MD = []
MD_name = []
for row in sheet.rows:
    row_data = []
    for cell in row:
        row_data.append(cell.value)
    MD.append(row_data)
    MD_name.append(row_data[1])
del MD[0]
del MD_name[0]
MD = MD[:173]
MD_name = MD_name[:173]

aB_MD = aB.copy()
aB_MD_name = aB_name.copy()
t = 0
for i in range(len(MD)):
    if MD[i][1] not in aB_name:
        t += 1
        aB_MD.append(MD[i])
        aB_MD_name.append(MD[i][1])

MD_aB = MD.copy()
MD_aB_name = MD_name.copy()
t = 0
for i in range(len(aB)):
    if aB[i][1] not in MD_name:
        t += 1
        MD_aB.append(aB[i])
        MD_aB_name.append(aB[i][1])


aB_index = []
MD_index = []
aB_MD_index = []
MD_aB_index = []

for i in range(len(aB)):
    aB[i][0] = i + 1
    if aB[i][2] != None:
        aB_index.append(aB[i][0])
for i in range(len(MD)):
    MD[i][0] = i + 1
    if MD[i][2] != None:
        MD_index.append(MD[i][0])
for i in range(len(aB_MD)):
    aB_MD[i][0] = i + 1
    if aB_MD[i][2] != None:
        aB_MD_index.append(aB_MD[i][0])
for i in range(len(MD_aB)):
    MD_aB[i][0] = i + 1
    if MD_aB[i][2] != None:
        MD_aB_index.append(MD_aB[i][0])


np.savetxt("./MDAD_data/microbe_genomic_emb/constant_index.txt", MD_index)
np.savetxt("./aBiofilm_data/microbe_genomic_emb/constant_index.txt", aB_index)
np.savetxt("./MDAD_data/microbe_genomic_emb_extra/constant_index.txt", MD_aB_index)
np.savetxt("./aBiofilm_data/microbe_genomic_emb_extra/constant_index.txt", aB_MD_index)


def SVD_embedding(genomic_sim_matrix):
    U, S, V = np.linalg.svd(genomic_sim_matrix)
    s = np.diag(S)**0.5
    return U@s, np.transpose(s@V)

def genomic_similarity(command=int, length=16):   #Length for seed fragment in microbe 16s rDNA genomic analysis usually between 14 and 16, which can be adjusted by readers.
    if command == 1:
        data = MD
        index = MD_index
        name = MD_name
    elif command == 2:
        data = aB
        index = aB_index
        name = aB_name
    elif command == 3:
        data = MD_aB
        index = MD_aB_index
        name = MD_aB_name
    elif command == 4:
        data = aB_MD
        index = aB_MD_index
        name = aB_MD_name
    print(index)
    l = len(index)
    genomic_similarity_matrix = np.zeros((l,l))
    for i in range(l):
        print(index[i])
        for j in range(l):
            n = len(data[index[i] - 1][2]) - length + 1
            m = 0
            for k in range(n):
                if data[index[i] - 1][2][k:k+length] in data[index[j] - 1][2]:
                    m += 1
            genomic_similarity_matrix[i][j] = m/n
    microbe_embedding_1, microbe_embedding_2 = SVD_embedding(genomic_similarity_matrix)
    if command % 2 == 0:
        fragment = "aBiofilm_data"
    else:
        fragment = "MDAD_data"
    if command <= 2:
        fragment2 = "microbe_genomic_emb"
    else:
        fragment2 = "microbe_genomic_emb_extra"
    #dict = {k: v for k, v in zip(name, [x+1 for x in range(len(name))])}
    with open("./" + fragment + "/" + fragment2 + "/microbe_index_dict.txt", 'w', encoding='utf-8') as f:
        json.dump(name, f, ensure_ascii=False, indent=4)
    lim = []
    l = len(index)
    ll = len(data)
    data_category = []
    for i in range(ll):
        data_category.append(data[i][6])
    microbe_emb_1 = np.zeros((ll, len(microbe_embedding_1[0])))
    microbe_emb_2 = np.zeros((ll, len(microbe_embedding_2[0])))
    other_category = []
    for i in range(l):
        d = index[i]-1
        microbe_emb_1[d] = microbe_embedding_1[i]
        microbe_emb_2[d] = microbe_embedding_2[i]
        other_category.append(data[d][6])
    for i in range(ll):
        if i + 1 not in index:
            category = data_category[i]
            if category in other_category:
                lim.append(i+1)
                n = 0
                emb1 = np.zeros_like(microbe_emb_1[0])
                emb2 = np.zeros_like(microbe_emb_2[0])
                for j in range(l):
                    if category == data[index[j]-1][6]:
                        emb1 += microbe_embedding_1[j]
                        emb2 += microbe_embedding_2[j]
                        n += 1
                microbe_emb_1[i] = emb1 / n
                microbe_emb_2[i] = emb2 / n
    np.savetxt("./" + fragment + "/" + fragment2 + "/limit_index.txt", lim)
    np.savetxt("./" + fragment + "/" + fragment2 + "/microbe_embedding_1.txt", microbe_emb_1)
    np.savetxt("./" + fragment + "/" + fragment2 + "/microbe_embedding_2.txt", microbe_emb_2)
    print("part "+str(command)+" done")

for i in range(1,5):
    genomic_similarity(command=i)