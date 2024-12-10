import pandas as pd
import time
import requests
import openpyxl

database = 'MDAD'

workbook = openpyxl.load_workbook('./' + database + '_data/drugs.xlsx')
sheet = workbook.active
dataa = []
for row in sheet.rows:
    row_data = []
    for cell in row:
        row_data.append(cell.value)
    dataa.append(row_data[1])

drug = pd.DataFrame(dataa, columns=['Drug_name'])
if database == 'MDAD':
    drug = drug[0:1373]
elif database == 'aBiofilm':
    drug = drug[0:1720]

################################################
########  Get the corresponding SMILES  ########
################################################

def replace_url(element):
    return url.format(element)

url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/txt"

new_urls = drug['Drug_name'].apply(lambda x: replace_url(x))

smi = []

num = 0
index = []
for url in new_urls:
    try:
        response = requests.get(url, verify=False)
        time.sleep(0.01)
        if len(response.text) == 106:
            num += 1
            smi.append(0)
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        else:
            smi.append(response.text)
            num += 1
            index.append(num)
        print(response.text)
    except Exception as e:
        print("Abnormalities:", str(e))
        smi.append("")

drug['SMILES'] = smi
print(index)
drug.to_excel("./" + database + "_data/drugSMILES.xlsx", index=False)
