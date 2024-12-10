from yacs.config import CfgNode as CN
import argparse
import torch
import json
import numpy as np
import os

parser = argparse.ArgumentParser(description="MiDSFN for Drug-Microbe prediction")
parser.add_argument('--cfg', required=True, type=str, help="path to config file", metavar='CFG')
parser.add_argument('--data', required=True, type=str, metavar='TASK', help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task",
                    choices=['MDAD', 'aBiofilm', 'MDAD_aBiofilm', 'MDAD_aBiofilm_discard', 'aBiofilm_MDAD',
                             'MDAD+aBiofilm', 'MDAD+aBiofilm_discard', 'aBiofilm_to_MDAD', 'Case_prediction',
                             'Ethnic_interaction'])
args = parser.parse_args()

_C = CN()

# Drug feature extractor
_C.DRUG = CN()
_C.DRUG.NODE_IN_FEATS = 75

_C.DRUG.PADDING = True

_C.DRUG.HIDDEN_LAYERS = [128, 128, 128]
_C.DRUG.NODE_IN_EMBEDDING = 128
_C.DRUG.MAX_NODES = 290
_C.DRUG.INNER_CHANNELS_1 = 8
_C.DRUG.INNER_CHANNELS_2 =16

# Microbe feature extractor
_C.MICROBE = CN()
_C.MICROBE.NUM_FILTERS = [128, 128, 128]
_C.MICROBE.KERNEL_SIZE = [3, 6, 9]
_C.MICROBE.EMBEDDING_DIM = 128
_C.MICROBE.PADDING = True

if args.split == "MDAD":
    fragment1 = "MDAD_data"
    fragment2 = "microbe_genomic_emb"
elif args.split == "aBiofilm":
    fragment1 = "aBiofilm_data"
    fragment2 = "microbe_genomic_emb"
elif args.split == "MDAD_aBiofilm" or args.split == "MDAD+aBiofilm" or args.split == "Case_prediction" or \
        args.split == "Ethnic_interaction" or args.split == "MDAD_aBiofilm_discard" or \
        args.split == "MDAD+aBiofilm_discard":
    fragment1 = "MDAD_data"
    fragment2 = "microbe_genomic_emb_extra"
elif args.split == "aBiofilm_MDAD" or args.split == "aBiofilm_to_MDAD":
    fragment1 = "aBiofilm_data"
    fragment2 = "microbe_genomic_emb_extra"

with open("./source_dealed_data/" + fragment1 + "/" + fragment2 + "/microbe_index_dict.txt", 'r',
          encoding='utf-8') as f:
    dictionary = json.load(f)
a = np.loadtxt("./source_dealed_data/" + fragment1 + "/" + fragment2 + "/constant_index.txt").tolist()
b = np.loadtxt("./source_dealed_data/" + fragment1 + "/" + fragment2 + "/limit_index.txt").tolist()
c = np.loadtxt("./source_dealed_data/" + fragment1 + "/" + fragment2 + "/microbe_embedding_1.txt").tolist()
d = np.loadtxt("./source_dealed_data/" + fragment1 + "/" + fragment2 + "/microbe_embedding_2.txt").tolist()
_C.MICROBE.GENOMIC_NODES = len(dictionary)
_C.MICROBE.GENOMIC_EMBEDDING_DIM = len(a)
_C.MICROBE.FIXED_ROWS = a
_C.MICROBE.LIMIT_ROWS = b
_C.MICROBE.EMBEDDING_VECTOR_1 = c
_C.MICROBE.EMBEDDING_VECTOR_2 = d
_C.MICROBE.INDEX_DICT = dictionary
_C.MICROBE.TRANSFORMER_HEADS = 2
_C.MICROBE.TRANSFORMER_EMB_DIM = 128

# Simple_Fusion setting
_C.SIMPLE_FUSION = CN()
_C.SIMPLE_FUSION.OUTPUT_FEATS = 0 if args.split == "aBiofilm_MDAD" or args.split == "MDAD_aBiofilm" else 128

# BCN setting
_C.BCN = CN()
_C.BCN.HEADS = 2

# MLP decoder

_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = 2

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 5e-5
_C.SOLVER.DA_LR = 1e-3
_C.SOLVER.SEED = 2048

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result/" + args.split
_C.RESULT.SAVE_MODEL = True

# Domain adaptation
_C.DA = CN()
_C.DA.TASK = False
_C.DA.METHOD = "CDAN"
_C.DA.USE = False
_C.DA.INIT_EPOCH = 10
_C.DA.LAMB_DA = 1
_C.DA.RANDOM_LAYER = False
_C.DA.ORIGINAL_RANDOM = False
_C.DA.RANDOM_DIM = None
_C.DA.USE_ENTROPY = True

# Comet config, ignore it If not installed.
_C.COMET = CN()
# Please change to your own workspace name on comet.
_C.COMET.WORKSPACE = "MagicWings49"
_C.COMET.PROJECT_NAME = "MiDSFN"
_C.COMET.USE = False
_C.COMET.TAG = None

def get_cfg_defaults():
    return _C.clone()
