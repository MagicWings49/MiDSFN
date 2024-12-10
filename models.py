import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from microbe_small_act import SmallActivationNetwork
from microbe_transformer import MicrobeTransformer
import dgl
from fastkan import FastKAN

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


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class DrugBAN(nn.Module):
    def __init__(self, **config):
        super(DrugBAN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        drug_inner_channels_1 = config["DRUG"]["INNER_CHANNELS_1"]
        drug_inner_channels_2 = config["DRUG"]["INNER_CHANNELS_2"]
        microbe_emb_dim = config["MICROBE"]["EMBEDDING_DIM"]
        num_filters = config["MICROBE"]["NUM_FILTERS"]
        kernel_size = config["MICROBE"]["KERNEL_SIZE"]
        bilinear_out_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        microbe_padding = config["MICROBE"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]

        ban_heads = config["BCN"]["HEADS"]
        genomic_nodes = config["MICROBE"]["GENOMIC_NODES"]
        genomic_emb_dim = config["MICROBE"]["GENOMIC_EMBEDDING_DIM"]
        fixed_rows = config["MICROBE"]["FIXED_ROWS"]
        limit_rows = config["MICROBE"]["LIMIT_ROWS"]
        emb_vector1 = config["MICROBE"]["EMBEDDING_VECTOR_1"]
        emb_vector2 = config["MICROBE"]["EMBEDDING_VECTOR_2"]
        dictionary = config["MICROBE"]["INDEX_DICT"]
        self.simple_output_dims = config["SIMPLE_FUSION"]["OUTPUT_FEATS"]
        trans_heads = config["MICROBE"]["TRANSFORMER_HEADS"]
        trans_emb_dim = config["MICROBE"]["TRANSFORMER_EMB_DIM"]
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.microbe_extractor = MicrobeTransformer(embedding_dim=trans_emb_dim, n_heads=trans_heads)
        if self.simple_output_dims > 0:
            self.drug_graphCNN = CustomCNN(drug_inner_channels_1, drug_inner_channels_2, self.simple_output_dims)
            self.dictionary = dictionary
            self.microbe_genomicResNet = SmallActivationNetwork(microbe_nodes=genomic_nodes,
                                                                microbe_emb_dim=genomic_emb_dim, fixed_rows=fixed_rows,
                                                                limit_rows=limit_rows, m_e_1=emb_vector1, m_e_2=emb_vector2,
                                                                output_dim=self.simple_output_dims)

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=bilinear_out_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.decoder_name = config["DECODER"]["NAME"]
        self.mlp_classifier = MLPDecoder(in_dim=bilinear_out_dim, in_dim2=self.simple_output_dims, hidden_dim=mlp_hidden_dim,
                                         out_dim=mlp_out_dim, binary=out_binary)
        self.kan_classifier = FastKAN(layers_hidden=[bilinear_out_dim + self.simple_output_dims, 512, 256, out_binary])

    def forward(self, bg_d, v_p, a_m, m_n, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_p = self.microbe_extractor(v_p)
        f, att = self.bcn(v_d, v_p)
        if self.simple_output_dims > 0:
            a_m = a_m.view(-1, 1, 330, 330)
            d_e = self.drug_graphCNN(a_m)
            m_n = [self.dictionary.index(m_n[i]) for i in range(len(m_n))]
            m_n = torch.tensor(m_n)
            m_e = self.microbe_genomicResNet(m_n)
            ff = d_e*m_e
            f = torch.cat((f, ff), 1)

        if self.decoder_name == "KAN":
            score = self.kan_classifier(f)
        if self.decoder_name == "MLP":
            score = self.mlp_classifier(f)

        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class MicrobeCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(MicrobeCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(73, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(73, embedding_dim)
        self.activation = nn.Linear(37, 90, bias=False)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.activation(v)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v

class CustomCNN(nn.Module):
    def __init__(self, inner_channels_1, inner_channels_2, output_dim):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=inner_channels_1, kernel_size=2, stride=2, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(in_channels=inner_channels_1, out_channels=inner_channels_2, kernel_size=3, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.fc = nn.Linear(inner_channels_2 * 9 * 9, output_dim)

    def forward(self, x):

        x = F.silu(self.conv1(x))
        x = self.pool1(x)
        x = F.silu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        #print(x[0])
        return x

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, in_dim2, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim+in_dim2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
