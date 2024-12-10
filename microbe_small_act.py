import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SmallActivationNetwork(nn.Module):
    def __init__(self, microbe_nodes, microbe_emb_dim, fixed_rows, limit_rows, m_e_1, m_e_2, output_dim, trainable_rows=None):
        super(SmallActivationNetwork, self).__init__()
        self.output_dim = output_dim
        self.microbe_genomic_emb = torch.FloatTensor([[(m_e_1[i][j] + m_e_2[i][j]) / 2 for j in range(len(m_e_1[i]))] for i in range(len(m_e_1))])
        self.linear_layer = nn.Linear(microbe_emb_dim, microbe_nodes)
        nn.init.xavier_normal_(self.linear_layer.weight,
                               gain=((microbe_nodes + microbe_emb_dim) / (microbe_nodes * 2)) ** 0.5)
        #self.identity = nn.Identity()

        # The additional vector of those microorganisms with gene strips is fixed to behavior 0 and does not participate in the gradient calculation.
        with torch.no_grad():
            for i in fixed_rows:
                j = int(i - 1)
                self.linear_layer.weight.data[j].zero_()

        # Setting up trainable vector inputs for microorganisms for which no gene strips were found.
        if trainable_rows is not None:
            for i in range(self.linear_layer.weight.size(0)):
                j = int(i - 1)
                self.linear_layer.weight.data[j].requires_grad = (i in trainable_rows and i not in fixed_rows)

        # Set additional limit-size trainable vectors for microorganisms for which no gene bars were found but which have the same species of microorganisms present in this experiment.
        for i, row in enumerate(self.linear_layer.weight):
            if i+1 in limit_rows:
                limit = abs(row[0]/microbe_emb_dim)
                row.data = torch.clamp(row.data, -limit, limit)

        self.fc1 = nn.Linear(microbe_emb_dim, microbe_emb_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(microbe_emb_dim, affine=True, track_running_stats=True) #, weight_atol=1e-3)


        self.fc2 = nn.Linear(microbe_emb_dim, microbe_emb_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(microbe_emb_dim, affine=True, track_running_stats=True) #, weight_atol=1e-3)

        self.fc_out = nn.Linear(microbe_emb_dim, output_dim)

    def forward(self, indices):
        x = torch.index_select(self.microbe_genomic_emb + self.linear_layer.weight, dim=0, index=indices)
        #np.savetxt('./result/Case_prediction/microbal_genomic_embedding.txt', (self.microbe_genomic_emb + self.linear_layer.weight).data.numpy())
        x1 = self.bn1(self.fc1(x))
        x1 = torch.tanh(x1)
        x1 = x1 + x
        x2 = self.bn2(self.fc2(x1))
        x2 = torch.tanh(x2)
        x2 = x2 + x1

        out = self.fc_out(x2)
        return out
