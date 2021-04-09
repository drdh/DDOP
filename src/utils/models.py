import torch.nn as nn
import torch.nn.functional as F

class MLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLPBase, self).__init__()
        self.l1 = nn.Linear(num_inputs, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, num_outputs)

    def forward(self, inputs):
        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = F.normalize(x,dim=-1)
        return x