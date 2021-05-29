import torch.nn as nn
import torch.nn.functional as F
import torch as th

from utils.models import MLPBase

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        if self.args.msg_pass:
            self.fc1 = nn.Linear(input_shape + self.args.msg_dim, args.rnn_hidden_dim)
            self.msg_net = MLPBase(input_shape + self.args.msg_dim,self.args.msg_dim)
        else:
            self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs):
        if self.args.msg_pass:
            msg_inputs = inputs.reshape(-1, self.args.n_agents, inputs.shape[-1])
            # msg_inputs = F.normalize(msg_inputs, dim=-1)
            # inputs = F.normalize(inputs, dim=-1)
            msg_up = inputs.new(msg_inputs.shape[0], self.args.msg_dim).zero_()

            for i in reversed(range(self.args.n_agents)):
                msg_up = self.msg_net(th.cat([msg_inputs[:, i], msg_up], dim=-1))
                # msg_up = F.normalize(msg_up, dim=-1)
            msg_down = [msg_up]
            for i in range(self.args.n_agents - 1):
                m_down = self.msg_net(th.cat([msg_inputs[:, i], msg_down[i]], dim=-1))
                # m_down = F.normalize(m_down)
                msg_down.append(m_down)
            msgs = th.stack(msg_down, dim=1).reshape(-1, self.args.msg_dim)
            inputs_n_msg = th.cat([inputs, msgs], dim=-1)
            x = F.relu(self.fc1(inputs_n_msg))
            # x = F.relu(self.fc1(inputs))
        else:
            x = F.relu(self.fc1(inputs))

        # x = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(x))
        q = self.fc3(h)
        # q = self.fc3(x)
        return q
