import torch as th
import torch.nn as nn
import torch.nn.functional as F


class OffPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(OffPGCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape + self.n_agents * self.n_actions, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, inputs, actions):
        inputs = th.cat((inputs, actions), dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch):
        return batch["state"][:]

    def _build_target_inputs(self, batch):
        return batch["next_state"][:]

    def critic_build_inputs_raw(self, states, obs, bs):
        inputs = []
        # state, obs, action
        inputs.append(states[:].unsqueeze(1).repeat(1, self.n_agents, 1))
        inputs.append(obs[:])
        # agent id
        inputs.append(th.eye(self.n_agents, device='cuda:0' if self.args.use_cuda else 'cpu').unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        return input_shape