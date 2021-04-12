import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):
    """It will be used for Deterministic DOP"""
    def __init__(self, input_shape, out_fn, args):
        super(MLPAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)
        # self.fc3 = nn.Linear(args.hidden_dim, 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.out_fn = out_fn

    def _build_inputs(self, batch, t, idx):
        # Delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t, idx])
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t, idx]))
            else:
                inputs.append(batch["actions"][:, t-1, idx])

        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        return inputs

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        out = self.out_fn(self.fc3(x))
        return out
