from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class ContinuousMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.share_param = self.args.obs_agent_id

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs],
                                                            test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        if share_param:
            agent_inputs = self._build_inputs(ep_batch, t)
            agent_outs = self.agent(agent_inputs)
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        else:
            bs = ep_batch.batch_size
            agent_outs = []
            for idx in range(self.n_agents):
                agent_inputs = self.agent[idx]._build_inputs(ep_batch, t, idx)
                out = self.agent(agent_inputs)
                agent_outs.append(out)
            return th.stack(agent_outs, dim=1)

    def init_hidden(self, batch_size):
        return

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        if share_param:
            self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        else:
            self.agent = th.nn.ModuleList([agent_REGISTRY[self.args.agent](input_shape, self.args)
                                           for _ in range(self.n_agents)])

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
