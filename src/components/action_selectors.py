import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
from .noise import OUNoise

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class OUNoiseSelector():

    def __init__(self, args):
        self.args = args

        self.noise = OUNoise(args.action_dim, args.mu, args.theta,
                             args.sigma, args.scale)

        self.action_low = args.action_low
        self.action_high = args.action_high

    def select_action(self, agent_inputs, test_mode=False):
        action = agent_inputs.clone()
        if not test_mode:
            noise = th.Tensor(self.noise.eval())
            action += noise
        action = action.clamp(self.action_low, self.action_high)
        return action


REGISTRY["OU_noise"] = OUNoiseSelector


class NormalNoiseSelector():

    def __init__(self, args):
        self.args = args

        self.action_low = args.action_low
        self.action_high = args.action_high

    def select_action(self, agent_inputs, test_mode=False):
        action = agent_inputs.clone()
        if not test_mode:
            noise = th.randn_like(action)
            action += noise
        action = action.clamp(self.action_low, self.action_high)
        return action


REGISTRY["Normal_noise"] = NormalNoiseSelector
        