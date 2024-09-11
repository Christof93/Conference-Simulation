"""This is a full example of using Tianshou with MARL to train agents, complete with argument parsing (CLI) and logging.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""

import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple, Union, Any
from collections import OrderedDict, deque

import gymnasium
from gymnasium.spaces.utils import flatten_space, flatdim, unflatten
import numpy as np
import torch
import torch.nn as nn
from tianshou.data import Collector, VectorReplayBuffer, Batch, to_numpy
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

import env.reputation_environment as rep_env


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument(
        "--target-mean-rewards",
        type=float,
        default=50,
        help="the expected mean rewards",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=2,
        help="the learned agent plays as the"
        " agent_id-th player. Choices are 1 and 2.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="the path of agent pth file " "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--opponent-path",
        type=str,
        default="",
        help="the path of opponent agent pth file "
        "for resuming from a pre-trained agent",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

class MultiHeadNet(nn.Module):
    def __init__(self, state_shape, action_shapes, hidden_sizes):
        super().__init__()
        layers = [
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
        ]
        for size_in, size_out in zip(hidden_sizes[::2], hidden_sizes[1::2]):
            layers+=[nn.Linear(size_in, size_out), nn.ReLU(inplace=True)]
        self.shared_model = nn.Sequential(*layers)            
        self.heads = nn.ModuleList([nn.Linear(128, np.prod(action_shape)) for action_shape in action_shapes])

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        shared_output = self.shared_model(obs.view(batch, -1))
        logits = torch.cat([head(shared_output) for head in self.heads], dim=-1)
        return logits, state


def get_agents(
    args: argparse.Namespace = get_args(),
    agents: Optional[Tuple[BasePolicy]] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    DQNPolicy.forward = forward
    BasePolicy.map_action = map_action
    env = get_env()
    # observation_space = (
    #     env.observation_space["observation"]
    #     if isinstance(env.observation_space, gymnasium.spaces.Dict)
    #     else env.observation_space
    # )
    args.state_shape = (flatdim(env.observation_space),)
    args.action_shape = (flatdim(env.action_space),)
    if agents is None:
        agent_policies = {}
        for agent in env.agents:
            net = MultiHeadNet(
                args.state_shape,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
            ).to(args.device)
            if optim is None:
                optim = torch.optim.Adam(net.parameters(), lr=args.lr)
            agent_policies[agent] = DQNPolicy(
                net,
                optim,
                args.gamma,
                args.n_step,
                target_update_freq=args.target_update_freq,
                action_space = env.action_space
            )
            agent_policies[agent].separate_actions = get_action_separations(self.action_space)
            if args.resume_path:
                agent_policies[agent].load_state_dict(torch.load(args.resume_path))
        agents = list(agent_policies.values())

    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def get_env(render_mode=None):
    env = PettingZooEnv(rep_env.env({}, render_mode=render_mode))
    return env

def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
    """Map raw network output to action range in gym's env.action_space.

    This function is called in :meth:`~tianshou.data.Collector.collect` and only
    affects action sending to env. Remapped action will not be stored in buffer
    and thus can be viewed as a part of env (a black box action transformation).

    Action mapping includes 2 standard procedures: bounding and scaling. Bounding
    procedure expects original action range is (-inf, inf) and maps it to [-1, 1],
    while scaling procedure expects original action range is (-1, 1) and maps it
    to [action_space.low, action_space.high]. Bounding procedure is applied first.

    :param act: a data batch or numpy.ndarray which is the action taken by
        policy.forward.

    :return: action in the same form of input "act" but remap to the target action
        space.
    """
    if isinstance(self.action_space, gymnasium.spaces.Box) and \
            isinstance(act, np.ndarray):
        # currently this action mapping only supports np.ndarray action
        if self.action_bound_method == "clip":
            act = np.clip(act, -1.0, 1.0)
        elif self.action_bound_method == "tanh":
            act = np.tanh(act)
        if self.action_scaling:
            assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                "action scaling only accepts raw action range = [-1, 1]"
            low, high = self.action_space.low, self.action_space.high
            act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
    if isinstance(self.action_space, gymnasium.spaces.Dict) and \
        isinstance(act, np.ndarray):
        act = [convert_output_to_env_action(self.action_space, deque(env_act)) for env_act in act]
    return act

def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_next = obs.obs if hasattr(obs, "obs") else obs
        logits, hidden = model(obs_next, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        np_tensor = to_numpy(q)
        _, act = get_actions_from_q(self.action_space, np_tensor)
        return Batch(logits=logits, act=act, state=hidden)

def get_action_separations(space):
    return _get_action_separations(space, [])

def _get_action_separations(space, separations):
    for space in space.spaces.values():
        if isinstance(space, gymnasium.spaces.Discrete):
            separations.append((len(separations), space.n))
        if isinstance(space, gymnasium.spaces.Dict):
            separations = _get_action_separations(space, separations)
    return separations

def get_actions_from_q(action_space, q):
    q_remaining, acts = _get_actions_from_q(action_space, q, [])
    assert q_remaining.shape[1] == 0
    selected_actions = [np.argmax(act, axis=1) for act in acts]
    return acts, np.transpose(np.array(selected_actions))

def _get_actions_from_q(action_space, q, acts):
    for space in action_space.spaces.values():
        if isinstance(space, gymnasium.spaces.Discrete):
            acts.append(q[:, :space.n])
            q = q[:, space.n:]
        if isinstance(space, gymnasium.spaces.Dict):
            q, acts = _get_actions_from_q(space, q, acts)
    return q, acts

def _convert_output_to_env_action(action_space, output, valid_action):
    for name, space in action_space.spaces.items():
        if isinstance(space, gymnasium.spaces.Discrete):
            valid_action[name] = output.popleft()
        if isinstance(space, gymnasium.spaces.Dict):
            output, valid_action[name] = _convert_output_to_env_action(space, output, OrderedDict())
    return output, valid_action

def convert_output_to_env_action(action_space, output):
    output, env_action = _convert_output_to_env_action(action_space, output, OrderedDict())
    assert len(output) == 0
    return env_action

def train_agent(
    args: argparse.Namespace = get_args(),
    agents: Optional[Tuple[BasePolicy]] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:
    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== agent setup =========
    policy, optim, agents = get_agents(
        args, agents, optim=optim
    )

    # ======== collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=False,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # ======== tensorboard logging setup =========
    log_path = os.path.join(args.logdir, "academic_reputation", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # ======== callback functions used during training =========
    def save_best_fn(policy):
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, "academic_reputation", "dqn", "policy.pth"
            )
        torch.save(
            policy.policies[agents[args.agent_id - 1]].state_dict(), model_save_path
        )

    def stop_fn(mean_rewards):
        return mean_rewards >= args.target_mean_rewards

    def train_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, args.agent_id - 1]

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    return result, policy.policies[agents[args.agent_id - 1]]


# ======== a test function that tests a pre-trained agent ======
def watch(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
) -> None:
    env = DummyVectorEnv([lambda: get_env(render_mode="human")])
    policy, optim, agents = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent
    )
    policy.eval()
    policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=False)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, args.agent_id - 1].mean()}, length: {lens.mean()}")


if __name__ == "__main__":
    # train the agent and watch its performance in a match!
    args = get_args()
    result, agent = train_agent(args)
    watch(args, agent)