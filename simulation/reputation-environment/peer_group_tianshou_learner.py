"""Tianshou learner for PeerGroupEnvironment (PettingZoo ParallelEnv wrapper).

This script mirrors the structure of tianshou_learner.py but targets
reputation-environment/env/peer_group_environment.PeerGroupEnvironment
with Dict observation/action spaces.
"""

import argparse
import os
from collections import OrderedDict, deque
from copy import deepcopy
from typing import Any, Optional, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch.nn as nn
from env.peer_group_environment import PeerGroupEnvironment
from gymnasium.spaces.utils import flatdim
from tianshou.data import (Batch, Collector, ReplayBuffer, VectorReplayBuffer,
                           to_numpy)
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
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
    parser.add_argument("--target-mean-rewards", type=float, default=50)
    parser.add_argument("--watch", default=False, action="store_true")
    parser.add_argument("--agent-id", type=int, default=1)
    parser.add_argument("--resume-path", type=str, default="")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # PeerGroupEnvironment knobs
    parser.add_argument("--start-agents", type=int, default=20)
    parser.add_argument("--max-agents", type=int, default=80)
    parser.add_argument("--max-peer-group-size", type=int, default=60)
    parser.add_argument("--n-groups", type=int, default=4)
    parser.add_argument("--n-projects", type=int, default=6)
    parser.add_argument("--max-projects-per-agent", type=int, default=6)
    parser.add_argument("--max-agent-age", type=int, default=1000)
    parser.add_argument("--max-rewardless-steps", type=int, default=50)
    parser.add_argument("--growth-rate", type=float, default=0.02)
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


class MultiHeadNet(nn.Module):
    def __init__(self, state_shape, action_shapes, hidden_sizes):
        super().__init__()
        layers = [nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True)]
        for size_in, size_out in zip(hidden_sizes[::2], hidden_sizes[1::2]):
            layers += [nn.Linear(size_in, size_out), nn.ReLU(inplace=True)]
        self.shared_model = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            [nn.Linear(128, np.prod(action_shape)) for action_shape in action_shapes]
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        shared_output = self.shared_model(obs.view(batch, -1))
        logits = torch.cat([head(shared_output) for head in self.heads], dim=-1)
        return logits, state


def get_env(render_mode=None):
    # Instantiate PeerGroupEnvironment and wrap as PettingZooEnv
    env = PeerGroupEnvironment(
        start_agents=get_args().start_agents,
        max_agents=get_args().max_agents,
        max_peer_group_size=get_args().max_peer_group_size,
        n_groups=get_args().n_groups,
        n_projects=get_args().n_projects,
        max_projects_per_agent=get_args().max_projects_per_agent,
        max_agent_age=get_args().max_agent_age,
        max_rewardless_steps=get_args().max_rewardless_steps,
        growth_rate=get_args().growth_rate,
        render_mode=render_mode,
    )
    return PettingZooEnv(env)


def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
    # Map the raw network output to the action space (Box or Dict)
    if isinstance(self.action_space, gymnasium.spaces.Box) and isinstance(
        act, np.ndarray
    ):
        if self.action_bound_method == "clip":
            act = np.clip(act, -1.0, 1.0)
        elif self.action_bound_method == "tanh":
            act = np.tanh(act)
        if self.action_scaling:
            assert np.min(act) >= -1.0 and np.max(act) <= 1.0
            low, high = self.action_space.low, self.action_space.high
            act = low + (high - low) * (act + 1.0) / 2.0
    if isinstance(self.action_space, gymnasium.spaces.Dict) and isinstance(
        act, np.ndarray
    ):
        act = [
            convert_output_to_env_action(self.action_space, deque(env_act))
            for env_act in act
        ]
    return act


def forward(
    self,
    batch: Batch,
    state: Optional[Union[dict, Batch, np.ndarray]] = None,
    model: str = "model",
    input: str = "obs",
    **kwargs: Any,
) -> Batch:
    model = getattr(self, model)
    obs = batch[input]
    obs_next = obs.obs if hasattr(obs, "obs") else obs
    logits, hidden = model(obs_next, state=state, info=batch.info)
    q = self.compute_q_value(logits, getattr(obs, "mask", None))
    if not hasattr(self, "max_action_num"):
        self.max_action_num = q.shape[1]
    np_tensor = to_numpy(q)
    q_vals, act = get_actions_from_q(self.action_space, np_tensor)
    return Batch(logits=logits, act=act, state=hidden)


def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
    for sub_act, logit_split_indices in zip(
        batch.act, get_action_separations(self.action_space)
    ):
        sub_batch = Batch(
            act=sub_act,
            logits=batch.logits[:, np.arange(*logit_split_indices)],
            state=batch.state,
        )
        sub_batch = self.compute_nstep_return(
            batch,
            buffer,
            indices,
            self._target_q,
            self._gamma,
            self._n_step,
            self._rew_norm,
        )
        batch.returns += sub_batch.returns / len(batch.act)
    return batch


def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
    batch = buffer[indices]
    result = self(batch, input="obs_next")
    if self._target:
        target_q = self(batch, model="model_old", input="obs_next").logits
    else:
        target_q = result.logits
    if self._is_double:
        target_qs = []
        for act, action_indices in zip(
            result.act.T, get_action_separations(self.action_space)
        ):
            per_action_target_q = target_q[:, np.arange(*action_indices)][
                np.arange(act.shape[0]), act
            ]
            target_qs.append(per_action_target_q)
        return target_qs
    else:
        return target_q.max(dim=1)[0]


def get_action_separations(space):
    return _get_action_separations(space, [])


def _get_action_separations(space, separations):
    for space in space.spaces.values():
        if isinstance(space, gymnasium.spaces.Discrete):
            current = sum([b - a for a, b in separations]) if separations else 0
            separations.append((current, current + space.n))
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
            acts.append(q[:, : space.n])
            q = q[:, space.n :]
        if isinstance(space, gymnasium.spaces.Dict):
            q, acts = _get_actions_from_q(space, q, acts)
    return q, acts


def _convert_output_to_env_action(action_space, output, valid_action):
    for name, space in action_space.spaces.items():
        if isinstance(space, gymnasium.spaces.Discrete):
            valid_action[name] = output.popleft()
        if isinstance(space, gymnasium.spaces.Dict):
            output, valid_action[name] = _convert_output_to_env_action(
                space, output, OrderedDict()
            )
    return output, valid_action


def convert_output_to_env_action(action_space, output):
    output, env_action = _convert_output_to_env_action(
        action_space, output, OrderedDict()
    )
    assert len(output) == 0
    return env_action


def get_agents(
    args: argparse.Namespace = get_args(),
    agents: Optional[Tuple[BasePolicy]] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    DQNPolicy.forward = forward
    DQNPolicy._target_q = _target_q
    BasePolicy.map_action = map_action
    BasePolicy.process_fn = process_fn

    env = get_env()
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
                action_space=env.action_space,
            )
            agent_policies[agent].separate_actions = get_action_separations(
                env.action_space
            )
            if args.resume_path:
                agent_policies[agent].load_state_dict(torch.load(args.resume_path))
        agents = list(agent_policies.values())

    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    agents: Optional[Tuple[BasePolicy]] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    policy, optim, agents = get_agents(args, agents, optim=optim)

    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=False,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    log_path = os.path.join(args.logdir, "peer_group", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        model_save_path = os.path.join(args.logdir, "peer_group", "dqn", "policy.pth")
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


if __name__ == "__main__":
    args = get_args()
    result, agent = train_agent(args)
