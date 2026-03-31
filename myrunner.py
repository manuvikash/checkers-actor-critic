"""
Self-play training loop for 6x6 Checkers: both players use the same Actor–Critic.
Optional snapshot opponent every ``snapshot_interval`` episodes; with probability
``mix_prev``, player_1 uses the snapshot policy (older self).
"""
from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from myagent import SelfPlayActorCritic
from mycheckersenv import BOARD_SIZE, NUM_CELLS, env


def load_checkpoint(path: str, learner: SelfPlayActorCritic) -> None:
    """Load weights from ``torch.save`` output (dict with ``model``) or a raw state_dict."""
    try:
        ckpt = torch.load(path, map_location=learner.device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=learner.device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        learner.load_state_dict(ckpt["model"])
    else:
        learner.load_state_dict(ckpt)
    print(f"Loaded weights from {path}")


def format_action_move(action: int) -> str:
    """Decode env action index into from/to board coordinates (row, col)."""
    fr = action // NUM_CELLS
    to = action % NUM_CELLS
    r0, c0 = divmod(fr, BOARD_SIZE)
    r1, c1 = divmod(to, BOARD_SIZE)
    return f"({r0},{c0}) → ({r1},{c1})"


def play_episode(
    environment,
    learner: SelfPlayActorCritic,
    opponent: SelfPlayActorCritic | None,
    mix_prev: float,
    deterministic: bool,
    render: bool,
) -> Tuple[List[Dict], List[int], List[float], List[bool], Dict[str, float]]:
    environment.reset()
    obs_store: List[Dict[str, Any]] = []
    act_store: List[int] = []
    rew_store: List[float] = []
    done_store: List[bool] = []
    latest_cum: Dict[str, float] = {a: 0.0 for a in environment.possible_agents}

    for agent in environment.agent_iter():
        obs, _, term, trunc, _ = environment.last()
        if term or trunc:
            environment.step(None)
            continue

        use_prev = (
            opponent is not None
            and random.random() < mix_prev
            and agent == "player_1"
        )
        policy = opponent if use_prev else learner

        action, _, _ = policy.act(obs, deterministic=deterministic)

        if render and environment.unwrapped.board is not None:
            print(environment.unwrapped._render_ansi())
            move = format_action_move(action)
            print(f"  {agent} -> {move}  (action_id={action})\n")

        obs_store.append(obs)
        act_store.append(action)

        prev = float(environment._cumulative_rewards[agent])
        environment.step(action)
        post = float(environment._cumulative_rewards[agent])
        if render:
            print(f"    step_reward={post - prev:.4f}  cumulative_reward[{agent}]={post:.3f}\n")
        rew_store.append(post - prev)
        done_store.append(False)

        for a in environment.possible_agents:
            if a in environment._cumulative_rewards:
                latest_cum[a] = float(environment._cumulative_rewards[a])

    if done_store:
        done_store[-1] = True

    return obs_store, act_store, rew_store, done_store, latest_cum


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Self-play Actor–Critic on 6x6 Checkers")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--capture-reward",
        type=float,
        default=0.05,
        help="Dense reward per captured piece in one jump segment",
    )
    p.add_argument("--max-moves", type=int, default=400)
    p.add_argument(
        "--snapshot-interval",
        type=int,
        default=50,
        help="Refresh frozen opponent copy every N episodes; 0 disables",
    )
    p.add_argument(
        "--mix-prev",
        type=float,
        default=0.2,
        help="Probability that player_1 uses the snapshot policy",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save", type=str, default="checkers_ac.pt")
    p.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to a .pt file from this runner (loads 'model' weights before running)",
    )
    p.add_argument("--demo", action="store_true", help="Print board each step (one episode)")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Greedy actions (argmax); useful with --demo --load",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="Run this many deterministic games after training",
    )
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    learner = SelfPlayActorCritic(gamma=args.gamma, lr=args.lr)
    if args.load:
        load_checkpoint(args.load, learner)

    opponent: SelfPlayActorCritic | None = None
    if args.snapshot_interval > 0:
        opponent = SelfPlayActorCritic(gamma=args.gamma, lr=args.lr)
        opponent.load_state_dict(learner.state_dict())

    base_kwargs: Dict[str, Any] = dict(
        max_moves=args.max_moves,
        capture_reward=args.capture_reward,
    )

    total_eps = 1 if args.demo else args.episodes

    for ep in range(total_eps):
        base_kwargs["render_mode"] = "ansi" if args.demo else None
        e = env(**base_kwargs)
        demo_eval_only = args.demo and args.load
        observations, actions, rewards, dones, cum = play_episode(
            e,
            learner,
            opponent,
            mix_prev=args.mix_prev,
            deterministic=args.deterministic,
            render=args.demo,
        )
        e.close()

        if len(observations) > 0 and not demo_eval_only:
            stats = learner.update_on_episode(observations, actions, rewards, dones)
        else:
            stats = {"loss": 0.0, "entropy": 0.0, "mean_return": 0.0}

        if (
            args.snapshot_interval > 0
            and opponent is not None
            and (ep + 1) % args.snapshot_interval == 0
        ):
            opponent.load_state_dict(learner.state_dict())

        if not args.demo and (ep + 1) % args.log_every == 0:
            p0 = cum.get("player_0", 0.0)
            p1 = cum.get("player_1", 0.0)
            print(
                f"ep {ep+1}/{args.episodes} "
                f"loss={stats['loss']:.4f} ent={stats['entropy']:.3f} "
                f"cumulative_reward player_0={p0:.3f} player_1={p1:.3f} "
                f"mean_ep_return={stats.get('mean_return', 0):.3f}"
            )

        if args.demo:
            p0 = cum.get("player_0", 0.0)
            p1 = cum.get("player_1", 0.0)
            print(
                "Final cumulative rewards (PettingZoo `last()` / `_cumulative_rewards`): "
                f"player_0={p0:.3f}, player_1={p1:.3f}"
            )

    if args.demo and args.load:
        print(
            "Skipping save (--demo with --load: eval only). "
            "Omit --load on --demo for one training step, or train without --demo to save."
        )
    else:
        torch.save(dict(model=learner.state_dict(), args=vars(args)), args.save)
        print(f"Saved {args.save}")

    if args.eval_episodes > 0:
        wins = [0, 0, 0]
        for _ in range(args.eval_episodes):
            e = env(render_mode=None, max_moves=args.max_moves, capture_reward=0.0)
            _, _, _, _, cum = play_episode(
                e, learner, None, 0.0, deterministic=True, render=False
            )
            e.close()
            if cum.get("player_0", 0) > cum.get("player_1", 0):
                wins[0] += 1
            elif cum.get("player_1", 0) > cum.get("player_0", 0):
                wins[1] += 1
            else:
                wins[2] += 1
        print(f"eval wins p0/p1/draw: {wins[0]}/{wins[1]}/{wins[2]}")


if __name__ == "__main__":
    main()
