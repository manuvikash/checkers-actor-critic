"""
6x6 Checkers — PettingZoo AEC environment (American checkers–style).

Dark cells only: (row + col) % 2 == 0. Each side fills **two** back ranks (6 men);
the center stays empty so slides exist on a compact board. Player 0 is top,
moves toward increasing row; player 1 bottom, decreasing row. Kings use four
diagonals; men move/capture forward. Captures are mandatory; among captures,
the move must maximize jump count. Multi-jump ends if a man crowns mid-sequence.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

BOARD_SIZE = 6
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
MAX_ACTIONS = NUM_CELLS * NUM_CELLS

EMPTY = 0
P0_MAN = 1
P0_KING = 2
P1_MAN = 3
P1_KING = 4

DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _cell_index(r: int, c: int) -> int:
    return r * BOARD_SIZE + c


def _index_cell(idx: int) -> Tuple[int, int]:
    return idx // BOARD_SIZE, idx % BOARD_SIZE


def _on_board(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def _is_dark(r: int, c: int) -> bool:
    return (r + c) % 2 == 0

# Which player owns the piece in a specific index
def _owner(piece: int) -> int | None:
    if piece == EMPTY:
        return None
    return 0 if piece <= P0_KING else 1


def _is_king(piece: int) -> bool:
    return piece in (P0_KING, P1_KING)


def _is_mine(piece: int, player: int) -> bool:
    if player == 0:
        return piece in (P0_MAN, P0_KING)
    return piece in (P1_MAN, P1_KING)


def _is_opp(piece: int, player: int) -> bool:
    if player == 0:
        return piece in (P1_MAN, P1_KING)
    return piece in (P0_MAN, P0_KING)


def _forward_dirs(player: int) -> Tuple[Tuple[int, int], ...]:
    return ((1, -1), (1, 1)) if player == 0 else ((-1, -1), (-1, 1))


def env(**kwargs):
    e = raw_env(**kwargs)
    e = wrappers.TerminateIllegalWrapper(e, illegal_reward=-1)
    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    return e


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "checkers_6x6_v0",
        "is_parallelizable": False,
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        max_moves: int = 400,
        capture_reward: float = 0.0,
    ):
        super().__init__()
        EzPickle.__init__(self, render_mode, max_moves, capture_reward)
        self.render_mode = render_mode
        self.max_moves = max_moves
        self.capture_reward = capture_reward

        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]

        self.observation_spaces = {
            a: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(BOARD_SIZE, BOARD_SIZE, 4), dtype=np.float32
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(MAX_ACTIONS,), dtype=np.int8
                    ),
                }
            )
            for a in self.agents
        }
        self.action_spaces = {a: spaces.Discrete(MAX_ACTIONS) for a in self.agents}

        self.board: np.ndarray | None = None
        self._agent_selector: agent_selector | None = None
        self._move_count = 0
        self._legal_moves: List[Tuple[int, int]] = []
        self._legal_paths: Dict[Tuple[int, int], List[Tuple[int, ...]]] = {}

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.board = self._initial_board()
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._move_count = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._refresh_legal_moves()

    def observe(self, agent):
        assert self.board is not None
        idx = self.possible_agents.index(agent)
        mine = idx
        opp = 1 - idx

        obs = np.zeros((BOARD_SIZE, BOARD_SIZE, 4), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = int(self.board[r, c])
                if p == EMPTY:
                    continue
                o = _owner(p)
                king = _is_king(p)
                if o == mine:
                    if king:
                        obs[r, c, 1] = 1.0
                    else:
                        obs[r, c, 0] = 1.0
                elif o == opp:
                    if king:
                        obs[r, c, 3] = 1.0
                    else:
                        obs[r, c, 2] = 1.0

        mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
        if agent == self.agent_selection:
            for fr, to in self._legal_moves:
                mask[fr * NUM_CELLS + to] = 1
        return {"observation": obs, "action_mask": mask}

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        pair = (action // NUM_CELLS, action % NUM_CELLS)

        if pair not in self._legal_paths:
            self.rewards[agent] = 0.0
            self.terminations = {a: True for a in self.agents}
            self._accumulate_rewards()
            self.agent_selection = self._agent_selector.next()
            return

        paths = self._legal_paths[pair]
        path_cells = paths[0]
        captures = self._apply_path(path_cells)

        if self.capture_reward != 0.0 and captures > 0:
            self.rewards[agent] = self.capture_reward * captures
        else:
            self.rewards[agent] = 0.0

        self._move_count += 1
        winner = self._check_winner()
        if winner is not None:
            self.rewards[self.possible_agents[winner]] += 1.0
            self.rewards[self.possible_agents[1 - winner]] -= 1.0
            self.terminations = {a: True for a in self.agents}
            self._accumulate_rewards()
            self.agent_selection = self._agent_selector.next()
            return

        if self._move_count >= self.max_moves:
            self.truncations = {a: True for a in self.agents}
            self._accumulate_rewards()
            self.agent_selection = self._agent_selector.next()
            return

        self._accumulate_rewards()
        self._clear_rewards()
        self.agent_selection = self._agent_selector.next()
        self._refresh_legal_moves()

        next_pl = self.possible_agents.index(self.agent_selection)
        # If you don't have any legal moves, you loose
        if not self._legal_moves:
            self.rewards[self.possible_agents[1 - next_pl]] += 1.0
            self.rewards[self.possible_agents[next_pl]] -= 1.0
            self.terminations = {a: True for a in self.agents}
            self._accumulate_rewards()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("Call render() with render_mode set.")
            return None
        return self._render_ansi()

    def close(self):
        pass

    def _initial_board(self) -> np.ndarray:
        """Two filled ranks per side; center ranks empty so diagonal slides exist."""
        b = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for r in range(2):
            for c in range(BOARD_SIZE):
                if _is_dark(r, c):
                    b[r, c] = P0_MAN
        for r in range(4, BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if _is_dark(r, c):
                    b[r, c] = P1_MAN
        return b

    def _refresh_legal_moves(self):
        assert self.board is not None
        p = self.possible_agents.index(self.agent_selection)
        moves, paths_map = _enumerate_moves(self.board, p)
        self._legal_moves = moves
        self._legal_paths = paths_map

    def _apply_path(self, path: Tuple[int, ...]) -> int:
        assert self.board is not None
        agent = self.agent_selection
        player = self.possible_agents.index(agent)
        coords = [_index_cell(i) for i in path]
        captures = 0
        piece = int(self.board[coords[0]])
        self.board[coords[0]] = EMPTY
        for i in range(len(coords) - 1):
            r0, c0 = coords[i]
            r1, c1 = coords[i + 1]
            dr = r1 - r0
            dc = c1 - c0
            steps = max(abs(dr), abs(dc))
            if steps == 2:
                mid_r, mid_c = r0 + dr // 2, c0 + dc // 2
                if _is_opp(int(self.board[mid_r, mid_c]), player):
                    self.board[mid_r, mid_c] = EMPTY
                    captures += 1
        r_last, c_last = coords[-1]
        self.board[r_last, c_last] = piece
        self._maybe_promote(r_last, c_last)
        return captures

    def _maybe_promote(self, r: int, c: int):
        assert self.board is not None
        p = int(self.board[r, c])
        if p == P0_MAN and r == BOARD_SIZE - 1:
            self.board[r, c] = P0_KING
        elif p == P1_MAN and r == 0:
            self.board[r, c] = P1_KING

    def _check_winner(self) -> int | None:
        assert self.board is not None
        n0 = int(np.sum(np.isin(self.board, [P0_MAN, P0_KING])))
        n1 = int(np.sum(np.isin(self.board, [P1_MAN, P1_KING])))
        if n0 == 0:
            return 1
        if n1 == 0:
            return 0
        return None

    def _render_ansi(self) -> str:
        assert self.board is not None
        sym = {EMPTY: ".", P0_MAN: "o", P0_KING: "O", P1_MAN: "x", P1_KING: "X"}
        lines = []
        for r in range(BOARD_SIZE):
            lines.append(" ".join(sym[int(self.board[r, c])] for c in range(BOARD_SIZE)))
        return "\n".join(lines)


def _enumerate_moves(
    board: np.ndarray, player: int
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], List[Tuple[int, ...]]]]:
    captures: List[Tuple[int, ...]] = []
    slides: List[Tuple[int, ...]] = []

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if not _is_mine(int(board[r, c]), player):
                continue
            piece = int(board[r, c])
            king = _is_king(piece)
            cap_paths = _capture_paths_from(board, r, c, player)
            captures.extend(cap_paths)
            if not cap_paths:
                for path in _slide_moves(board, r, c, player, king):
                    slides.append(path)

    if captures:
        max_jumps = max(len(p) - 1 for p in captures)
        captures = [p for p in captures if len(p) - 1 == max_jumps]
    raw_paths = captures if captures else slides
    paths_map: Dict[Tuple[int, int], List[Tuple[int, ...]]] = {}
    ordered_pairs: List[Tuple[int, int]] = []
    for path in raw_paths:
        fr = path[0]
        to = path[-1]
        key = (fr, to)
        if key not in paths_map:
            ordered_pairs.append(key)
            paths_map[key] = []
        paths_map[key].append(path)

    ordered_pairs.sort()
    return ordered_pairs, paths_map


def _slide_moves(
    board: np.ndarray, r: int, c: int, player: int, king: bool
) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    dirs = DIRECTIONS if king else _forward_dirs(player)
    for dr, dc in dirs:
        r1, c1 = r + dr, c + dc
        if _on_board(r1, c1) and _is_dark(r1, c1) and int(board[r1, c1]) == EMPTY:
            out.append((_cell_index(r, c), _cell_index(r1, c1)))
    return out


def _capture_paths_from(board: np.ndarray, r: int, c: int, player: int) -> List[Tuple[int, ...]]:
    paths_idx: List[List[int]] = []

    def recur(brd: np.ndarray, cr: int, cc: int, path_idx: List[int]):
        piece_here = int(brd[cr, cc])
        dirs_use = DIRECTIONS if _is_king(piece_here) else _forward_dirs(player)
        extended = False
        for dr, dc in dirs_use:
            r_mid, c_mid = cr + dr, cc + dc
            r_land, c_land = cr + 2 * dr, cc + 2 * dc
            if not _on_board(r_land, c_land):
                continue
            midp = int(brd[r_mid, c_mid])
            landp = int(brd[r_land, c_land])
            if not _is_opp(midp, player) or landp != EMPTY:
                continue
            pie = int(brd[cr, cc])
            nboard = brd.copy()
            nboard[cr, cc] = EMPTY
            nboard[r_mid, c_mid] = EMPTY
            npiece = pie
            if pie == P0_MAN and r_land == BOARD_SIZE - 1:
                npiece = P0_KING
            elif pie == P1_MAN and r_land == 0:
                npiece = P1_KING
            promoted = npiece != pie
            nboard[r_land, c_land] = npiece
            land_i = _cell_index(r_land, c_land)
            new_path = path_idx + [land_i]
            if promoted:
                paths_idx.append(new_path)
                extended = True
                continue
            extended = True
            recur(nboard, r_land, c_land, new_path)
        if not extended and len(path_idx) > 1:
            paths_idx.append(list(path_idx))

    start_i = _cell_index(r, c)
    recur(board.copy(), r, c, [start_i])
    return [tuple(p) for p in paths_idx]

