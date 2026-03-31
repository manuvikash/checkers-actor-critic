# checkers_6x6_v0

| Import               | `from mycheckersenv import env` |
|----------------------|--------------------------------|
| Actions              | `Discrete(1296)` (encoded move) |
| Parallel API         | No                              |
| Manual Control       | No                              |
| Agents               | `['player_0', 'player_1']`     |
| Agents               | 2                               |

6×6 checkers on dark squares only, `(row + col) % 2 == 0`. **player_0** starts at the top (rows 0–1), moves toward increasing row; **player_1** starts at the bottom (rows 4–5), moving toward decreasing row. The two center ranks start empty so pieces have legal forward moves. Men move/capture diagonally forward; kings move/capture on all four diagonals (one step per move or jump). **Captures are mandatory** when any capture exists; among captures, a sequence must **maximize total jumps** (ties allowed). If a man becomes a king by landing on the far rank during a jump, **that jump ends the turn** (no further jumps with the new king). You win by **capturing all opponent pieces** or if the opponent has **no legal move** on their turn. Episodes **truncate** after `max_moves` (default 400) with draw-style zero terminal reward.

### Observation space

Dictionary observation per PettingZoo action-masking convention:

| Key             | Shape        | Dtype   | Description |
|----------------|-------------|---------|-------------|
| `observation`  | `(6, 6, 4)` | `float32` | Plane 0: acting agent’s men; plane 1: acting agent’s kings; plane 2: opponent men; plane 3: opponent kings (values in `{0,1}`). |
| `action_mask`  | `(1296,)`   | `int8` | `1` iff the action index is legal **for the currently acting agent**; mask is all zeros when `observe` is called for a non-acting agent. |

Action index decodes as `from_cell * 36 + to_cell` with `cell = row * 6 + col`. A legal action is a **single-step slide** or a **complete multi-jump**; `(from, to)` picks one canonical path stored in the environment.

### Action space

| Label | Action                             |
|------:|------------------------------------|
| `a`   | Move or jump encoded as `from_cell * 36 + to_cell`, with `from_cell`, `to_cell` in `[0, 35]`. |

### Rewards

| Situation | Reward |
|-----------|--------|
| Win (capture all opponent men/kings **or** opponent stalemated with no legal move) | `+1` |
| Loss | `-1` |
| Illegal move (wrapped env) | current player `-1`, episode ends |
| Optional dense capture shaping | `capture_reward × (# pieces captured this turn)`, default `0` in env constructor; runner uses a small positive default for learning |
| Draw / timeout (`max_moves`) | `0` (truncation) |

Rewards for a step are applied to **both** players when the game ends on that step; intermediate capture shaping applies only to the acting agent.

### Termination

- **Termination:** all pieces of one side removed, or the side to move has no legal actions, or an illegal action under `TerminateIllegalWrapper`.
- **Truncation:** move count reaches `max_moves`.

### Usage

```python
from mycheckersenv import env

e = env(render_mode="ansi")
e.reset(seed=42)

for agent in e.agent_iter():
    observation, reward, termination, truncation, info = e.last()
    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        # sample only where mask == 1
        action = e.action_space(agent).sample(mask)
    e.step(action)

e.close()
```

### Training (self-play)

```bash
pip install -r requirements.txt
python myrunner.py --episodes 2000 --log-every 50 --save checkers_ac.pt
python myrunner.py --demo   # one exploratory game with ASCII board
```

### Dependencies

See `requirements.txt`. The course allows **torch 2.8.0** (or the listed tensorflow/scikit-learn alternatives) for function approximation; this repo uses Torch in `myagent.py`.

### Note

Course text may mention `README.py`; PettingZoo-style env documentation is this **`README.md`**.
