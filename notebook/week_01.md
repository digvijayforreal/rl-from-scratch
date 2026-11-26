## 2026-11-24 — first multi-game PPO run
Model: SmallMultiGameNet, 109,239 params, seed 0, 600k steps, ~11 min (dashboard)
Eval (200 eps, seed 999): catch 1.00 (random -0.78), reach 0.94 (random -0.16)
Both at optimal. Final entropy 0.246, dormant neurons 3.1%.
Checkpoint: results/dashboard_u0585_20261124-162050.pt

## 2026-11-25 - adding Maze and Snake, first multi-game comparisons
All runs: SmallMultiGameNet (109,239 params), seed 0, 4 copies of each game, tested on 500 fresh
games per game (seed 999). "Skill" = (score - random) / (best - random). Models are in results/capacity/.
Reference scores: maze random -0.31 / best +0.94 (estimated from 2000 mazes); snake random -1.15 /
best +0.71 (scripted BFS player, 300 games).

Mistake and fix: the first Snake had no step cost. Alone it scored +0.03 after 400k moves and the
diagnosis showed 200/200 timeouts and 0.08 apples per game: it learned to stall (0 beats risking a -1
crash). Even with the apple next to the head it went up with 98% probability. Fix: -0.01 per step,
like Reach and Maze. Two runs were thrown away (Snake alone, and the first 4-game run).

Sampled play (the way it trains), skill per game, 400k moves per game unless noted:
| trained how                              | catch | reach | maze | snake |
| catch+reach (published, 300k each)       | 100%  | 100%  |  -   |  -    |
| catch+reach+maze, from scratch           | 100%  | 100%  | 66%  |  -    |
| all four, from scratch                   |  97%  |  99%  | 65%  |  9%   |
| snake alone                              |   -   |   -   |  -   | 44%   |
| published model, then only maze+snake    |  79%  |  99%  | 56%  | 58%   |
Maze alone: did not learn in 400k (best-move play reached the goal in 11/200 games).

What surprised me:
- Best-move play is much worse than sampled play on Maze (g3 model: 24% vs 66%). The network has no
  memory, so when its best move is into a wall the board does not change and it repeats the bump.
- Continuing from the published model gave Snake a big head start (58% vs 9%). That is the opposite
  of plasticity loss, so the plasticity story needs its own proper experiment (more seeds).
- Training on only the new games made Catch drop from 100% to 79%. Reach did not drop, probably
  because Maze is Reach with walls.
- Maze only learned when trained next to Reach.

Not answered (runs stopped before finishing):
- Does Snake reach a high score in the 4-game mix with 1M moves per game? (400k was clearly too few.)
- Does continuing from the published model on all four games, mixed, keep Catch at 100%?
- Is Snake's slowness the network running out of room, or just needing more moves? Needs a longer
  run, then the medium network (about 432k params) for comparison.
Everything above is one seed. Do not quote it as a result without 5+ seeds.
