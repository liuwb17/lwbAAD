AHC001 / AtCoder Ad. Read n (50..200), then n lines x_i y_i r_i.
Place n axis-aligned, positive-area, integer-coordinate rectangles a b c d inside
[0,10000]^2, in input order. No positive-area overlap; touching edges is allowed.
The rectangle for i should contain (x_i+0.5,y_i+0.5); otherwise its satisfaction is 0.
For contained anchors, p_i = 1 - (1 - min(area_i,r_i)/max(area_i,r_i))^2.
Score = round(1e9 * mean(p_i)), higher is better. Sum r_i = 1e8.
Read stdin, write only the n rectangles to stdout. Standard Python library only.
The official time limit is 5 seconds; local configuration may impose less.
All evaluator files and case splits are immutable and outside your solution.
Optimize an algorithm that generalizes, never hardcode cases or inspect the filesystem.

Useful directions: balanced growth, target-area priority, spatial clearance,
rectangle translations/shrink-expand moves, simulated annealing, ruin-and-recreate,
recursive spatial partitioning, alternative initialization, reduced collision-check costs.
These are ideas to explore, not claims that any one method will work best.
