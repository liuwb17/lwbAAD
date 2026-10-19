"""Default rectangle-layout problem prompt shared by providers."""
PROBLEM='''Design a standalone C++17 optimizer. Input: n in [50,200], then n lines of
distinct integer x_i,y_i in [0,9999] and positive integer target areas r_i summing
to 100000000. Output n integer rectangles a_i b_i c_i d_i in input order,
0<=a_i<c_i<=10000, 0<=b_i<d_i<=10000, with no positive-area overlaps.
If a rectangle contains its point (x_i+0.5,y_i+0.5), satisfaction is
1-(1-min(area_i,r_i)/max(area_i,r_i))^2; otherwise satisfaction is zero.
External score is round(1e9 * mean satisfaction). Maximize generalizable score.
Read stdin; write only n rectangles to stdout. Standard C++17 library only.
Your full program must parse argv[1] as its wall-clock search budget in seconds
(default 4.8), supporting both 0.6 and 4.8 second evaluations. Finish promptly,
including input/output and setup: hard process limit is min(5.0,budget+0.15).
One CPU, 512 MiB. No files, networking, system commands, environment variables,
runtime compilation, known-input lookup, or evaluator inspection. Do not rely on
hardware cycle-frequency constants. Algorithm randomness must be reproducible.
You are responsible for the entire initial algorithm and all subsequent code.
No initial solution or human-written algorithm skeleton is provided.'''
