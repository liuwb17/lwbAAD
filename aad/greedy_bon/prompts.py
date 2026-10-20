"""One stateless prompt shared by both simple methods."""
import re

GUIDANCE = """You are an optimization algorithm researcher. Develop a strong algorithm for
the supplied problem under its time and memory limits. Reason about feasible
states, objective structure, search moves, data structures and computational
cost before implementing. You may make structural changes or replace the
approach when useful. Preserve validity on every input; use the available
runtime effectively with a safety margin. Do not optimize for source brevity.
Use only the supplied problem and program; do not reconstruct named submissions.
Return one complete C++17 program in a single cpp code block.
If randomness is used, take the seed from the AAD_RANDOM_SEED environment
variable (default 0), never from clocks, random_device or process identity.
Read standard input and write only the required answer to standard output.
For interactive tasks, obey the interaction protocol and flush output.
"""


def messages(problem: str, parent_source: str | None = None) -> list[dict]:
    # No candidate IDs, scores, slot numbers, method labels, history or donors.
    user = "Problem and execution constraints:\n" + problem
    if parent_source is None:
        user += "\n\nDesign and implement an initial solution from scratch."
    else:
        user += "\n\nImprove the following parent algorithm.\n```cpp\n" + parent_source + "\n```"
    return [{"role": "system", "content": GUIDANCE}, {"role": "user", "content": user}]


def extract_source(response: str) -> str:
    blocks = re.findall(r"```(?:cpp|c\+\+|cxx)?\s*\n(.*?)```", response, re.S)
    if len(blocks) != 1 or not blocks[0].strip():
        raise ValueError("Expected exactly one complete C++ code block")
    return blocks[0].strip() + "\n"
