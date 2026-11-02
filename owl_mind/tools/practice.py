"""build_hint and analyze_complexity.

Every hint build_hint can return comes from a table an author wrote. The model chooses a rung
and never supplies the words, so this tool cannot emit a solution.
"""

import re
from typing import Any

from langchain_core.tools import tool

MAX_LEVEL = 3

# Rungs, in order: orient -> narrow -> concrete method. Level 3 gives the
# student the first step and the shape of the loop; it stops before the result.
# Keyed by technique family, because "what should I try" has a different answer
# for a graph problem than for a recurrence.
_LADDER: dict[str, tuple[str, str, str]] = {
    "graph": (
        "Start by naming the graph. What is a node here, and what makes two nodes adjacent?",
        "You need an order to visit nodes in. One structure gives you level-by-level order "
        "and another gives you depth-first -- which does this problem's wording ask for?",
        "Before the loop, set up the frontier with the starting node and an empty visited "
        "set. Decide what you record the first time you reach a node. Then trace it by hand "
        "on a three-node example and check the visited set does what you expected.",
    ),
    "dynamic programming": (
        "Can you state the answer for a smaller version of this input in terms of the answer "
        "for an even smaller one? That sentence is the whole problem.",
        "Write the recurrence down, including the base case, before writing any code. What "
        "exactly does your table index mean?",
        "Fill the first three entries of the table by hand. If entry three needs something "
        "you have not computed yet, your iteration order is wrong -- fix the order, not the "
        "recurrence.",
    ),
    "recursion": (
        "What is the smallest input where you already know the answer without recursing?",
        "Assume the function is already correct for smaller inputs. What single step turns "
        "those into the answer for this one?",
        "Check that every recursive call makes the input strictly smaller. Then trace the "
        "call stack for the smallest input above the base case and watch it return.",
    ),
    "sorting": (
        "What property does the output need, and is it a total order or a partial one?",
        "Does this need a full sort, or only the smallest few elements? The answer changes "
        "which technique fits.",
        "Work the first partition or the first merge by hand on six elements, and check your "
        "loop bounds against the element that ends up on the boundary.",
    ),
    "hashing": (
        "What is the key, and what do you need to look up by it?",
        "What happens when two keys land in the same bucket? Your answer to that is the "
        "design decision.",
        "Walk three insertions through your table by hand, including one deliberate "
        "collision, and check what a lookup for the second key does.",
    ),
    "concurrency": (
        "Which piece of state is shared, and which threads write to it?",
        "What sequence of interleavings would produce the wrong answer? Describe one.",
        "Mark every read-modify-write on shared state. Check each one is inside the same "
        "lock, and that the locks are always acquired in the same order.",
    ),
}

# Used when the topic is unknown or absent. Deliberately Socratic rather than
# generic encouragement -- a hint that says "keep trying" spends a tool call
# for nothing.
_DEFAULT: tuple[str, str, str] = (
    "Restate the problem in your own words, including what the input is and what the output "
    "has to satisfy.",
    "What have you tried, and what specifically went wrong? The failure usually names the "
    "technique you need.",
    "Solve the smallest interesting instance by hand, writing down each step. The steps you "
    "took are the algorithm; now check which of them your code is missing.",
)

_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("graph", ("bfs", "dfs", "graph", "dijkstra", "traversal", "shortest path", "tree")),
    ("dynamic programming", ("dynamic programming", "dp", "memo", "knapsack", "subsequence")),
    ("recursion", ("recursion", "recursive", "backtrack", "divide and conquer")),
    ("sorting", ("sort", "quicksort", "mergesort", "heapsort", "partition")),
    ("hashing", ("hash", "dictionary", "map", "set", "two-sum", "two sum")),
    ("concurrency", ("concurrency", "thread", "mutex", "lock", "deadlock", "race")),
)


def family_for(topic: str) -> str:
    """Map a topic to a technique family. Unknown topics get the default ladder."""
    lowered = topic.lower()
    for family, markers in _FAMILIES:
        if any(marker in lowered for marker in markers):
            return family
    return ""


def all_hints() -> frozenset[str]:
    """Every string this tool can return. The no-solution test asserts against it."""
    return frozenset(hint for ladder in (*_LADDER.values(), _DEFAULT) for hint in ladder)


@tool
def build_hint(level: int, topic: str = "") -> dict[str, Any]:
    """Get a hint for a stuck student. Level 1 orients, level 2 narrows to a technique, level 3
    gives a concrete first step. No level gives the solution, and there is no level 4."""
    level = max(1, min(MAX_LEVEL, level))
    family = family_for(topic)
    ladder = _LADDER.get(family, _DEFAULT)
    return {
        "level": level,
        "family": family or "general",
        "hint": ladder[level - 1],
        "next_level_available": level < MAX_LEVEL,
        "withholds_solution": True,
    }


# Pattern-based, and it says so. It reads the source's shape and never runs it. A confident wrong
# answer would be worse than no tool, so the caveat is part of the result.

_LOOP = re.compile(r"^(\s*)(for|while)\b")
_SORT = re.compile(r"\b(sorted|\.sort)\s*\(")
_HALVING = re.compile(r"//\s*2|>>\s*1|/\s*2\b|\bmid\b")
_DEF = re.compile(r"^\s*def\s+(\w+)")


def analyze(code: str) -> dict[str, Any]:
    """Name the dominant complexity class from the source's shape."""
    lines = code.splitlines()
    evidence: list[str] = []

    depth, deepest = 0, 0
    indents: list[int] = []
    for line in lines:
        match = _LOOP.match(line)
        if not match:
            continue
        indent = len(match.group(1))
        while indents and indents[-1] >= indent:
            indents.pop()
        indents.append(indent)
        depth = len(indents)
        deepest = max(deepest, depth)

    names = _DEF.findall(code)
    # A function that calls itself more than once branches; once is linear or
    # logarithmic depending on how the argument shrinks.
    recursive_calls = 0
    if names:
        recursive_calls = len(re.findall(rf"\b{re.escape(names[0])}\s*\(", code)) - 1

    halving = bool(_HALVING.search(code))
    sorts = bool(_SORT.search(code))

    if deepest >= 2:
        complexity = f"O(n^{deepest})"
        evidence.append(f"{deepest} nested loops")
    elif recursive_calls >= 2:
        complexity = "O(2^n)"
        evidence.append(f"{names[0]} calls itself {recursive_calls} times per level")
    elif sorts:
        complexity = "O(n log n)"
        evidence.append("a sort dominates")
    elif deepest == 1 and halving:
        complexity = "O(log n)"
        evidence.append("one loop that halves its range")
    elif deepest == 1:
        complexity = "O(n)"
        evidence.append("one loop over the input")
    elif recursive_calls == 1 and halving:
        complexity = "O(log n)"
        evidence.append("recursion that halves its input")
    elif recursive_calls == 1:
        complexity = "O(n)"
        evidence.append("recursion that shrinks its input by a constant")
    else:
        complexity = "O(1)"
        evidence.append("no loops or recursion found")

    if sorts and deepest >= 1:
        evidence.append("a sort is also present; it is dominated by the loops")

    return {
        "complexity": complexity,
        "evidence": evidence,
        "loop_depth": deepest,
        "caveat": (
            "Pattern-based: read from the shape of this source only. It does not "
            "run the code and cannot see the cost of functions it calls."
        ),
    }


@tool
def analyze_complexity(code: str) -> dict[str, Any]:
    """Name the time-complexity class of a snippet from its structure. It does not run the code
    or see the cost of called functions, so treat the result as evidence to explain."""
    return analyze(code)
