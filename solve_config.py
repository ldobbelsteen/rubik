from multiprocessing import cpu_count
from typing import ClassVar


def gods_number(n: int):
    """Return the known God number for a specific n."""
    match n:
        case 1:
            return 0
        case 2:
            return 11
        case 3:
            return 20
        case _:
            raise Exception(f"God's number not known for n = {n}")


class Tactics:
    """A set of tactics to be used by the Z3 solver. This is useful for more
    compactly respresenting a set of tactics.
    """

    TACTIC_ENCODINGS: ClassVar[dict[str, str]] = {
        "a": "aig",
        "bb": "bit-blast",
        "bti": "blast-term-ite",
        "c2b": "card2bv",
        "cs": "ctx-simplify",
        "css": "ctx-solver-simplify",
        "ds": "dom-simplify",
        "esb": "elim-small-bv",
        "eti": "elim-term-ite",
        "e2b": "eq2bv",
        "l2c": "lia2card",
        "l2p": "lia2pb",
        "nb": "normalize-bounds",
        "p2b": "pb2bv",
        "pbb": "propagate-bv-bounds",
        "pi": "propagate-ineqs",
        "pv": "propagate-values",
        "pa": "purify-arith",
        "sp": "sat-preprocess",
        "s": "simplify",
        "se": "solve-eqs",
    }

    def __init__(self, tactics: list[str]):
        """Create a new set of tactics."""
        self.tactics = tactics

    def get(self):
        """Return the tactics."""
        return self.tactics

    @staticmethod
    def from_str(s: str):
        """Create a new set of tactics from a string."""
        tactics_encoded = s.split(";")
        tactics = [Tactics.TACTIC_ENCODINGS[t] for t in tactics_encoded]
        return Tactics(tactics)

    def __str__(self):
        """Return a string representation of the tactics."""
        return ";".join(
            [
                list(self.TACTIC_ENCODINGS.keys())[
                    list(self.TACTIC_ENCODINGS.values()).index(v)
                ]
                for v in self.tactics
            ]
        )


class SolveConfig:
    """Configuration for a solve operation that can be passed to the solver."""

    def __init__(
        self,
        tactics=Tactics.from_str("se;s;ds;sp"),
        move_size=2,
        max_solver_threads=cpu_count() - 1,
        enable_n2_move_filters_1_and_2=True,
        enable_n3_move_filters_1_and_2=True,
        enable_n3_move_filters_3_and_4=True,
        apply_theorem_11a=False,
        apply_theorem_11b=False,
        ban_repeated_states=False,
        k_search_start=8,
        enable_corner_min_patterns=False,
        enable_edge_min_patterns=False,
    ):
        """Create a new solve configuration."""
        self.tactics = tactics
        self.move_size = move_size
        self.max_solver_threads = max_solver_threads
        self.enable_n2_move_filters_1_and_2 = enable_n2_move_filters_1_and_2
        self.enable_n3_move_filters_1_and_2 = enable_n3_move_filters_1_and_2
        self.enable_n3_move_filters_3_and_4 = enable_n3_move_filters_3_and_4
        self.apply_theorem_11a = apply_theorem_11a
        self.apply_theorem_11b = apply_theorem_11b
        self.ban_repeated_states = ban_repeated_states
        self.k_search_start = k_search_start
        self.enable_corner_min_patterns = enable_corner_min_patterns
        self.enable_edge_min_patterns = enable_edge_min_patterns
