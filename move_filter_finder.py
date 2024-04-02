"""Script for finding move sequence filters which filter out
symmetric move sequences.
"""

import argparse
import re
from enum import Enum
from itertools import chain
from multiprocessing import cpu_count

import z3

import move_symmetries
from puzzle import MoveSeq
from tools import print_stamped


class Variable:
    """Represents a variable in the filter model."""

    def __init__(self, name: str, step: int):
        """Initializes a new variable given its name and step."""
        self.name = name
        self.s = step

    def ref(self):
        """Returns a reference to the Z3 boolean."""
        return z3.Bool(str(self))

    def __str__(self):
        """Returns a human-readable representation of the variable."""
        return f"{self.name}(s + {self.s})"

    @staticmethod
    def from_str(s: str):
        """Parses a variable from a string."""
        parsed = re.search(r"(.+)\(s \+ (\d+)\)", s)
        if parsed is None:
            raise Exception(f"invalid variable string: {s}")
        return Variable(parsed.group(1), int(parsed.group(2)))


class Operator(Enum):
    """Represents the supported comparison operators."""

    EQ = "=="
    INEQ = "!="
    LT = ">"
    ST = "<"
    LTE = ">="
    STE = "<="


class Condition:
    """Represents a condition in the filter model."""

    def __init__(
        self,
        left: Variable,
        op: Operator,
        right: int | Variable,
    ):
        """Initializes a new condition given its left-hand side, operator,
        and right-hand side. The right hand side can be a variable or integer.
        """
        self.left = left
        self.op = op
        self.right = right

    def ref(self):
        """Returns a reference to the Z3 boolean."""
        return z3.Bool(str(self))

    def __str__(self):
        """Returns a human-readable representation of the condition."""
        return f"{self.left} {self.op.value} {self.right}"

    @staticmethod
    def from_str(s: str):
        """Parses a condition from a string."""
        parsed = re.search(r"(.+) (.+) (.+)", s)
        if parsed is None:
            raise Exception(f"invalid variable string: {s}")
        right_raw = parsed.group(3)
        return Condition(
            Variable.from_str(parsed.group(1)),
            Operator(parsed.group(2)),
            int(right_raw) if right_raw.isnumeric() else Variable.from_str(right_raw),
        )


class FilterComponent:
    """Represents a part of a filter in the filter model."""

    def __init__(
        self,
        name: str,
        domain: list[int],
        k: int,
        solver: z3.Optimize,
    ):
        """Initializes a new filter component given its name, domain,
        and sequence length.
        """
        assert domain == sorted(domain)
        assert len(domain) >= 2
        self.name = name
        self.domain = domain
        self.k = k

        self.vars = [Variable(name, step) for step in range(k)]
        self.conditions: list[list[Condition]] = [[] for _ in range(k)]
        for step in range(k):
            for val in domain:
                self.conditions[step].append(
                    Condition(self.vars[step], Operator.EQ, val)
                )
                self.conditions[step].append(
                    Condition(self.vars[step], Operator.INEQ, val)
                )
            for f in range(step + 1, k):
                self.conditions[step].append(
                    Condition(self.vars[step], Operator.EQ, self.vars[f])
                )
                self.conditions[step].append(
                    Condition(self.vars[step], Operator.INEQ, self.vars[f])
                )
                self.conditions[step].append(
                    Condition(self.vars[step], Operator.LT, self.vars[f])
                )
                self.conditions[step].append(
                    Condition(self.vars[step], Operator.ST, self.vars[f])
                )
                self.conditions[step].append(
                    Condition(self.vars[step], Operator.LTE, self.vars[f])
                )
                self.conditions[step].append(
                    Condition(self.vars[step], Operator.STE, self.vars[f])
                )

        # Disable comparators incompatible with booleans when domain is binary.
        if len(domain) == 2:
            for step in range(k):
                for cond in self.conditions[step]:
                    if cond.op in (
                        Operator.LT,
                        Operator.ST,
                        Operator.LTE,
                        Operator.STE,
                    ):
                        solver.add(z3.Not(cond.ref()))

    def facilitates(self, step: int, vs: list[int]):
        """Returns a Z3 expression that represents whether the filter allows the list
        of values at the specified step.
        """
        conds = []
        for cond in self.conditions[step]:
            right = cond.right if isinstance(cond.right, int) else vs[cond.right.s]
            match cond.op:
                case Operator.EQ:
                    conds.append(z3.Implies(cond.ref(), vs[step] == right))
                case Operator.INEQ:
                    conds.append(z3.Implies(cond.ref(), vs[step] != right))
                case Operator.LT:
                    conds.append(z3.Implies(cond.ref(), vs[step] > right))
                case Operator.ST:
                    conds.append(z3.Implies(cond.ref(), vs[step] < right))
                case Operator.LTE:
                    conds.append(z3.Implies(cond.ref(), vs[step] >= right))
                case Operator.STE:
                    conds.append(z3.Implies(cond.ref(), vs[step] <= right))
        return z3.And(conds)

    def not_facilitates(self, step: int, vs: list[int]):
        """Returns a Z3 expression that represents whether the filter does not allow
        the list of values at the specified step. This is equal to the negation of the
        facilitates function, but is faster since it uses ors.
        """
        conds = []
        for cond in self.conditions[step]:
            right = cond.right if isinstance(cond.right, int) else vs[cond.right.s]
            match cond.op:
                case Operator.EQ:
                    conds.append(z3.And(cond.ref(), vs[step] != right))
                case Operator.INEQ:
                    conds.append(z3.And(cond.ref(), vs[step] == right))
                case Operator.LT:
                    conds.append(z3.And(cond.ref(), vs[step] <= right))
                case Operator.ST:
                    conds.append(z3.And(cond.ref(), vs[step] >= right))
                case Operator.LTE:
                    conds.append(z3.And(cond.ref(), vs[step] < right))
                case Operator.STE:
                    conds.append(z3.And(cond.ref(), vs[step] > right))
        return z3.Or(conds)

    def conditions_from_model(self, m: z3.ModelRef):
        """Get the enabled condition from the model."""
        return [
            cond
            for step in range(self.k)
            for cond in self.conditions[step]
            if z3.is_true(m.get_interp(cond.ref()))
        ]


class Filter:
    """Represents a filter in the filter model."""

    def __init__(self, k: int, solver: z3.Optimize):
        """Initializes a new filter given the sequence length.
        Creates a filter component for ax, hi and dr.
        """
        self.k = k
        self.ax = FilterComponent("ax", [0, 1, 2], k, solver)
        self.hi = FilterComponent("hi", [0, 1], k, solver)
        self.dr = FilterComponent("dr", [0, 1, 2], k, solver)

    def facilitates(self, axs: list[int], his: list[int], drs: list[int]):
        """Returns a Z3 expression that represents whether the filter allows the lists
        of values at the specified step.
        """
        return z3.And(
            [
                z3.And(
                    self.ax.facilitates(step, axs),
                    self.hi.facilitates(step, his),
                    self.dr.facilitates(step, drs),
                )
                for step in range(self.k)
            ]
        )

    def not_facilitates(self, axs: list[int], his: list[int], drs: list[int]):
        """Returns a Z3 expression that represents whether the filter does not allow
        the lists of values at the specified step. This is equal to the negation of the
        facilitates function, but is faster since it uses ors.
        """
        return z3.Or(
            [
                z3.Or(
                    self.ax.not_facilitates(step, axs),
                    self.hi.not_facilitates(step, his),
                    self.dr.not_facilitates(step, drs),
                )
                for step in range(self.k)
            ]
        )

    def facilitates_seq(self, ms: MoveSeq):
        """Wrapper around the facilitates function that takes a move sequence."""
        return self.facilitates(
            [m[0] for m in ms],
            [1 if m[1] else 0 for m in ms],
            [m[2] for m in ms],
        )

    def not_facilitates_seq(self, ms: MoveSeq):
        """Wrapper around the not facilitates function that takes a move sequence."""
        return self.not_facilitates(
            [m[0] for m in ms],
            [1 if m[1] else 0 for m in ms],
            [m[2] for m in ms],
        )

    def all_conditions(self):
        """Return all conditions in the filter."""
        return [
            cond
            for cond in chain(
                *chain(self.ax.conditions),
                *chain(self.hi.conditions),
                *chain(self.dr.conditions),
            )
        ]

    def conditions_from_model(self, m: z3.ModelRef):
        """Get the enabled conditions from the model."""
        return (
            self.ax.conditions_from_model(m)
            + self.hi.conditions_from_model(m)
            + self.dr.conditions_from_model(m)
        )


def find(n: int, k: int):
    """Finds a new move sequence filter for the given puzzle size and sequence length.
    Uses the pre-generated move symmetry files and maximizes the number of filtered
    sequences.
    """
    print_stamped("building model...")

    z3.set_param("parallel.enable", True)
    z3.set_param("parallel.threads.max", cpu_count() - 1)
    solver = z3.Optimize()
    filter = Filter(k, solver)

    # Cumulate all filterable move sequences.
    filterable: list[MoveSeq] = []
    for seq, syms in move_symmetries.load_symmetries(n, k, False).items():
        filterable.extend(syms)

        # If the original move sequence is also of length d,
        # it too can be filtered out, as long as one stays unfiltered.
        if len(seq) == k:
            filterable.append(seq)
            solver.add(z3.Or([filter.not_facilitates_seq(s) for s in [*syms, seq]]))

    if len(filterable) == 0:
        raise Exception("there are no move sequences to filter")

    # Disallow filtering unique move sequences.
    print_stamped("loading unique sequences...")
    not_filterable = move_symmetries.load_unique(n, k, False)
    not_filterable_conditions = [filter.not_facilitates_seq(s) for s in not_filterable]
    print_stamped("ingesting unique sequences...")
    solver.add(z3.And(not_filterable_conditions))
    print_stamped("finished ingesting...")

    # Add the main objective of maximizing the number of filtered sequences.
    filtered_count = z3.Sum(
        [z3.If(filter.facilitates_seq(f), 1, 0) for f in filterable]
    )
    solver.maximize(filtered_count)

    # As a secondary objective, add minimizing the number of conditions.
    cond_count = z3.Sum([z3.If(c.ref(), 1, 0) for c in filter.all_conditions()])
    symb_cond_count = z3.Sum(
        [
            z3.If(c.ref(), 1, 0)
            for c in filter.all_conditions()
            if not isinstance(c.right, int)
        ]
    )
    ste_cond_count = z3.Sum(
        [z3.If(c.ref(), 1, 0) for c in filter.all_conditions() if c.op == Operator.STE]
    )
    lte_cond_count = z3.Sum(
        [z3.If(c.ref(), 1, 0) for c in filter.all_conditions() if c.op == Operator.LTE]
    )
    st_cond_count = z3.Sum(
        [z3.If(c.ref(), 1, 0) for c in filter.all_conditions() if c.op == Operator.ST]
    )
    lt_cond_count = z3.Sum(
        [z3.If(c.ref(), 1, 0) for c in filter.all_conditions() if c.op == Operator.LT]
    )
    ineq_cond_count = z3.Sum(
        [z3.If(c.ref(), 1, 0) for c in filter.all_conditions() if c.op == Operator.INEQ]
    )
    eq_cond_count = z3.Sum(
        [z3.If(c.ref(), 1, 0) for c in filter.all_conditions() if c.op == Operator.EQ]
    )
    solver.minimize(cond_count)
    solver.minimize(symb_cond_count)
    solver.minimize(ste_cond_count)
    solver.minimize(lte_cond_count)
    solver.minimize(st_cond_count)
    solver.minimize(lt_cond_count)
    solver.minimize(ineq_cond_count)
    solver.minimize(eq_cond_count)

    print_stamped("solving...")
    solver.check()

    m = solver.model()
    print_stamped(f"found filter for {m.evaluate(filtered_count)}...")
    print([str(cond) for cond in filter.conditions_from_model(m)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    args = parser.parse_args()
    find(args.n, args.k)
