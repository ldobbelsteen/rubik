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
    def __init__(self, name: str, s: int):
        self.name = name
        self.s = s

    def ref(self):
        return z3.Bool(str(self))

    def __str__(self):
        return f"{self.name}(s + {self.s})"

    @staticmethod
    def from_str(s: str):
        parsed = re.search(r"(.+)\(s \+ (\d+)\)", s)
        if parsed is None:
            raise Exception(f"invalid variable string: {s}")
        return Variable(parsed.group(1), int(parsed.group(2)))


class Operator(Enum):
    EQ = "=="
    INEQ = "!="
    LT = ">"
    ST = "<"
    LTE = ">="
    STE = "<="


class Condition:
    def __init__(
        self,
        left: Variable,
        op: Operator,
        right: int | Variable,
    ):
        self.left = left
        self.op = op
        self.right = right

    def ref(self):
        return z3.Bool(str(self))

    def __str__(self):
        return f"{self.left} {self.op.value} {self.right}"

    @staticmethod
    def from_str(s: str):
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
    def __init__(
        self,
        name: str,
        domain: list[int],
        k: int,
        solver: z3.Optimize,
    ):
        assert domain == sorted(domain)
        assert len(domain) >= 2
        self.name = name
        self.domain = domain
        self.k = k

        self.vars = [Variable(name, s) for s in range(k)]
        self.conditions: list[list[Condition]] = [[] for _ in range(k)]
        for s in range(k):
            for v in domain:
                self.conditions[s].append(Condition(self.vars[s], Operator.EQ, v))
                self.conditions[s].append(Condition(self.vars[s], Operator.INEQ, v))
            for f in range(s + 1, k):
                self.conditions[s].append(
                    Condition(self.vars[s], Operator.EQ, self.vars[f])
                )
                self.conditions[s].append(
                    Condition(self.vars[s], Operator.INEQ, self.vars[f])
                )
                self.conditions[s].append(
                    Condition(self.vars[s], Operator.LT, self.vars[f])
                )
                self.conditions[s].append(
                    Condition(self.vars[s], Operator.ST, self.vars[f])
                )
                self.conditions[s].append(
                    Condition(self.vars[s], Operator.LTE, self.vars[f])
                )
                self.conditions[s].append(
                    Condition(self.vars[s], Operator.STE, self.vars[f])
                )

        # Disable comparators incompatible with booleans when domain is binary.
        if len(domain) == 2:
            for s in range(k):
                for cond in self.conditions[s]:
                    if (
                        cond.op == Operator.LT
                        or cond.op == Operator.ST
                        or cond.op == Operator.LTE
                        or cond.op == Operator.STE
                    ):
                        solver.add(z3.Not(cond.ref()))

    def facilitates(self, s: int, vs: list[int]):
        conds = []
        for cond in self.conditions[s]:
            right = cond.right if isinstance(cond.right, int) else vs[cond.right.s]
            match cond.op:
                case Operator.EQ:
                    conds.append(z3.Implies(cond.ref(), vs[s] == right))
                case Operator.INEQ:
                    conds.append(z3.Implies(cond.ref(), vs[s] != right))
                case Operator.LT:
                    conds.append(z3.Implies(cond.ref(), vs[s] > right))
                case Operator.ST:
                    conds.append(z3.Implies(cond.ref(), vs[s] < right))
                case Operator.LTE:
                    conds.append(z3.Implies(cond.ref(), vs[s] >= right))
                case Operator.STE:
                    conds.append(z3.Implies(cond.ref(), vs[s] <= right))
        return z3.And(conds)

    def not_facilitates(self, s: int, vs: list[int]):
        conds = []
        for cond in self.conditions[s]:
            right = cond.right if isinstance(cond.right, int) else vs[cond.right.s]
            match cond.op:
                case Operator.EQ:
                    conds.append(z3.And(cond.ref(), vs[s] != right))
                case Operator.INEQ:
                    conds.append(z3.And(cond.ref(), vs[s] == right))
                case Operator.LT:
                    conds.append(z3.And(cond.ref(), vs[s] <= right))
                case Operator.ST:
                    conds.append(z3.And(cond.ref(), vs[s] >= right))
                case Operator.LTE:
                    conds.append(z3.And(cond.ref(), vs[s] < right))
                case Operator.STE:
                    conds.append(z3.And(cond.ref(), vs[s] > right))
        return z3.Or(conds)

    def conditions_from_model(self, m: z3.ModelRef):
        result: list[Condition] = []
        for s in range(self.k):
            for cond in self.conditions[s]:
                if z3.is_true(m.get_interp(cond.ref())):
                    result.append(cond)
        return result


class Filter:
    def __init__(self, k: int, solver: z3.Optimize):
        self.k = k
        self.ax = FilterComponent("ax", [0, 1, 2], k, solver)
        self.hi = FilterComponent("hi", [0, 1], k, solver)
        self.dr = FilterComponent("dr", [0, 1, 2], k, solver)

    def facilitates_seq(self, ms: MoveSeq):
        return self.facilitates(
            [m[0] for m in ms],
            [1 if m[1] else 0 for m in ms],
            [m[2] for m in ms],
        )

    def not_facilitates_seq(self, ms: MoveSeq):
        return self.not_facilitates(
            [m[0] for m in ms],
            [1 if m[1] else 0 for m in ms],
            [m[2] for m in ms],
        )

    def facilitates(self, axs: list[int], his: list[int], drs: list[int]):
        return z3.And(
            [
                z3.And(
                    self.ax.facilitates(s, axs),
                    self.hi.facilitates(s, his),
                    self.dr.facilitates(s, drs),
                )
                for s in range(self.k)
            ]
        )

    def not_facilitates(self, axs: list[int], his: list[int], drs: list[int]):
        return z3.Or(
            [
                z3.Or(
                    self.ax.not_facilitates(s, axs),
                    self.hi.not_facilitates(s, his),
                    self.dr.not_facilitates(s, drs),
                )
                for s in range(self.k)
            ]
        )

    def all_conditions(self):
        return [
            cond
            for cond in chain(
                *chain(self.ax.conditions),
                *chain(self.hi.conditions),
                *chain(self.dr.conditions),
            )
        ]

    def conditions_from_model(self, m: z3.ModelRef):
        return (
            self.ax.conditions_from_model(m)
            + self.hi.conditions_from_model(m)
            + self.dr.conditions_from_model(m)
        )


def find(n: int, k: int):
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
            solver.add(z3.Or([filter.not_facilitates_seq(s) for s in syms + [seq]]))

    if len(filterable) == 0:
        raise Exception("there are no move sequences to filter")

    # Disallow filtering unique move sequences.
    for seq in move_symmetries.load_unique(n, k, False):
        solver.add(filter.not_facilitates_seq(seq))

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
    eq_cond_count = z3.Sum(
        [z3.If(c.ref(), 1, 0) for c in filter.all_conditions() if c.op == Operator.EQ]
    )
    ineq_cond_count = z3.Sum(
        [z3.If(c.ref(), 1, 0) for c in filter.all_conditions() if c.op == Operator.INEQ]
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
