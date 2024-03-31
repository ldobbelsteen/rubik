import argparse
import re
from enum import Enum
from itertools import chain

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

        def conflicting_const(left: int, op: Operator, right: int) -> bool:
            match op:
                case Operator.EQ:
                    return left != right
                case Operator.INEQ:
                    return left == right
                case Operator.LT:
                    return left <= right
                case Operator.ST:
                    return left >= right
                case Operator.LTE:
                    return left < right
                case Operator.STE:
                    return left > right

        # Prepare expressions which determine if values are facilitated
        # purely by the const conditions.
        self.const_facilitates = [
            [
                z3.And(
                    [
                        z3.Not(cond.ref())
                        for cond in self.conditions[s]
                        if isinstance(cond.right, int)
                        and conflicting_const(v, cond.op, cond.right)
                    ]
                )
                for v in domain
            ]
            for s in range(self.k)
        ]

    def facilitates(self, s: int, vs: list[int]):
        conds = []

        # Require being facilitated by the constant equalities.
        conds.append(self.const_facilitates[s][self.domain.index(vs[s])])

        # Require symbolic comparisons to hold when enabled.
        for cond in self.conditions[s]:
            if not isinstance(cond.right, int):
                match cond.op:
                    case Operator.EQ:
                        conds.append(z3.Implies(cond.ref(), vs[s] == vs[cond.right.s]))
                    case Operator.INEQ:
                        conds.append(z3.Implies(cond.ref(), vs[s] != vs[cond.right.s]))
                    case Operator.LT:
                        conds.append(z3.Implies(cond.ref(), vs[s] > vs[cond.right.s]))
                    case Operator.ST:
                        conds.append(z3.Implies(cond.ref(), vs[s] < vs[cond.right.s]))
                    case Operator.LTE:
                        conds.append(z3.Implies(cond.ref(), vs[s] >= vs[cond.right.s]))
                    case Operator.STE:
                        conds.append(z3.Implies(cond.ref(), vs[s] <= vs[cond.right.s]))

        return z3.And(conds)

    def facilitates_count(self):
        # TODO: make correct
        return z3.Product(
            [
                z3.Sum(
                    [
                        z3.If(self.const_facilitates[s][di], 1, 0)
                        for di in range(len(self.domain))
                    ]
                )
                for s in range(self.k)
            ]
        )

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

    def facilitates(self, seq: MoveSeq):
        assert len(seq) == self.k
        return z3.And(
            [
                z3.And(
                    self.ax.facilitates(s, [m[0] for m in seq]),
                    self.hi.facilitates(s, [1 if m[1] else 0 for m in seq]),
                    self.dr.facilitates(s, [m[2] for m in seq]),
                )
                for s in range(self.k)
            ]
        )

    def facilitates_count(self):
        return z3.Product(
            [
                self.ax.facilitates_count(),
                self.hi.facilitates_count(),
                self.dr.facilitates_count(),
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
            solver.add(z3.Or([z3.Not(filter.facilitates(s)) for s in syms + [seq]]))

    if len(filterable) == 0:
        raise Exception("there are no move sequences to filter")

    # Make sure the filter does not filter too much. The filter should match exactly
    # with the number of newly filtered move sequences plus the number of previously
    # filtered move sequences that are filtered again. This ensures that no unique
    # moves sequences are filtered.
    filtered_count = z3.Sum([z3.If(filter.facilitates(f), 1, 0) for f in filterable])
    print_stamped("ingesting previously filtered...")
    refiltered_count = z3.Sum(
        [
            z3.If(filter.facilitates(f), 1, 0)
            for f in move_symmetries.load_filtered(n, k, False)
        ]
    )
    print_stamped("finished ingesting previously filtered...")
    solver.add(filter.facilitates_count() == filtered_count + refiltered_count)

    # Add the main objective of maximizing the number of filtered sequences.
    solver.maximize(filtered_count)

    # As secondary objectives, add minimizing the number of conditions.
    condition_count = z3.Sum(
        [z3.If(cond.ref(), 1, 0) for cond in filter.all_conditions()]
    )
    solver.minimize(condition_count)

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
