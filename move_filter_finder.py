import argparse
from itertools import chain

import z3

import move_symmetries
from puzzle import MoveSeq
from tools import print_stamped


def at_most_one(ls: list[z3.BoolRef]):
    return z3.PbLe([(v, 1) for v in ls], 1)


class VariableSeq:
    def __init__(
        self,
        name: str,
        domain: list[int],
        k: int,
        solver: z3.Optimize,
    ):
        assert len(domain) >= 2
        self.name = name
        self.domain = domain
        self.k = k

        # Variables indicating whether each possible condition is enabled or not.
        self.const_eqs = [
            [z3.Bool(f"{name}(s + {s}) == {v}") for v in domain] for s in range(k)
        ]
        self.const_ineqs = [
            [z3.Bool(f"{name}(s + {s}) != {v}") for v in domain] for s in range(k)
        ]
        self.symb_eqs = [
            [z3.Bool(f"{name}(s + {s}) == {name}(s + {f})") for f in range(s + 1, k)]
            for s in range(k)
        ]
        self.symb_ineqs = [
            [z3.Bool(f"{name}(s + {s}) != {name}(s + {f})") for f in range(s + 1, k)]
            for s in range(k)
        ]
        self.symb_lts = [
            [z3.Bool(f"{name}(s + {s}) > {name}(s + {f})") for f in range(s + 1, k)]
            for s in range(k)
        ]
        self.symb_sts = [
            [z3.Bool(f"{name}(s + {s}) < {name}(s + {f})") for f in range(s + 1, k)]
            for s in range(k)
        ]
        self.symb_ltes = [
            [z3.Bool(f"{name}(s + {s}) ≥ {name}(s + {f})") for f in range(s + 1, k)]
            for s in range(k)
        ]
        self.symb_stes = [
            [z3.Bool(f"{name}(s + {s}) ≤ {name}(s + {f})") for f in range(s + 1, k)]
            for s in range(k)
        ]

        # Add AMO restrictions, which might help the solver.
        for s in range(k):
            solver.add(at_most_one(self.const_eqs[s] + self.const_ineqs[s]))
            for f in range(k - s - 1):
                solver.add(at_most_one([self.symb_eqs[s][f], self.symb_ineqs[s][f]]))
                solver.add(at_most_one([self.symb_eqs[s][f], self.symb_lts[s][f]]))
                solver.add(at_most_one([self.symb_eqs[s][f], self.symb_sts[s][f]]))
                solver.add(at_most_one([self.symb_sts[s][f], self.symb_ltes[s][f]]))
                solver.add(at_most_one([self.symb_lts[s][f], self.symb_stes[s][f]]))

        # Disable comparators incompatible with or
        # superfluous for booleans when domain is binary.
        if len(domain) == 2:
            for s in range(k):
                solver.add(z3.And([z3.Not(v) for v in self.const_ineqs[s]]))
                solver.add(z3.And([z3.Not(v) for v in self.symb_lts[s]]))
                solver.add(z3.And([z3.Not(v) for v in self.symb_sts[s]]))
                solver.add(z3.And([z3.Not(v) for v in self.symb_ltes[s]]))
                solver.add(z3.And([z3.Not(v) for v in self.symb_stes[s]]))

        # Prepare condition refs of values being allowed by the const conditions.
        self.allowed_by_const = [
            [
                z3.And(
                    z3.Not(
                        self.const_ineqs[s][i]
                    ),  # to be allowed, should not be inequal
                    z3.And(
                        [z3.Not(eq) for j, eq in enumerate(self.const_eqs[s]) if j != i]
                    ),  # to be allowed, equality to other values should be false
                )
                for i in range(len(self.domain))
            ]
            for s in range(self.k)
        ]

    def allows(self, s: int, value: int, next_values: list[int]):
        assert len(next_values) == (self.k - s - 1)
        di = self.domain.index(value)
        conds = []

        # Require being allowed by the constant equalities.
        conds.append(self.allowed_by_const[s][di])

        # Require symbolic comparisons to hold when enabled.
        for f in range(len(next_values)):
            conds.append(z3.Implies(self.symb_eqs[s][f], value == next_values[f]))
            conds.append(z3.Implies(self.symb_ineqs[s][f], value != next_values[f]))
            conds.append(z3.Implies(self.symb_lts[s][f], value > next_values[f]))
            conds.append(z3.Implies(self.symb_sts[s][f], value < next_values[f]))
            conds.append(z3.Implies(self.symb_ltes[s][f], value >= next_values[f]))
            conds.append(z3.Implies(self.symb_stes[s][f], value <= next_values[f]))

        return z3.And(conds)

    def allowed_count(self):
        def allowed_by_const_product(excluded_steps: list[int]):
            return z3.Product(
                [
                    z3.Sum(
                        [
                            z3.If(self.allowed_by_const[s][i], 1, 0)
                            for i in range(len(self.domain))
                        ]
                    )
                    for s in range(self.k)
                    if s not in excluded_steps
                ]
            )

        return (
            allowed_by_const_product([])
            - z3.Sum(
                [
                    self.symb_eqs[s][f - s - 1]
                    * (
                        z3.Sum(
                            [
                                z3.If(
                                    z3.And(
                                        self.allowed_by_const[s][i],
                                        self.allowed_by_const[f][j],
                                    ),
                                    1,
                                    0,
                                )
                                for i in range(len(self.domain))
                                for j in range(len(self.domain))
                                if i != j
                            ]
                        )
                        * allowed_by_const_product([s, f])
                    )
                    for s in range(self.k)
                    for f in range(s + 1, self.k)
                ]
            )
            - z3.Sum(
                [
                    self.symb_ineqs[s][f - s - 1]
                    * (
                        z3.Sum(
                            [
                                z3.If(
                                    z3.And(
                                        self.allowed_by_const[s][i],
                                        self.allowed_by_const[f][i],
                                    ),
                                    1,
                                    0,
                                )
                                for i in range(len(self.domain))
                            ]
                        )
                        * allowed_by_const_product([s, f])
                    )
                    for s in range(self.k)
                    for f in range(s + 1, self.k)
                ]
            )
            - z3.Sum(
                [
                    self.symb_lts[s][f - s - 1]
                    * (
                        z3.Sum(
                            [
                                z3.If(
                                    z3.And(
                                        self.allowed_by_const[s][i],
                                        self.allowed_by_const[f][j],
                                    ),
                                    1,
                                    0,
                                )
                                for i in range(len(self.domain))
                                for j in range(len(self.domain))
                                if i <= j
                            ]
                        )
                        * allowed_by_const_product([s, f])
                    )
                    for s in range(self.k)
                    for f in range(s + 1, self.k)
                ]
            )
            - z3.Sum(
                [
                    self.symb_sts[s][f - s - 1]
                    * (
                        z3.Sum(
                            [
                                z3.If(
                                    z3.And(
                                        self.allowed_by_const[s][i],
                                        self.allowed_by_const[f][j],
                                    ),
                                    1,
                                    0,
                                )
                                for i in range(len(self.domain))
                                for j in range(len(self.domain))
                                if i >= j
                            ]
                        )
                        * allowed_by_const_product([s, f])
                    )
                    for s in range(self.k)
                    for f in range(s + 1, self.k)
                ]
            )
            - z3.Sum(
                [
                    self.symb_ltes[s][f - s - 1]
                    * (
                        z3.Sum(
                            [
                                z3.If(
                                    z3.And(
                                        self.allowed_by_const[s][i],
                                        self.allowed_by_const[f][j],
                                    ),
                                    1,
                                    0,
                                )
                                for i in range(len(self.domain))
                                for j in range(len(self.domain))
                                if i < j
                            ]
                        )
                        * allowed_by_const_product([s, f])
                    )
                    for s in range(self.k)
                    for f in range(s + 1, self.k)
                ]
            )
            - z3.Sum(
                [
                    self.symb_stes[s][f - s - 1]
                    * (
                        z3.Sum(
                            [
                                z3.If(
                                    z3.And(
                                        self.allowed_by_const[s][i],
                                        self.allowed_by_const[f][j],
                                    ),
                                    1,
                                    0,
                                )
                                for i in range(len(self.domain))
                                for j in range(len(self.domain))
                                if i > j
                            ]
                        )
                        * allowed_by_const_product([s, f])
                    )
                    for s in range(self.k)
                    for f in range(s + 1, self.k)
                ]
            )
        )

    def conditions_from_model(self, model: z3.ModelRef):
        result: list[z3.BoolRef] = []
        for s in range(self.k):
            for v in chain(
                self.const_eqs[s],
                self.const_ineqs[s],
                self.symb_eqs[s],
                self.symb_ineqs[s],
                self.symb_lts[s],
                self.symb_sts[s],
                self.symb_ltes[s],
                self.symb_stes[s],
            ):
                if z3.is_true(model.get_interp(v)):
                    result.append(v)
        return result


def find(n: int, d: int):
    print_stamped("building model...")

    solver = z3.Optimize()
    axs = VariableSeq("ax", [0, 1, 2], d, solver)
    his = VariableSeq("hi", [0, 1], d, solver)
    drs = VariableSeq("dr", [0, 1, 2], d, solver)

    def is_filtered(seq: MoveSeq):
        """Return the restrictions of a move sequence being filtered."""
        assert len(seq) == d
        seq_axs = [m[0] for m in seq]
        seq_his = [1 if m[1] else 0 for m in seq]
        seq_drs = [m[2] for m in seq]
        return z3.And(
            [
                z3.And(
                    axs.allows(s, seq_axs[s], seq_axs[s + 1 :]),
                    his.allows(s, seq_his[s], seq_his[s + 1 :]),
                    drs.allows(s, seq_drs[s], seq_drs[s + 1 :]),
                )
                for s in range(d)
            ]
        )

    # Cumulate all filterable move sequences.
    filterable: list[MoveSeq] = []
    for seq, syms in move_symmetries.load_symmetries(n, d, False).items():
        filterable.extend(syms)

        # If the original move sequence is also of length d,
        # it too can be filtered out, as long as one stays unfiltered.
        if len(seq) == d:
            filterable.append(seq)
            solver.add(z3.Or([z3.Not(is_filtered(s)) for s in syms + [seq]]))

    if len(filterable) == 0:
        raise Exception("there are no move sequences to filter")

    # Add the main objective of maximizing the number of filtered sequences.
    filtered_count = z3.Sum([z3.If(is_filtered(f), 1, 0) for f in filterable])
    solver.maximize(filtered_count)

    # Make sure the filter does not filter too much. The filter should match exactly
    # with the number of newly filtered move sequences plus the number of previously
    # filtered move sequences. This ensures that no unique moves sequences are filtered.
    # TODO: add filtered from lower depths somehow
    refiltered_count = z3.Sum(
        [
            z3.If(is_filtered(f), 1, 0)
            for f in move_symmetries.load_filtered_padded(n, d)
        ]
    )
    solver.add(
        z3.Product(
            axs.allowed_count(),
            his.allowed_count(),
            drs.allowed_count(),
        )
        == filtered_count + refiltered_count
    )

    # As secondary objectives, add minimizing the number of conditions.
    symb_ste_count = (
        (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.symb_stes[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.symb_stes[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.symb_stes[s]]))
    )
    symb_lte_count = (
        (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.symb_ltes[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.symb_ltes[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.symb_ltes[s]]))
    )
    symb_st_count = (
        (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.symb_sts[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.symb_sts[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.symb_sts[s]]))
    )
    symb_lt_count = (
        (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.symb_lts[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.symb_lts[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.symb_lts[s]]))
    )
    symb_ineq_count = (
        (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.symb_ineqs[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.symb_ineqs[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.symb_ineqs[s]]))
    )
    symb_eq_count = (
        (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.symb_eqs[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.symb_eqs[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.symb_eqs[s]]))
    )
    const_ineq_count = (
        (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.const_ineqs[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.const_ineqs[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.const_ineqs[s]]))
    )
    const_eq_count = (
        (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.const_eqs[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.const_eqs[s]]))
        + (z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.const_eqs[s]]))
    )
    condition_count = (
        const_eq_count
        + const_ineq_count
        + symb_eq_count
        + symb_ineq_count
        + symb_lt_count
        + symb_st_count
        + symb_lte_count
        + symb_ste_count
    )
    solver.minimize(condition_count)
    solver.minimize(symb_ste_count)
    solver.minimize(symb_lte_count)
    solver.minimize(symb_st_count)
    solver.minimize(symb_lt_count)
    solver.minimize(symb_ineq_count)
    solver.minimize(symb_eq_count)
    solver.minimize(const_ineq_count)
    solver.minimize(const_eq_count)

    print_stamped("solving...")
    solver.check()

    m = solver.model()
    print_stamped(f"found filter for {m.evaluate(filtered_count)}...")
    print(axs.conditions_from_model(m))
    print(his.conditions_from_model(m))
    print(drs.conditions_from_model(m))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("d", type=int)
    args = parser.parse_args()
    find(args.n, args.d)
