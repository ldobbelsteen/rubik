import argparse
from itertools import chain

import z3

import move_symmetries
from puzzle import MoveSeq
from tools import print_stamped


class VarConditions:
    def __init__(self, name: str, domain: list[int], s: int, d: int):
        self.name = name
        self.domain = domain
        self.s = s
        self.d = d

        # Declare Z3 variables for each variant.
        self.const_eq = [z3.Bool(f"{name}(s + {s}) == {v}") for v in domain]
        self.const_ineq = [z3.Bool(f"{name}(s + {s}) != {v}") for v in domain]
        self.symb_eq = [
            z3.Bool(f"{name}(s + {s}) == {name}(s + {f})") for f in range(s + 1, d)
        ]
        self.symb_ineq = [
            z3.Bool(f"{name}(s + {s}) != {name}(s + {f})") for f in range(s + 1, d)
        ]
        self.symb_lt = [
            z3.Bool(f"{name}(s + {s}) > {name}(s + {f})") for f in range(s + 1, d)
        ]
        self.symb_st = [
            z3.Bool(f"{name}(s + {s}) < {name}(s + {f})") for f in range(s + 1, d)
        ]
        self.symb_lte = [
            z3.Bool(f"{name}(s + {s}) ≥ {name}(s + {f})") for f in range(s + 1, d)
        ]
        self.symb_ste = [
            z3.Bool(f"{name}(s + {s}) ≤ {name}(s + {f})") for f in range(s + 1, d)
        ]

    def facilitates(
        self,
        v: int,
        next_values: list[int],
        next_conditions: list["VarConditions"],
    ):
        assert len(next_values) == (self.d - self.s - 1) == len(next_conditions)
        idx = self.domain.index(v)
        conflicting = []

        # Disallow equality to different constants.
        for i, const_eq in enumerate(self.const_eq):
            if i != idx:
                conflicting.append(const_eq)

        # Disallow inequality to the value.
        conflicting.append(self.const_ineq[idx])

        # Disallow symbolic equality to a next condition that does not
        # facilitate or that has a different value.
        for s in range(len(self.symb_eq)):
            conflicting.append(
                z3.And(
                    self.symb_eq[s],
                    z3.Or(
                        v != next_values[s],
                        z3.Not(
                            next_conditions[s].facilitates(
                                v,
                                next_values[s + 1 :],
                                next_conditions[s + 1 :],
                            )
                        ),
                    ),
                )
            )

        # Disallow symbolic inequality to a next condition that facilitates
        # or that has the same value.
        for s in range(len(self.symb_ineq)):
            conflicting.append(
                z3.And(
                    self.symb_ineq[s],
                    z3.Or(
                        v == next_values[s],
                        next_conditions[s].facilitates(
                            v,
                            next_values[s + 1 :],
                            next_conditions[s + 1 :],
                        ),
                    ),
                )
            )

        # Disallow larger than comparison with a next condition that facilitates
        # no larger values or has smaller than or equal value.
        for s in range(len(self.symb_lt)):
            conflicting.append(
                z3.And(
                    self.symb_lt[s],
                    z3.Or(
                        v <= next_values[s],
                        z3.Not(
                            z3.Or(
                                [
                                    next_conditions[s].facilitates(
                                        nv,
                                        next_values[s + 1 :],
                                        next_conditions[s + 1 :],
                                    )
                                    for nv in self.domain[idx + 1 :]
                                ]
                            )
                        ),
                    ),
                )
            )

        # Disallow smaller than comparison with a next condition that facilitates
        # no smaller values or has larger than or equal value.
        for s in range(len(self.symb_st)):
            conflicting.append(
                z3.And(
                    self.symb_st[s],
                    z3.Or(
                        v >= next_values[s],
                        z3.Not(
                            z3.Or(
                                [
                                    next_conditions[s].facilitates(
                                        nv,
                                        next_values[s + 1 :],
                                        next_conditions[s + 1 :],
                                    )
                                    for nv in self.domain[:idx]
                                ]
                            )
                        ),
                    ),
                )
            )

        # Disallow larger than or equal comparison with a next condition that
        # facilitates no larger than or equal values or has smaller value.
        for s in range(len(self.symb_lte)):
            conflicting.append(
                z3.And(
                    self.symb_lte[s],
                    z3.Or(
                        v < next_values[s],
                        z3.Not(
                            z3.Or(
                                [
                                    next_conditions[s].facilitates(
                                        nv,
                                        next_values[s + 1 :],
                                        next_conditions[s + 1 :],
                                    )
                                    for nv in self.domain[idx:]
                                ]
                            )
                        ),
                    ),
                )
            )

        # Disallow smaller than or equal comparison with a next condition that
        # facilitates no smaller than or equal values or has larger value.
        for s in range(len(self.symb_ste)):
            conflicting.append(
                z3.And(
                    self.symb_ste[s],
                    z3.Or(
                        v > next_values[s],
                        z3.Not(
                            z3.Or(
                                [
                                    next_conditions[s].facilitates(
                                        nv,
                                        next_values[s + 1 :],
                                        next_conditions[s + 1 :],
                                    )
                                    for nv in self.domain[: idx + 1]
                                ]
                            )
                        ),
                    ),
                )
            )

        return z3.Not(z3.Or(conflicting))

    def extract_from_model(self, model: z3.ModelRef) -> list[z3.BoolRef]:
        result = []
        for v in self.const_eq:
            if z3.is_true(model.get_interp(v)):
                result.append(v)
        for v in self.const_ineq:
            if z3.is_true(model.get_interp(v)):
                result.append(v)
        for v in self.symb_eq:
            if z3.is_true(model.get_interp(v)):
                result.append(v)
        for v in self.symb_ineq:
            if z3.is_true(model.get_interp(v)):
                result.append(v)
        for v in self.symb_lt:
            if z3.is_true(model.get_interp(v)):
                result.append(v)
        for v in self.symb_st:
            if z3.is_true(model.get_interp(v)):
                result.append(v)
        for v in self.symb_lte:
            if z3.is_true(model.get_interp(v)):
                result.append(v)
        for v in self.symb_ste:
            if z3.is_true(model.get_interp(v)):
                result.append(v)
        return result


def find(n: int, d: int):
    print_stamped("building model...")

    solver = z3.Optimize()
    ax_conditions = [VarConditions("ax", [0, 1, 2], s, d) for s in range(d)]
    hi_conditions = [VarConditions("hi", [0, 1], s, d) for s in range(d)]
    dr_conditions = [VarConditions("dr", [0, 1, 2], s, d) for s in range(d)]

    def is_filtered(seq: MoveSeq):
        """Return the restrictions of a move sequence
        being filtered by the conditions."""
        axs = [m[0] for m in seq]
        his = [1 if m[1] else 0 for m in seq]
        drs = [m[2] for m in seq]
        assert len(seq) == d
        return z3.And(
            [
                z3.And(
                    ax_conditions[s].facilitates(
                        axs[s], axs[s + 1 :], ax_conditions[s + 1 :]
                    ),
                    hi_conditions[s].facilitates(
                        his[s], his[s + 1 :], hi_conditions[s + 1 :]
                    ),
                    dr_conditions[s].facilitates(
                        drs[s], drs[s + 1 :], dr_conditions[s + 1 :]
                    ),
                )
                for s in range(d)
            ]
        )

    # Build the objective of filtering out as many symmetrical moves as possible.
    filterable_cumulative = []
    for seq, syms in move_symmetries.load_unfiltered(n, d, False).items():
        filterable = [is_filtered(s) for s in syms]

        # If the original move sequence is also of length d,
        # it too can be filtered out, as long as one stays unfiltered.
        if len(seq) == d:
            filterable.append(is_filtered(seq))
            solver.add(z3.Or([z3.Not(f) for f in filterable]))

        # Add filtered to sum objective.
        filterable_cumulative.extend(filterable)

    # Prevent the unique moves from being filtered out.
    unique = move_symmetries.load_unique(n, d, False)
    for i, seq in enumerate(unique):
        solver.add(z3.Not(is_filtered(seq)))
        if i != 0 and i % int(len(unique) / 100) == 0:
            print_stamped(f"{int(100 * (i / len(unique)))}%")

    # Add objectives to the solver.
    if len(filterable_cumulative) == 0:
        raise Exception("there are no unfiltered move sequences")
    filtered_count = z3.Sum([z3.If(f, 1, 0) for f in filterable_cumulative])
    condition_count = z3.Sum(
        [
            z3.If(v, 1, 0)
            for c in chain(ax_conditions, hi_conditions, dr_conditions)
            for v in c.const_eq
            + c.const_ineq
            + c.symb_eq
            + c.symb_ineq
            + c.symb_lt
            + c.symb_st
            + c.symb_lte
            + c.symb_ste
        ]
    )
    symb_lte_count = z3.Sum(
        [
            z3.If(v, 1, 0)
            for c in chain(ax_conditions, hi_conditions, dr_conditions)
            for v in c.symb_lte
        ]
    )
    symb_ste_count = z3.Sum(
        [
            z3.If(v, 1, 0)
            for c in chain(ax_conditions, hi_conditions, dr_conditions)
            for v in c.symb_ste
        ]
    )
    symb_lt_count = z3.Sum(
        [
            z3.If(v, 1, 0)
            for c in chain(ax_conditions, hi_conditions, dr_conditions)
            for v in c.symb_lt
        ]
    )
    symb_st_count = z3.Sum(
        [
            z3.If(v, 1, 0)
            for c in chain(ax_conditions, hi_conditions, dr_conditions)
            for v in c.symb_st
        ]
    )
    symb_ineq_count = z3.Sum(
        [
            z3.If(v, 1, 0)
            for c in chain(ax_conditions, hi_conditions, dr_conditions)
            for v in c.symb_ineq
        ]
    )
    symb_eq_count = z3.Sum(
        [
            z3.If(v, 1, 0)
            for c in chain(ax_conditions, hi_conditions, dr_conditions)
            for v in c.symb_eq
        ]
    )
    const_ineq_count = z3.Sum(
        [
            z3.If(v, 1, 0)
            for c in chain(ax_conditions, hi_conditions, dr_conditions)
            for v in c.const_ineq
        ]
    )
    const_eq_count = z3.Sum(
        [
            z3.If(v, 1, 0)
            for c in chain(ax_conditions, hi_conditions, dr_conditions)
            for v in c.const_eq
        ]
    )
    solver.maximize(filtered_count)
    solver.minimize(condition_count)
    solver.minimize(symb_lte_count)
    solver.minimize(symb_ste_count)
    solver.minimize(symb_lt_count)
    solver.minimize(symb_st_count)
    solver.minimize(symb_ineq_count)
    solver.minimize(symb_eq_count)
    solver.minimize(const_ineq_count)
    solver.minimize(const_eq_count)

    print_stamped("solving...")
    solver.check()

    m = solver.model()
    print_stamped(f"found filter for {m.evaluate(filtered_count)}...")
    for s in range(d):
        axi = ax_conditions[s].extract_from_model(m)
        if axi is not None:
            print(axi)
        hii = hi_conditions[s].extract_from_model(m)
        if hii is not None:
            print(hii)
        dri = dr_conditions[s].extract_from_model(m)
        if dri is not None:
            print(dri)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("d", type=int)
    args = parser.parse_args()
    find(args.n, args.d)
