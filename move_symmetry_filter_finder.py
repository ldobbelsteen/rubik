import argparse

import z3

import move_symmetries
from puzzle import MoveSeq
from tools import print_stamped


class VariableSeq:
    def __init__(
        self,
        name: str,
        domain: list[int],
        k: int,
    ):
        self.name = name
        self.domain = domain
        self.k = k

        # Variables indicating whether each possible condition is enabled or not.
        self.const_eqs = [
            [z3.Bool(f"{name}(s + {s}) == {v}") for v in domain] for s in range(k)
        ]
        self.const_ineqs = (
            [[z3.Bool(f"{name}(s + {s}) != {v}") for v in domain] for s in range(k)]
            if len(domain) > 2
            else None
        )
        self.symb_eqs = [
            [z3.Bool(f"{name}(s + {s}) == {name}(s + {f})") for f in range(s + 1, k)]
            for s in range(k)
        ]
        self.symb_ineqs = [
            [z3.Bool(f"{name}(s + {s}) != {name}(s + {f})") for f in range(s + 1, k)]
            for s in range(k)
        ]
        self.symb_lts = (
            [
                [z3.Bool(f"{name}(s + {s}) > {name}(s + {f})") for f in range(s + 1, k)]
                for s in range(k)
            ]
            if len(domain) > 2
            else None
        )
        self.symb_sts = (
            [
                [z3.Bool(f"{name}(s + {s}) < {name}(s + {f})") for f in range(s + 1, k)]
                for s in range(k)
            ]
            if len(domain) > 2
            else None
        )
        self.symb_ltes = (
            [
                [z3.Bool(f"{name}(s + {s}) ≥ {name}(s + {f})") for f in range(s + 1, k)]
                for s in range(k)
            ]
            if len(domain) > 2
            else None
        )
        self.symb_stes = (
            [
                [z3.Bool(f"{name}(s + {s}) ≤ {name}(s + {f})") for f in range(s + 1, k)]
                for s in range(k)
            ]
            if len(domain) > 2
            else None
        )

    def allows(self, s: int, value: int, subsequent_values: list[int]):
        assert len(subsequent_values) == (self.k - s - 1)
        di = self.domain.index(value)
        constraints = []

        # Disallow equality to different constants.
        for i, const_eq in enumerate(self.const_eqs[s]):
            if i != di:
                constraints.append(z3.Not(const_eq))

        # Disallow inequality to the value.
        if self.const_ineqs is not None:
            if len(self.const_ineqs[s]) > 0:
                constraints.append(z3.Not(self.const_ineqs[s][di]))

        for f in range(len(self.symb_eqs[s])):
            constraints.append(
                z3.Implies(self.symb_eqs[s][f], value == subsequent_values[f])
            )

        for f in range(len(self.symb_ineqs[s])):
            constraints.append(
                z3.Implies(self.symb_ineqs[s][f], value != subsequent_values[f])
            )

        if self.symb_lts is not None:
            for f in range(len(self.symb_lts[s])):
                constraints.append(
                    z3.Implies(self.symb_lts[s][f], value > subsequent_values[f])
                )

        if self.symb_sts is not None:
            for f in range(len(self.symb_sts[s])):
                constraints.append(
                    z3.Implies(self.symb_sts[s][f], value < subsequent_values[f])
                )

        if self.symb_ltes is not None:
            for f in range(len(self.symb_ltes[s])):
                constraints.append(
                    z3.Implies(self.symb_ltes[s][f], value >= subsequent_values[f])
                )

        if self.symb_stes is not None:
            for f in range(len(self.symb_stes[s])):
                constraints.append(
                    z3.Implies(self.symb_stes[s][f], value <= subsequent_values[f])
                )

        return z3.And(constraints)

    def allowed_count(self):
        return z3.Int("TODO")

    def conditions_from_model(self, model: z3.ModelRef):
        result: list[z3.BoolRef] = []
        for s in range(self.k):
            for v in self.const_eqs[s]:
                if z3.is_true(model.get_interp(v)):
                    result.append(v)
            if self.const_ineqs is not None:
                for v in self.const_ineqs[s]:
                    if z3.is_true(model.get_interp(v)):
                        result.append(v)
            for v in self.symb_eqs[s]:
                if z3.is_true(model.get_interp(v)):
                    result.append(v)
            for v in self.symb_ineqs[s]:
                if z3.is_true(model.get_interp(v)):
                    result.append(v)
            if self.symb_lts is not None:
                for v in self.symb_lts[s]:
                    if z3.is_true(model.get_interp(v)):
                        result.append(v)
            if self.symb_sts is not None:
                for v in self.symb_sts[s]:
                    if z3.is_true(model.get_interp(v)):
                        result.append(v)
            if self.symb_ltes is not None:
                for v in self.symb_ltes[s]:
                    if z3.is_true(model.get_interp(v)):
                        result.append(v)
            if self.symb_stes is not None:
                for v in self.symb_stes[s]:
                    if z3.is_true(model.get_interp(v)):
                        result.append(v)
        return result


def find(n: int, d: int):
    print_stamped("building model...")

    solver = z3.Optimize()
    axs = VariableSeq("ax", [0, 1, 2], d)
    his = VariableSeq("hi", [0, 1], d)
    drs = VariableSeq("dr", [0, 1, 2], d)

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

    # Build the objective of filtering out as many symmetrical moves as possible.
    to_filter_cumulative = []
    for seq, syms in move_symmetries.load_symmetries(n, d, False).items():
        to_filter = [is_filtered(s) for s in syms]

        # If the original move sequence is also of length d,
        # it too can be filtered out, as long as one stays unfiltered.
        if len(seq) == d:
            to_filter.append(is_filtered(seq))
            solver.add(z3.Or([z3.Not(f) for f in to_filter]))

        # Add filtered to sum objective.
        to_filter_cumulative.extend(to_filter)

    # Add objectives to the solver.
    if len(to_filter_cumulative) == 0:
        raise Exception("there are no move sequences to filter")
    filtered_count = z3.Sum([z3.If(f, 1, 0) for f in to_filter_cumulative])
    symb_lte_count = (
        (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.symb_ltes[s]])
            if axs.symb_ltes is not None
            else 0
        )
        + (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.symb_ltes[s]])
            if his.symb_ltes is not None
            else 0
        )
        + (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.symb_ltes[s]])
            if drs.symb_ltes is not None
            else 0
        )
    )
    symb_ste_count = (
        (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.symb_stes[s]])
            if axs.symb_stes is not None
            else 0
        )
        + (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.symb_stes[s]])
            if his.symb_stes is not None
            else 0
        )
        + (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.symb_stes[s]])
            if drs.symb_stes is not None
            else 0
        )
    )
    symb_lt_count = (
        (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.symb_lts[s]])
            if axs.symb_lts is not None
            else 0
        )
        + (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.symb_lts[s]])
            if his.symb_lts is not None
            else 0
        )
        + (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.symb_lts[s]])
            if drs.symb_lts is not None
            else 0
        )
    )
    symb_st_count = (
        (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.symb_sts[s]])
            if axs.symb_sts is not None
            else 0
        )
        + (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.symb_sts[s]])
            if his.symb_sts is not None
            else 0
        )
        + (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.symb_sts[s]])
            if drs.symb_sts is not None
            else 0
        )
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
        (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in axs.const_ineqs[s]])
            if axs.const_ineqs is not None
            else 0
        )
        + (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in his.const_ineqs[s]])
            if his.const_ineqs is not None
            else 0
        )
        + (
            z3.Sum([z3.If(v, 1, 0) for s in range(d) for v in drs.const_ineqs[s]])
            if drs.const_ineqs is not None
            else 0
        )
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
        + symb_st_count
        + symb_lt_count
        + symb_ste_count
        + symb_lte_count
    )
    # TODO: add already filtered sequences that are filtered on the left side of the eq
    # TODO: update move symmetries script to not cut branches st. we have all filtered
    solver.add(
        filtered_count
        == axs.allowed_count() * his.allowed_count() * drs.allowed_count()
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
    print(axs.conditions_from_model(m))
    print(his.conditions_from_model(m))
    print(drs.conditions_from_model(m))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("d", type=int)
    args = parser.parse_args()
    find(args.n, args.d)
