import mdd as _mdd
import mdd.nx as _mddnx
import aiger
import networkx
import sys
import os
from functools import reduce
from datetime import datetime


def tree(n: int) -> dict:
    """Tree representation of new_color function. If no path of conditions applies, the output is ("f", "y", "x")."""
    return {
        ("ma", 0): {
            ("f", 4): {
                ("mi", 0): {
                    ("md", 0): (4, "~x", "y"),
                    ("md", 1): (4, "x", "~y"),
                    ("md", 2): (4, "~y", "~x"),
                }
            },
            ("f", 5): {
                ("mi", n - 1): {
                    ("md", 0): (5, "x", "~y"),
                    ("md", 1): (5, "~x", "y"),
                    ("md", 2): (5, "~y", "~x"),
                }
            },
            ("f", 0): {
                ("mi", "y"): {
                    ("md", 0): (1, "y", "x"),
                    ("md", 1): (3, "y", "x"),
                    ("md", 2): (2, "y", "x"),
                }
            },
            ("f", 1): {
                ("mi", "y"): {
                    ("md", 0): (2, "y", "x"),
                    ("md", 1): (0, "y", "x"),
                    ("md", 2): (3, "y", "x"),
                }
            },
            ("f", 2): {
                ("mi", "y"): {
                    ("md", 0): (3, "y", "x"),
                    ("md", 1): (1, "y", "x"),
                    ("md", 2): (0, "y", "x"),
                }
            },
            ("f", 3): {
                ("mi", "y"): {
                    ("md", 0): (0, "y", "x"),
                    ("md", 1): (2, "y", "x"),
                    ("md", 2): (1, "y", "x"),
                }
            },
        },
        ("ma", 1): {
            ("f", 3): {
                ("mi", 0): {
                    ("md", 0): (3, "x", "~y"),
                    ("md", 1): (3, "~x", "y"),
                    ("md", 2): (3, "~y", "~x"),
                }
            },
            ("f", 1): {
                ("mi", n - 1): {
                    ("md", 0): (1, "~x", "y"),
                    ("md", 1): (1, "x", "~y"),
                    ("md", 2): (1, "~y", "~x"),
                }
            },
            ("f", 0): {
                ("mi", "x"): {
                    ("md", 0): (5, "y", "x"),
                    ("md", 1): (4, "y", "x"),
                    ("md", 2): (2, "~y", "~x"),
                }
            },
            ("f", 5): {
                ("mi", "x"): {
                    ("md", 0): (2, "~y", "~x"),
                    ("md", 1): (0, "y", "x"),
                    ("md", 2): (4, "y", "x"),
                }
            },
            ("f", 2): {
                ("mi", "~x"): {
                    ("md", 0): (4, "~y", "~x"),
                    ("md", 1): (5, "~y", "~x"),
                    ("md", 2): (0, "~y", "~x"),
                }
            },
            ("f", 4): {
                ("mi", "x"): {
                    ("md", 0): (0, "y", "x"),
                    ("md", 1): (2, "~y", "~x"),
                    ("md", 2): (5, "y", "x"),
                }
            },
        },
        ("ma", 2): {
            ("f", 0): {
                ("mi", 0): {
                    ("md", 0): (0, "~x", "y"),
                    ("md", 1): (0, "x", "~y"),
                    ("md", 2): (0, "~y", "~x"),
                }
            },
            ("f", 2): {
                ("mi", n - 1): {
                    ("md", 0): (2, "x", "~y"),
                    ("md", 1): (2, "~x", "y"),
                    ("md", 2): (2, "~y", "~x"),
                }
            },
            ("f", 1): {
                ("mi", "x"): {
                    ("md", 0): (4, "~x", "y"),
                    ("md", 1): (5, "x", "~y"),
                    ("md", 2): (3, "~y", "~x"),
                }
            },
            ("f", 4): {
                ("mi", "~y"): {
                    ("md", 0): (3, "~x", "y"),
                    ("md", 1): (1, "x", "~y"),
                    ("md", 2): (5, "~y", "~x"),
                }
            },
            ("f", 3): {
                ("mi", "~x"): {
                    ("md", 0): (5, "~x", "y"),
                    ("md", 1): (4, "x", "~y"),
                    ("md", 2): (1, "~y", "~x"),
                }
            },
            ("f", 5): {
                ("mi", "y"): {
                    ("md", 0): (1, "~x", "y"),
                    ("md", 1): (3, "x", "~y"),
                    ("md", 2): (4, "~y", "~x"),
                }
            },
        },
    }


def new_color(
    n: int, ma: int, mi: int, md: int, f: int, y: int, x: int
) -> tuple[int, int, int]:
    """Get color of cell based on last state, move and cell location."""
    if ma == 0:
        if f == 4 and mi == 0:
            if md == 0:
                return (4, n - 1 - x, y)
            elif md == 1:
                return (4, x, n - 1 - y)
            elif md == 2:
                return (4, n - 1 - y, n - 1 - x)
        elif f == 5 and mi == n - 1:
            if md == 0:
                return (5, x, n - 1 - y)
            elif md == 1:
                return (5, n - 1 - x, y)
            elif md == 2:
                return (5, n - 1 - y, n - 1 - x)
        elif f == 0 and mi == y:
            if md == 0:
                return (1, y, x)
            elif md == 1:
                return (3, y, x)
            elif md == 2:
                return (2, y, x)
        elif f == 1 and mi == y:
            if md == 0:
                return (2, y, x)
            elif md == 1:
                return (0, y, x)
            elif md == 2:
                return (3, y, x)
        elif f == 2 and mi == y:
            if md == 0:
                return (3, y, x)
            elif md == 1:
                return (1, y, x)
            elif md == 2:
                return (0, y, x)
        elif f == 3 and mi == y:
            if md == 0:
                return (0, y, x)
            elif md == 1:
                return (2, y, x)
            elif md == 2:
                return (1, y, x)
    elif ma == 1:
        if f == 3 and mi == 0:
            if md == 0:
                return (3, x, n - 1 - y)
            elif md == 1:
                return (3, n - 1 - x, y)
            elif md == 2:
                return (3, n - 1 - y, n - 1 - x)
        elif f == 1 and mi == n - 1:
            if md == 0:
                return (1, n - 1 - x, y)
            elif md == 1:
                return (1, x, n - 1 - y)
            elif md == 2:
                return (1, n - 1 - y, n - 1 - x)
        elif f == 0 and mi == x:
            if md == 0:
                return (5, y, x)
            elif md == 1:
                return (4, y, x)
            elif md == 2:
                return (2, n - 1 - y, n - 1 - x)
        elif f == 5 and mi == x:
            if md == 0:
                return (2, n - 1 - y, n - 1 - x)
            elif md == 1:
                return (0, y, x)
            elif md == 2:
                return (4, y, x)
        elif f == 2 and mi == n - 1 - x:
            if md == 0:
                return (4, n - 1 - y, n - 1 - x)
            elif md == 1:
                return (5, n - 1 - y, n - 1 - x)
            elif md == 2:
                return (0, n - 1 - y, n - 1 - x)
        elif f == 4 and mi == x:
            if md == 0:
                return (0, y, x)
            elif md == 1:
                return (2, n - 1 - y, n - 1 - x)
            elif md == 2:
                return (5, y, x)
    elif ma == 2:
        if f == 0 and mi == 0:
            if md == 0:
                return (0, n - 1 - x, y)
            elif md == 1:
                return (0, x, n - 1 - y)
            elif md == 2:
                return (0, n - 1 - y, n - 1 - x)
        elif f == 2 and mi == n - 1:
            if md == 0:
                return (2, x, n - 1 - y)
            elif md == 1:
                return (2, n - 1 - x, y)
            elif md == 2:
                return (2, n - 1 - y, n - 1 - x)
        elif f == 1 and mi == x:
            if md == 0:
                return (4, n - 1 - x, y)
            elif md == 1:
                return (5, x, n - 1 - y)
            elif md == 2:
                return (3, n - 1 - y, n - 1 - x)
        elif f == 4 and mi == n - 1 - y:
            if md == 0:
                return (3, n - 1 - x, y)
            elif md == 1:
                return (1, x, n - 1 - y)
            elif md == 2:
                return (5, n - 1 - y, n - 1 - x)
        elif f == 3 and mi == n - 1 - x:
            if md == 0:
                return (5, n - 1 - x, y)
            elif md == 1:
                return (4, x, n - 1 - y)
            elif md == 2:
                return (1, n - 1 - y, n - 1 - x)
        elif f == 5 and mi == y:
            if md == 0:
                return (1, n - 1 - x, y)
            elif md == 1:
                return (3, x, n - 1 - y)
            elif md == 2:
                return (4, n - 1 - y, n - 1 - x)
    return (f, y, x)


def tree_to_aiger(
    interface: _mdd.Interface, tree: dict, leaf_index: int, default_output: str
) -> aiger.BoolExpr:
    """Create AIGER circuit based on tree represented by a dictionary.."""

    def symbolic_equality(
        varname1: str, varname2: str, inverted: bool
    ) -> aiger.BoolExpr:
        var1 = (
            interface.var(varname1)
            if varname1 != interface.output.name
            else interface.output
        )
        var2 = (
            interface.var(varname2)
            if varname2 != interface.output.name
            else interface.output
        )
        return reduce(
            lambda x, y: x | y,
            [
                aiger.BoolExpr(var1.expr() == var1.encode(i))
                & aiger.BoolExpr(
                    var2.expr() == var2.encode(i if not inverted else n - 1 - i)
                )
                for i in range(n)  # domain of [0..n-1] is assumed
            ],
        )

    def condition_to_aiger(condition: tuple[str, str | int]) -> aiger.BoolExpr:
        inp, value = condition
        if type(value) is str:
            if value[0] == "~":
                return symbolic_equality(inp, value[1:], True)
            else:
                return symbolic_equality(inp, value, False)
        else:
            return aiger.BoolExpr(
                interface.var(inp).expr() == interface.var(inp).encode(value)
            )

    def conditions_to_aiger(conditions: list[tuple[str, str | int]]) -> aiger.BoolExpr:
        return reduce(lambda x, y: x & y, map(condition_to_aiger, conditions))

    def get_paths(
        subtree, prefix: list[tuple[str, str | int]]
    ) -> list[tuple[list[tuple[str, str | int]], str | int]]:
        if type(subtree) is tuple:  # leaf reached
            return [(prefix, subtree[leaf_index])]
        result = []
        for condition, subsubtree in subtree.items():
            result.extend(get_paths(subsubtree, prefix + [condition]))
        return result

    paths = get_paths(tree, [])

    # If all conditions on a path hold, the output should be equal to the leaf at the end.
    implication_tuples = [
        (
            conditions_to_aiger(conditions),
            interface.output.expr() == interface.output.encode(leaf),
        )
        for conditions, leaf in paths
    ]

    # If none of the paths hold, the output is the specified default leaf.
    implication_tuples.append(
        (
            (~(reduce(lambda x, y: x | y, [cond for cond, _ in implication_tuples]))),
            interface.output.expr() == interface.output.encode(default_output),
        )
    )

    implications = []
    for condition, leaf in implication_tuples:
        implications.append(condition.implies(leaf))
    return reduce(lambda x, y: x & y, implications)


def write_mdd_to_pdf(mdd: _mdd.DecisionDiagram, path: str):
    """Write an MDD to a PDF file for inspection."""
    digraph = _mddnx.to_nx(mdd, symbolic_edges=False)
    pydot = networkx.nx_pydot.to_pydot(digraph)
    pydot.set_size("5,2!")
    pydot.set_ratio("fill")
    pydot.write_pdf(path)


def shortcut_is_subset(sc1: tuple, sc2: tuple):
    """Check if shortcut sc1 is a subset of shortcut sc2"""
    for i in range(len(sc2)):
        if sc2[i] is None:
            if sc1[i] is not None:
                return False
        else:
            if sc1[i] != sc2[i] and sc1[i] is not None:
                return False
    return True


def shortcuts_conflict(sc1: tuple, sc2: tuple):
    """Check if two shortcuts have different values for the same variable."""
    return not all(
        [sc1[i] is None or sc2[i] is None or sc1[i] == sc2[i] for i in range(len(sc1))]
    )


def shortcuts_combined(sc1: tuple, sc2: tuple):
    """Combine two shortcuts into one. It is assumed that they do not conflict."""
    return tuple([sc1[i] if sc1[i] is not None else sc2[i] for i in range(len(sc1))])


def shortcut_output(
    leaf_index: int,
    f: int,
    y: int,
    x: int,
    sc: tuple[int | None, int | None, int | None],
) -> int:
    """Get the output (at leaf_index in the leaves) corresponding to a shortcut."""
    ma = sc[0] if sc[0] is not None else 0
    mi = sc[1] if sc[1] is not None else 0
    md = sc[2] if sc[2] is not None else 0
    return new_color(n, ma, mi, md, f, y, x)[leaf_index]


def shortcut_holds(
    sc: tuple[int | None, int | None, int | None],
    ma: int | None,
    mi: int | None,
    md: int | None,
):
    if sc[0] is None or sc[0] == ma:
        if sc[1] is None or sc[1] == mi:
            if sc[2] is None or sc[2] == md:
                return True
    return False


def compute_shortcuts_per_output(
    tree: dict,
    leaf_index: int,
    output_name: str,
    output_domain: list[str | int],
    default_output: str | int,
):
    """Get all shortcuts of one of the output values (the one at leaf_index in the leaves)."""
    # All possible inputs and their domains.
    input_domains = {
        "f": list(range(6)),
        "y": list(range(n)),
        "x": list(range(n)),
        "ma": list(range(3)),
        "mi": list(range(n)),
        "md": list(range(3)),
    }

    # Parse the tree into an MDD.
    interface = _mdd.Interface(inputs=input_domains, output=output_domain)
    expr = tree_to_aiger(interface, tree, leaf_index, default_output)
    mdd = interface.lift(expr)
    write_mdd_to_pdf(mdd, f"shortcuts/dim{n}/mdd-{output_name}.pdf")

    def has_one_leaf(mdd: _mdd.DecisionDiagram):
        """Check whether an MDD has exactly one leaf."""
        graph = _mddnx.to_nx(mdd)
        encountered = False
        for node in graph.nodes():
            if graph.out_degree(node) == 0:
                if encountered:
                    return False
                encountered = True
        return encountered

    # For every combination of f, y and x, compute subsets of the move representation
    # which result in the MDD only having one leaf node when applied. These subsets are
    # as small as possible (a.k.a. any of its subsets result in zero leaf nodes).
    shortcuts: list[
        list[list[set[tuple[tuple[int | None, int | None, int | None], int]]]]
    ] = [[[set() for _ in range(n)] for _ in range(n)] for _ in range(6)]
    for f in range(6):
        mdd1 = mdd.let({"f": f})
        for y in range(n):
            mdd2 = mdd1.let({"y": y})
            for x in range(n):
                mdd3 = mdd2.let({"x": x})
                for ma in list(range(3)) + [None]:
                    mdd4 = mdd3.let({"ma": ma}) if ma is not None else mdd3
                    for mi in list(range(n)) + [None]:
                        mdd5 = mdd4.let({"mi": mi}) if mi is not None else mdd4
                        for md in list(range(3)) + [None]:
                            mdd6 = mdd5.let({"md": md}) if md is not None else mdd5
                            if has_one_leaf(mdd6):
                                new_shortcut = (
                                    (ma, mi, md),
                                    shortcut_output(leaf_index, f, y, x, (ma, mi, md)),
                                )
                                current_shortcuts = shortcuts[f][y][x]
                                # Check if no existing shortcut is a subset.
                                if all(
                                    [
                                        not shortcut_is_subset(csc[0], new_shortcut[0])
                                        for csc in current_shortcuts
                                    ]
                                ):
                                    # Remove any supersets, which are weaker than this new shortcut.
                                    for csc in set(current_shortcuts):
                                        if shortcut_is_subset(new_shortcut[0], csc[0]):
                                            current_shortcuts.remove(csc)
                                    current_shortcuts.add(new_shortcut)

    return shortcuts


def intersect_shortcuts(
    f: set[tuple[tuple[int | None, int | None, int | None], int]],
    y: set[tuple[tuple[int | None, int | None, int | None], int]],
    x: set[tuple[tuple[int | None, int | None, int | None], int]],
):
    """Create shortcuts for (f, y, x) by intersecting all combinations of their separate shortcuts."""
    result: set[
        tuple[tuple[int | None, int | None, int | None], tuple[int, int, int]]
    ] = set()
    for scf in f:
        for scy in y:
            for scx in x:
                if (
                    not shortcuts_conflict(scf[0], scy[0])
                    and not shortcuts_conflict(scy[0], scx[0])
                    and not shortcuts_conflict(scx[0], scf[0])
                ):
                    shortcut = shortcuts_combined(
                        shortcuts_combined(scf[0], scy[0]), scx[0]
                    )
                    output = (scf[1], scy[1], scx[1])
                    # Check if no existing shortcut is a subset.
                    if all([not shortcut_is_subset(sc[0], shortcut) for sc in result]):
                        # Remove any supersets, which are weaker than this new shortcut.
                        for sc in set(result):
                            if shortcut_is_subset(shortcut, sc[0]):
                                result.remove(sc)
                        result.add((shortcut, output))
    return result


def compute_shortcuts(n: int):
    """Build all shortcuts for specific n."""

    # Get separate f shortcuts.
    f_path = f"shortcuts/dim{n}/shortcuts-f.txt"
    if os.path.exists(f_path):
        print("using cached f shortcuts...")
        ssf = eval(open(f_path, "r").read())
    else:
        print("building f shortcuts...")
        ssf = compute_shortcuts_per_output(tree(n), 0, "f", list(range(6)) + ["f"], "f")
        open(f_path, "w").write(str(ssf))

    # Get separate y shortcuts.
    y_path = f"shortcuts/dim{n}/shortcuts-y.txt"
    if os.path.exists(y_path):
        print("using cached y shortcuts...")
        ssy = eval(open(y_path, "r").read())
    else:
        print("building y shortcuts...")
        ssy = compute_shortcuts_per_output(tree(n), 1, "y", ["x", "~x", "y", "~y"], "y")
        open(y_path, "w").write(str(ssy))

    # Get separate x shortcuts.
    x_path = f"shortcuts/dim{n}/shortcuts-x.txt"
    if os.path.exists(x_path):
        print("using cached x shortcuts...")
        ssx = eval(open(x_path, "r").read())
    else:
        print("building x shortcuts...")
        ssx = compute_shortcuts_per_output(tree(n), 2, "x", ["x", "~x", "y", "~y"], "x")
        open(x_path, "w").write(str(ssx))

    # Combine the three types of shortcuts.
    print("combining shortcuts...")
    return [
        [
            [
                intersect_shortcuts(ssf[f][y][x], ssy[f][y][x], ssx[f][y][x])
                for x in range(n)
            ]
            for y in range(n)
        ]
        for f in range(6)
    ]


def verify_shortcuts(
    shortcuts: list[
        list[
            list[
                set[
                    tuple[
                        tuple[int | None, int | None, int | None], tuple[int, int, int]
                    ]
                ]
            ]
        ]
    ],
):
    print("testing shortcuts...")

    # Test for valid shortcut and output domains.
    for f in range(6):
        for y in range(n):
            for x in range(n):
                for shortcut, output in shortcuts[f][y][x]:
                    assert shortcut[0] in list(range(3)) + [None]
                    assert shortcut[1] in list(range(n)) + [None]
                    assert shortcut[2] in list(range(3)) + [None]
                    assert output[0] in list(range(6))
                    assert output[1] in list(range(n))
                    assert output[2] in list(range(n))

    # Test for unique and correct output for all combinations of inputs.
    for f in range(6):
        for y in range(n):
            for x in range(n):
                s = shortcuts[f][y][x]
                for ma in list(range(3)):
                    for mi in list(range(n)):
                        for md in list(range(3)):
                            outputs = []
                            for shortcut, output in s:
                                if shortcut_holds(shortcut, ma, mi, md):
                                    assert output == new_color(n, ma, mi, md, f, y, x)
                                    outputs.append(output)
                            assert len(outputs) == 1


# python compute_shortcuts.py {n}
if __name__ == "__main__":
    n = int(sys.argv[1])

    result_file = f"shortcuts/dim{n}/shortcuts.txt"
    info_file = result_file + ".info"

    start = datetime.now()
    shortcuts = compute_shortcuts(n)
    verify_shortcuts(shortcuts)

    result_file = open(result_file, "w")
    result_file.write(str(shortcuts))
    result_file.close()

    info = f"took {datetime.now()-start} to complete"
    info_file = open(info_file, "w")
    info_file.write(info)
    info_file.close()
    print(info)
