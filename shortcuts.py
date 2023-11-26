import os
import sys
from datetime import datetime
from functools import reduce
from dd.autoref import BDD, Function
from misc import timestamped


def new_color_coordinates(
    n: int, ma: int, mi: int, md: int, f: int, y: int, x: int
) -> tuple[int, int, int]:
    """Get coordinates in previous state where new color of cell can be found.
    The front, right, back and left faces face upwards. The bottom and top faces
    both face upwards when rotating them towards you."""
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


def new_color_coordinates_tree(n: int):
    """Tree representation of new_color_coordinates function."""
    return {
        ("ma", 0): {
            ("f_in", 4): {
                ("mi", 0): {
                    ("md", 0): ("f_in", f"{n-1}-x_in", "y_in"),
                    ("md", 1): ("f_in", "x_in", f"{n-1}-y_in"),
                    ("md", 2): ("f_in", f"{n-1}-y_in", f"{n-1}-x_in"),
                }
            },
            ("f_in", 5): {
                ("mi", n - 1): {
                    ("md", 0): ("f_in", "x_in", f"{n-1}-y_in"),
                    ("md", 1): ("f_in", f"{n-1}-x_in", "y_in"),
                    ("md", 2): ("f_in", f"{n-1}-y_in", f"{n-1}-x_in"),
                }
            },
            ("f_in", 0): {
                ("mi", "y_in"): {
                    ("md", 0): ("f_in+1", "y_in", "x_in"),
                    ("md", 1): ("f_in+3", "y_in", "x_in"),
                    ("md", 2): ("f_in+2", "y_in", "x_in"),
                }
            },
            ("f_in", 1): {
                ("mi", "y_in"): {
                    ("md", 0): ("f_in+1", "y_in", "x_in"),
                    ("md", 1): ("f_in-1", "y_in", "x_in"),
                    ("md", 2): ("f_in+2", "y_in", "x_in"),
                }
            },
            ("f_in", 2): {
                ("mi", "y_in"): {
                    ("md", 0): ("f_in+1", "y_in", "x_in"),
                    ("md", 1): ("f_in-1", "y_in", "x_in"),
                    ("md", 2): ("f_in-2", "y_in", "x_in"),
                }
            },
            ("f_in", 3): {
                ("mi", "y_in"): {
                    ("md", 0): ("f_in-3", "y_in", "x_in"),
                    ("md", 1): ("f_in-1", "y_in", "x_in"),
                    ("md", 2): ("f_in-2", "y_in", "x_in"),
                }
            },
        },
        ("ma", 1): {
            ("f_in", 3): {
                ("mi", 0): {
                    ("md", 0): ("f_in", "x_in", f"{n-1}-y_in"),
                    ("md", 1): ("f_in", f"{n-1}-x_in", "y_in"),
                    ("md", 2): ("f_in", f"{n-1}-y_in", f"{n-1}-x_in"),
                }
            },
            ("f_in", 1): {
                ("mi", n - 1): {
                    ("md", 0): ("f_in", f"{n-1}-x_in", "y_in"),
                    ("md", 1): ("f_in", "x_in", f"{n-1}-y_in"),
                    ("md", 2): ("f_in", f"{n-1}-y_in", f"{n-1}-x_in"),
                }
            },
            ("f_in", 0): {
                ("mi", "x_in"): {
                    ("md", 0): ("f_in+5", "y_in", "x_in"),
                    ("md", 1): ("f_in+4", "y_in", "x_in"),
                    ("md", 2): ("f_in+2", f"{n-1}-y_in", f"{n-1}-x_in"),
                }
            },
            ("f_in", 5): {
                ("mi", "x_in"): {
                    ("md", 0): ("f_in-3", f"{n-1}-y_in", f"{n-1}-x_in"),
                    ("md", 1): ("f_in-5", "y_in", "x_in"),
                    ("md", 2): ("f_in-1", "y_in", "x_in"),
                }
            },
            ("f_in", 2): {
                ("mi", f"{n-1}-x_in"): {
                    ("md", 0): ("f_in+2", f"{n-1}-y_in", f"{n-1}-x_in"),
                    ("md", 1): ("f_in+3", f"{n-1}-y_in", f"{n-1}-x_in"),
                    ("md", 2): ("f_in-2", f"{n-1}-y_in", f"{n-1}-x_in"),
                }
            },
            ("f_in", 4): {
                ("mi", "x_in"): {
                    ("md", 0): ("f_in-4", "y_in", "x_in"),
                    ("md", 1): ("f_in-2", f"{n-1}-y_in", f"{n-1}-x_in"),
                    ("md", 2): ("f_in+1", "y_in", "x_in"),
                }
            },
        },
        ("ma", 2): {
            ("f_in", 0): {
                ("mi", 0): {
                    ("md", 0): ("f_in", f"{n-1}-x_in", "y_in"),
                    ("md", 1): ("f_in", "x_in", f"{n-1}-y_in"),
                    ("md", 2): ("f_in", f"{n-1}-y_in", f"{n-1}-x_in"),
                }
            },
            ("f_in", 2): {
                ("mi", n - 1): {
                    ("md", 0): ("f_in", "x_in", f"{n-1}-y_in"),
                    ("md", 1): ("f_in", f"{n-1}-x_in", "y_in"),
                    ("md", 2): ("f_in", f"{n-1}-y_in", f"{n-1}-x_in"),
                }
            },
            ("f_in", 1): {
                ("mi", "x_in"): {
                    ("md", 0): ("f_in+3", f"{n-1}-x_in", "y_in"),
                    ("md", 1): ("f_in+4", "x_in", f"{n-1}-y_in"),
                    ("md", 2): ("f_in+2", f"{n-1}-y_in", f"{n-1}-x_in"),
                }
            },
            ("f_in", 4): {
                ("mi", f"{n-1}-y_in"): {
                    ("md", 0): ("f_in-1", f"{n-1}-x_in", "y_in"),
                    ("md", 1): ("f_in-3", "x_in", f"{n-1}-y_in"),
                    ("md", 2): ("f_in+1", f"{n-1}-y_in", f"{n-1}-x_in"),
                }
            },
            ("f_in", 3): {
                ("mi", f"{n-1}-x_in"): {
                    ("md", 0): ("f_in+2", f"{n-1}-x_in", "y_in"),
                    ("md", 1): ("f_in+1", "x_in", f"{n-1}-y_in"),
                    ("md", 2): ("f_in-2", f"{n-1}-y_in", f"{n-1}-x_in"),
                }
            },
            ("f_in", 5): {
                ("mi", "y_in"): {
                    ("md", 0): ("f_in-4", f"{n-1}-x_in", "y_in"),
                    ("md", 1): ("f_in-2", "x_in", f"{n-1}-y_in"),
                    ("md", 2): ("f_in-1", f"{n-1}-y_in", f"{n-1}-x_in"),
                }
            },
        },
    }


def new_color_coordinates_tree_paths(
    n: int,
) -> set[tuple[set[tuple[str | int, str | int]], tuple[str, str, str]]]:
    """Return all paths of conditions to the leaves in the new_color_coordinates tree."""
    results = set()

    def traverse(subtree, current_conditions: frozenset[tuple[str | int, str | int]]):
        if isinstance(subtree, tuple):
            assert len(subtree) == 3
            results.add((current_conditions, subtree))
        else:
            assert isinstance(subtree, dict)
            for condition in list(subtree.keys()):
                traverse(subtree[condition], current_conditions | {condition})

    traverse(new_color_coordinates_tree(n), frozenset())
    return results


def encode(var_name: str, var_value: int):
    return f"{var_name}_{var_value}"


def equals(
    left: str | int,
    right: str | int,
    bdd: BDD,
    domains: dict[str, set[int]],
) -> Function:
    """Return condition of two values being equal. If a value is a string, it is
    assumed to be the name of a variable in the BDD."""
    left_is_int = isinstance(left, int)
    right_is_int = isinstance(right, int)
    if left_is_int and right_is_int:
        return bdd.true if left == right else bdd.false
    elif left_is_int and not right_is_int:
        return bdd.var(encode(right, left))
    elif not left_is_int and right_is_int:
        return bdd.var(encode(left, right))
    else:
        assert not left_is_int and not right_is_int
        return reduce(
            lambda x, y: x | y,
            [
                bdd.var(encode(left, overlap)) & bdd.var(encode(right, overlap))
                for overlap in domains[left].intersection(domains[right])
            ],
        )


def add_composite_variable(
    name: str, bdd: BDD, root: Function, domains: dict[str, set[int]]
) -> Function:
    """Add variable to the BDD consisting of two values that are added or subtracted. At most
    one of the values can be a string, which is interpreted as a BDD variable name."""
    assert name not in domains

    domain: set[int] = set()
    equivalences: set[tuple[Function, int]] = set()

    is_subtraction = "-" in name
    parts = name.split("-" if is_subtraction else "+")
    assert len(parts) == 2
    left, right = parts[0], parts[1]
    assert left.isnumeric() != right.isnumeric()
    if left.isnumeric():
        for right_value in domains[right]:
            if is_subtraction:
                value = int(left) - right_value
            else:
                value = int(left) + right_value
            domain.add(value)
            equivalences.add((bdd.var(encode(right, right_value)), value))
    if right.isnumeric():
        for left_value in domains[left]:
            if is_subtraction:
                value = left_value - int(right)
            else:
                value = left_value + int(right)
            domain.add(value)
            equivalences.add((bdd.var(encode(left, left_value)), value))

    for value in domain:
        bdd.declare(encode(name, value))
    domains[name] = domain

    for origin, value in equivalences:
        root = root & origin.equiv(bdd.var(encode(name, value)))

    return root


def add_variable(
    name: str,
    domain: set[int],
    bdd: BDD,
    root: Function,
    domains: dict[str, set[int]],
) -> Function:
    """Add variable to the BDD. The variable can take exactly one the values of the domain."""
    assert name not in domains

    variants = []
    for value in domain:
        bdd.declare(encode(name, value))
        variants.append(bdd.var(encode(name, value)))
    domains[name] = domain

    # At least one variable value is true.
    root = root & reduce(lambda x, y: x | y, variants)

    # At most one variable value is true.
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            root = root & (~(variants[i] & variants[j]))

    return root


def minimal_substitution_subsets(
    substitutions: dict[str, set[int]],
    bdd: BDD,
    root: Function,
):
    """Extract all minimal subsets of the substitutions such that regardless of which
    values the substitutions not in the subset take on, there is always a truth assignment."""
    subsets: set[frozenset[tuple[str, int]]] = set()

    def all_substitutions_truth(remaining: list[str], substituted: Function):
        if len(remaining) > 0:
            name = remaining[0]
            for value in substitutions[name]:
                if not all_substitutions_truth(
                    remaining[1:], bdd.let({encode(name, value): bdd.true}, substituted)
                ):
                    return False
            return True
        else:
            return substituted.count() > 0

    def subset_search(
        remaining: list[str],
        chosen: frozenset[tuple[str, int]],
        skipped: frozenset[str],
        substituted: Function,
    ):
        if len(remaining) > 0:
            name = remaining[0]
            subset_search(remaining[1:], chosen, skipped | {name}, substituted)
            for value in substitutions[name]:
                subset_search(
                    remaining[1:],
                    chosen | {(name, value)},
                    skipped,
                    bdd.let({encode(name, value): bdd.true}, substituted),
                )
        else:
            if all_substitutions_truth(list(skipped), substituted):
                if all(
                    [
                        not existing_subset.issubset(chosen)
                        for existing_subset in subsets
                    ]
                ):
                    existing_subsets_trash = []
                    for existing_subset in subsets:
                        if chosen.issubset(existing_subset):
                            existing_subsets_trash.append(existing_subset)
                    for existing_subset in existing_subsets_trash:
                        subsets.remove(existing_subset)
                    subsets.add(chosen)

    subset_search(list(substitutions.keys()), frozenset(), frozenset(), root)
    return subsets


def compute_shortcuts(n: int):
    bdd = BDD()
    root = bdd.true
    domains: dict[str, set[int]] = {}

    move_vars = {"ma": set(range(3)), "mi": set(range(n)), "md": set(range(3))}

    input_vars = {
        "f_in": set(range(6)),
        "y_in": set(range(n)),
        "x_in": set(range(n)),
    }

    output_vars = {
        "f_out": set(range(6)),
        "y_out": set(range(n)),
        "x_out": set(range(n)),
    }

    # Add variables.
    for name, domain in move_vars.items():
        root = add_variable(name, domain, bdd, root, domains)
    for name, domain in input_vars.items():
        root = add_variable(name, domain, bdd, root, domains)
    for name, domain in output_vars.items():
        root = add_variable(name, domain, bdd, root, domains)

    # Add all composite variables encountered in the tree.
    for conditions, output in new_color_coordinates_tree_paths(n):
        for left, right in conditions:
            if not isinstance(left, int) and left not in domains:
                root = add_composite_variable(left, bdd, root, domains)
            if not isinstance(right, int) and right not in domains:
                root = add_composite_variable(right, bdd, root, domains)
        if output[0] not in domains:
            root = add_composite_variable(output[0], bdd, root, domains)
        if output[1] not in domains:
            root = add_composite_variable(output[1], bdd, root, domains)
        if output[2] not in domains:
            root = add_composite_variable(output[2], bdd, root, domains)

    def output_equals(output: tuple[str | int, str | int, str | int]):
        """Return condition on the output being equal to a tuple."""
        return (
            equals("f_out", output[0], bdd, domains)
            & equals("y_out", output[1], bdd, domains)
            & equals("x_out", output[2], bdd, domains)
        )

    # Add the paths as restrictions.
    one_path_holds = bdd.false
    for conditions, output in new_color_coordinates_tree_paths(n):
        one_condition_false = bdd.false
        for left, right in conditions:
            one_condition_false = one_condition_false | ~equals(
                left, right, bdd, domains
            )
        root = root & (one_condition_false | output_equals(output))
        one_path_holds = one_path_holds | ~one_condition_false

    # If none of the paths hold, the default output of (f, y, x) holds.
    root = root & (one_path_holds | output_equals(("f_in", "y_in", "x_in")))

    # Extract shortcuts from the diagram.
    shortcuts: list[
        tuple[
            tuple[int | None, int | None, int | None],
            tuple[int, int, int],
            tuple[int, int, int],
        ]
    ] = []
    for subset in minimal_substitution_subsets(
        move_vars | input_vars | output_vars, bdd, root
    ):
        vals = {k: v for k, v in subset}
        shortcuts.append(
            (
                (vals.get("ma"), vals.get("mi"), vals.get("md")),
                (vals["f_in"], vals["y_in"], vals["x_in"]),
                (vals["f_out"], vals["y_out"], vals["x_out"]),
            )
        )

    shortcuts.sort(key=lambda sc: str(sc))  # Sort to make result deterministic.
    file = open(f"./shortcuts/dim{n}", "w")
    file.write(str(shortcuts))
    file.close()


def compute_missing_shortcuts(n_values: set[int]):
    for n in n_values:
        if not os.path.isfile(f"./shortcuts/dim{n}"):
            print(timestamped(f"missing shortcuts for dimension {n}, computing..."))
            compute_shortcuts(n)


def load_shortcuts_amalgamated(n: int):
    file = open(f"shortcuts/dim{n}", "r")
    shortcuts: list[
        tuple[
            tuple[int | None, int | None, int | None],
            tuple[int, int, int],
            tuple[int, int, int],
        ]
    ] = eval(file.read())
    file.close()
    return {
        ma: {
            mi: {
                md: [
                    (shortcut[1], shortcut[2])
                    for shortcut in shortcuts
                    if shortcut[0][0] == ma
                    and shortcut[0][1] == mi
                    and shortcut[0][2] == md
                ]
                for md in list(range(3)) + [None]
            }
            for mi in list(range(n)) + [None]
        }
        for ma in list(range(3)) + [None]
    }


# python shortcuts.py {n}
if __name__ == "__main__":
    start = datetime.now()
    compute_shortcuts(int(sys.argv[1]))
    print(f"took {datetime.now()-start} to complete!")
