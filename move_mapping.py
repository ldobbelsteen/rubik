import os
import sys
import ast
import inspect
import typing
import itertools
from datetime import datetime
from functools import reduce
from dd.autoref import BDD, Function
from misc import (
    print_stamped,
    create_parent_directory,
    corner_move_coord_mapping,
    corner_move_rotation_mapping,
    edge_move_coord_mapping,
    edge_move_rotation_mapping,
    center_move_coord_mapping,
    cubie_type,
)

MappingTreeOutput = tuple[int | str, ...]
MappingTreeEquality = tuple[str, str | int]
MappingTree = dict[MappingTreeEquality | None, "MappingTree"] | MappingTreeOutput


def mapping_to_tree(n: int, params: set[str], function: typing.Callable) -> MappingTree:
    """Convert a mapping function to a tree by parsing the function's AST."""

    def convert_value(v: ast.stmt | ast.expr) -> int | str:
        if isinstance(v, ast.Name):  # variable name
            if v.id == "n":  # n is known, so return that instead
                return n
            assert v.id in params  # variable should be an input
            return v.id
        elif isinstance(v, ast.Constant):  # constant integer
            return v.value
        elif isinstance(v, ast.BinOp):  # combination of two values with operator
            left = convert_value(v.left)
            right = convert_value(v.right)
            assert isinstance(left, int)
            assert isinstance(right, int)
            if isinstance(v.op, ast.Add):  # addition
                return left + right
            elif isinstance(v.op, ast.Sub):  # subtraction
                return left - right
            else:
                raise Exception(f"unsupported operator: {v.op}")
        else:
            raise Exception(f"unsupported value: {v}")

    def convert_return_to_output(r: ast.Return) -> MappingTreeOutput:
        assert isinstance(r.value, ast.Tuple)
        return tuple([convert_value(v) for v in r.value.elts])

    def convert_compare_to_equality(c: ast.Compare) -> tuple[str, int | str]:
        assert isinstance(c.left, ast.Name)  # left side is variable
        assert len(c.ops) == 1  # only one operator
        assert isinstance(c.ops[0], ast.Eq)  # operator is equals sign
        assert len(c.comparators) == 1  # left side is compared to a single value
        return (c.left.id, convert_value(c.comparators[0]))

    def recurse_nodes(nodes: list[ast.stmt], default_return: MappingTreeOutput | None):
        # Single return statement, so we always return.
        if len(nodes) == 1 and isinstance(nodes[0], ast.Return):
            return convert_return_to_output(nodes[0])

        # Extract default return if present. Else use the inherited return.
        if len(nodes) == 2:
            raw = nodes.pop()
            assert isinstance(raw, ast.Return)
            default_return = convert_return_to_output(raw)
        assert len(nodes) == 1

        subtree: MappingTree = {}

        def recurse_branches(b: ast.If):
            assert isinstance(b.test, ast.Compare)
            equality = convert_compare_to_equality(b.test)
            subtree[equality] = recurse_nodes(b.body, default_return)
            if len(b.orelse) > 0:
                assert len(b.orelse) == 1
                branch = b.orelse[0]
                assert isinstance(branch, ast.If)
                recurse_branches(branch)

        # Extract all branches recursively.
        branches = nodes.pop()
        assert isinstance(branches, ast.If)
        recurse_branches(branches)

        # Add default return as None condition.
        assert default_return is not None
        subtree[None] = default_return

        return subtree

    function_def = ast.parse(inspect.getsource(function)).body[0]
    assert isinstance(function_def, ast.FunctionDef)
    return recurse_nodes(function_def.body, None)


def extract_mapping_tree_paths(
    tree: MappingTree,
) -> dict[
    tuple[frozenset[MappingTreeEquality], frozenset[MappingTreeEquality]],
    MappingTreeOutput,
]:
    """Return all paths of equalities and inequalities that lead to leaves in
    a mapping tree."""

    result: dict[
        tuple[frozenset[MappingTreeEquality], frozenset[MappingTreeEquality]],
        MappingTreeOutput,
    ] = {}

    def recurse(
        subtree: MappingTree,
        equalities: frozenset[MappingTreeEquality],
    ):
        if isinstance(subtree, tuple):
            result[(equalities, frozenset())] = subtree
        else:
            for equality, subsubtree in subtree.items():
                if equality is not None:
                    recurse(
                        subsubtree,
                        equalities | {equality},
                    )

            default_return = subtree[None]
            assert isinstance(default_return, tuple)
            inequalities = frozenset(eq for eq in subtree.keys() if eq is not None)
            result[(equalities, inequalities)] = default_return

    recurse(tree, frozenset())
    return result


def encode(var_name: str, var_value: int):
    """Encode a variable with its value in a string."""
    return f"{var_name}_{var_value}"


def add_var(
    name: str,
    domain: set[int],
    bdd: BDD,
    root: Function,
) -> Function:
    """Add variable to the BDD. The variable is restricted to take exactly one
    the values of the domain."""

    variants = []
    for value in domain:
        encoding = encode(name, value)
        bdd.declare(encoding)
        variants.append(bdd.var(encoding))

    # At least one variable value is true.
    root = root & reduce(lambda x, y: x | y, variants)

    # At most one variable value is true.
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            root = root & (~(variants[i] & variants[j]))

    return root


def equality_condition(
    left: str, right: str | int, bdd: BDD, domains: dict[str, set[int]]
) -> Function:
    if isinstance(right, int):
        return bdd.var(encode(left, right))
    else:
        return reduce(
            lambda x, y: x | y,
            [
                bdd.var(encode(left, overlap)) & bdd.var(encode(right, overlap))
                for overlap in domains[left].intersection(domains[right])
            ],
        )


def minimal_substitution_subsets(
    domains: dict[str, set[int]],
    bdd: BDD,
    root: Function,
):
    """Extract all minimal-size substitutions for variables such that, regardless
    of which values the unsubstituted variables take, there is always a truth assignment."""
    subsets: set[frozenset[tuple[str, int]]] = set()

    def all_substitutions_truth(remaining: list[str], substituted: Function):
        if len(remaining) > 0:
            name = remaining[0]
            for value in domains[name]:
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
            for value in domains[name]:
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

    subset_search(list(domains.keys()), frozenset(), frozenset(), root)
    return subsets


def move_mappings(
    n: int,
) -> dict[
    str,
    tuple[
        typing.Callable,
        tuple[str, ...],  # input names
        set[tuple[int, ...]],  # input domain
        tuple[str, ...],  # output names
        set[tuple[int, ...]],  # output domain
    ],
]:
    corners: set[tuple[int, int, int]] = set()
    centers: set[tuple[int, int, int]] = set()
    edges: set[tuple[int, int, int]] = set()

    for x in range(n):
        for y in range(n):
            for z in range(n):
                match cubie_type(n, x, y, z):
                    case 0:
                        corners.add((x, y, z))
                    case 1:
                        centers.add((x, y, z))
                    case 2:
                        edges.add((x, y, z))

    return {
        "corner_coord": (
            corner_move_coord_mapping,
            ("x", "y", "z", "ma", "mi", "md"),
            set(
                corner + (ma, mi, md)
                for corner in corners
                for ma in range(3)
                for mi in range(n)
                for md in range(3)
            ),
            ("x_new", "y_new", "z_new"),
            corners,
        ),
        "corner_rotation": (
            corner_move_rotation_mapping,
            ("x", "y", "z", "r", "ma", "mi", "md"),
            set(
                corner + (r, ma, mi, md)
                for corner in corners
                for r in range(3)
                for ma in range(3)
                for mi in range(n)
                for md in range(3)
            ),
            ("r_new",),
            set([tuple([v]) for v in range(3)]),
        ),
        "edge_coord": (
            edge_move_coord_mapping,
            ("x", "y", "z", "ma", "mi", "md"),
            set(
                edge + (ma, mi, md)
                for edge in edges
                for ma in range(3)
                for mi in range(n)
                for md in range(3)
            ),
            ("x_new", "y_new", "z_new"),
            edges,
        ),
        "edge_rotation": (
            edge_move_rotation_mapping,
            ("x", "y", "z", "r", "ma", "mi", "md"),
            set(
                edge + (r, ma, mi, md)
                for edge in edges
                for r in range(3)
                for ma in range(3)
                for mi in range(n)
                for md in range(3)
            ),
            ("r_new",),
            set([tuple([v]) for v in range(3)]),
        ),
        "center_coord": (
            center_move_coord_mapping,
            ("x", "y", "z", "ma", "mi", "md"),
            set(
                center + (ma, mi, md)
                for center in centers
                for ma in range(3)
                for mi in range(n)
                for md in range(3)
            ),
            ("x_new", "y_new", "z_new"),
            centers,
        ),
    }


def file_path(n: int, name: str):
    return f"./move_mappings/n{n}-{name}.txt"


def generate(n: int, overwrite=False):
    for name, (
        function,
        input_names,
        input_domain,
        output_names,
        output_domain,
    ) in move_mappings(n).items():
        path = file_path(n, name)
        if not overwrite and os.path.isfile(path):
            continue
        create_parent_directory(path)

        if len(input_domain) == 0:
            continue  # can happen for n <= 2, where there are no edges nor centers

        print_stamped(f"generating move mappings '{name}' for n = {n}...")

        # Compute the mapping tree for this function.
        tree = mapping_to_tree(n, set(pn for pn in input_names), function)

        # Initialize separate domains for all inputs and outputs.
        domains: dict[str, set[int]] = {}
        for name in input_names:
            domains[name] = set()
        for name in output_names:
            domains[name] = set()

        # Compute separate domains for all variables.
        for i, name in enumerate(input_names):
            for input in input_domain:
                domains[name].add(input[i])
        for i, name in enumerate(output_names):
            for output in output_domain:
                domains[name].add(output[i])

        # Initialize a BDD with a root.
        bdd = BDD()
        root = bdd.true

        # Add all variables to the BDD.
        for name, domain in domains.items():
            root = add_var(name, domain, bdd, root)

        def input_equals(input: tuple[int, ...]):
            """Return condition on the input being equal to a specific input."""
            return reduce(
                lambda x, y: x & y,
                [
                    equality_condition(input_name, input[i], bdd, domains)
                    for i, input_name in enumerate(input_names)
                ],
            )

        def output_equals(output: MappingTreeOutput):
            """Return condition on the output being equal to a specific output."""
            return reduce(
                lambda x, y: x & y,
                [
                    equality_condition(output_name, output[i], bdd, domains)
                    for i, output_name in enumerate(output_names)
                ],
            )

        # Disallow inputs not from the input domain.
        input_domains = [d for n, d in domains.items() if n in input_names]
        for input in itertools.product(*input_domains):
            if input not in input_domain:
                root = root & ~input_equals(input)

        # Add the paths as restrictions.
        for (eqs, ineqs), output in extract_mapping_tree_paths(tree).items():
            path_cond = bdd.true
            for left, right in eqs:
                path_cond = path_cond & equality_condition(left, right, bdd, domains)
            for left, right in ineqs:
                path_cond = path_cond & ~equality_condition(left, right, bdd, domains)
            root = root & (~path_cond | output_equals(output))

        mappings: list[tuple[tuple[int | None, ...], tuple[int, ...]]] = []

        # Extract minimal mappings from the BDD.
        for subset in minimal_substitution_subsets(domains, bdd, root):
            vals = {k: v for k, v in subset}
            input = tuple([vals[n] if n in vals else None for n in input_names])
            output = tuple([vals[n] for n in output_names])
            mappings.append((input, output))

        # Sort to make result deterministic.
        mappings.sort(key=lambda sc: str(sc))

        with open(path, "w") as file:
            for input, output in mappings:
                input_str = "".join(["*" if i is None else str(i) for i in input])
                output_str = "".join([str(o) for o in output])
                file.write(f"{input_str} {output_str}\n")


def load(n: int):
    result: dict[str, list[tuple[dict[str, int], dict[str, int]]]] = {}

    for name, (_, input_names, _, output_names, _) in move_mappings(n).items():
        mappings: list[tuple[dict[str, int], dict[str, int]]] = []
        with open(file_path(n, name), "r") as file:
            for line in file:
                input_raw, output_raw = line.split(" ")

                input_dict = {
                    input_names[i]: v
                    for i, v in enumerate(
                        [None if c == "*" else int(c) for c in input_raw]
                    )
                    if v is not None
                }

                output_dict = {
                    output_names[i]: v
                    for i, v in enumerate([int(c) for c in output_raw])
                }

                mappings.append((input_dict, output_dict))

        result[name] = mappings


# e.g. python move_mapping.py {n}
if __name__ == "__main__":
    start = datetime.now()
    generate(int(sys.argv[1]), True)
    print(f"took {datetime.now()-start} to complete!")
