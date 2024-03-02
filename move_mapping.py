import os
import sys
import ast
import inspect
import typing
import itertools
from datetime import datetime
from functools import reduce
from dd.autoref import BDD, Function
from misc import print_stamped, create_parent_directory
from logic import (
    coord_mapping,
    corner_rotation_mapping,
    edge_rotation_mapping,
    cubie_type,
)

MappingTreeOutput = tuple[int | str, ...]
MappingTreeComparison = tuple[str, bool, str | int]
MappingTree = dict[MappingTreeComparison | None, "MappingTree"] | MappingTreeOutput


def mapping_to_tree(
    n: int, variables: set[str], mapping: typing.Callable
) -> MappingTree:
    """Convert a mapping function to a tree by parsing the function's AST."""

    def convert_n_minus_1_subtraction(v: ast.BinOp) -> str:
        assert isinstance(v.op, ast.Sub)
        assert isinstance(v.right, ast.Name)
        assert isinstance(v.left, ast.BinOp)
        assert isinstance(v.left.left, ast.Name)
        assert v.left.left.id == "n"
        assert isinstance(v.left.right, ast.Constant)
        assert v.left.right.value == 1
        return f"{n-1}-{v.right.id}"

    def convert_expression(v: ast.stmt | ast.expr) -> int | str:
        if isinstance(v, ast.Name):  # variable name
            if v.id == "n":  # n is known, so return that instead
                return n
            assert v.id in variables
            return v.id
        elif isinstance(v, ast.Constant):  # constant integer
            return v.value
        elif isinstance(v, ast.BinOp):
            return convert_n_minus_1_subtraction(v)
        else:
            raise Exception(f"unsupported expression: {v}")

    def convert_return(r: ast.Return) -> MappingTreeOutput:
        assert isinstance(r.value, ast.Tuple)
        return tuple([convert_expression(v) for v in r.value.elts])

    def convert_comparison(c: ast.Compare) -> tuple[str, bool, int | str]:
        assert isinstance(c.left, ast.Name)  # left side is variable
        assert len(c.ops) == 1  # only one operator
        assert len(c.comparators) == 1  # compared to a single value

        left = c.left.id
        operator = c.ops[0]
        right = convert_expression(c.comparators[0])

        if isinstance(operator, ast.Eq):  # equals sign
            return (left, True, right)
        elif isinstance(operator, ast.NotEq):  # not equals sign
            return (left, False, right)
        else:
            raise Exception(f"unsupported comparison operator: {operator}")

    def recurse_nodes(nodes: list[ast.stmt], default_return: MappingTreeOutput | None):
        # Single return statement, so we always return.
        if len(nodes) == 1 and isinstance(nodes[0], ast.Return):
            return convert_return(nodes[0])

        # Extract default return if present. Else use the inherited return.
        if len(nodes) == 2:
            return_node = nodes.pop()
            assert isinstance(return_node, ast.Return)
            default_return = convert_return(return_node)
        assert len(nodes) == 1

        subtree: MappingTree = {}

        def recurse_branches(b: ast.If):
            assert isinstance(b.test, ast.Compare)
            comparison = convert_comparison(b.test)
            subtree[comparison] = recurse_nodes(b.body, default_return)
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

    function_def = ast.parse(inspect.getsource(mapping)).body[0]
    assert isinstance(function_def, ast.FunctionDef)
    return recurse_nodes(function_def.body, None)


def extract_mapping_tree_paths(
    tree: MappingTree,
) -> dict[frozenset[MappingTreeComparison], MappingTreeOutput]:
    """Return all paths of comparisons that lead to leaves in a mapping tree."""
    result: dict[frozenset[MappingTreeComparison], MappingTreeOutput] = {}

    def recurse(
        subtree: MappingTree,
        current: frozenset[MappingTreeComparison],
    ):
        if isinstance(subtree, tuple):
            result[current] = subtree
        else:
            for comparison, subsubtree in subtree.items():
                if comparison is not None:
                    recurse(
                        subsubtree,
                        current | {comparison},
                    )

            default_return = subtree[None]
            assert isinstance(default_return, tuple)

            # If none of the comparisons hold, the default return should hold.
            negated_comparisons = frozenset(
                (eq[0], not eq[1], eq[2]) for eq in subtree.keys() if eq is not None
            )
            result[current | negated_comparisons] = default_return

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


def comparison_condition(
    left: str, equals: bool, right: str | int, bdd: BDD, domains: dict[str, set[int]]
) -> Function:
    if isinstance(right, int):
        if equals:
            return bdd.var(encode(left, right))
        else:
            return ~bdd.var(encode(left, right))
    else:
        if equals:
            return reduce(
                lambda x, y: x | y,
                [
                    bdd.var(encode(left, overlap)) & bdd.var(encode(right, overlap))
                    for overlap in domains[left].intersection(domains[right])
                ],
            )
        else:
            return reduce(
                lambda x, y: x & y,
                [
                    bdd.var(encode(left, overlap)) != bdd.var(encode(right, overlap))
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


def compute_move_mapping(
    n: int,
    mapping: typing.Callable,
    input_names: tuple[str, ...],
    input_domain: set[tuple[int, ...]],
    output_names: tuple[str, ...],
    output_domain: set[tuple[int, ...]],
) -> list[tuple[tuple[int | None, ...], tuple[int, ...]]]:
    tree = mapping_to_tree(n, set(pn for pn in input_names), mapping)

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
                comparison_condition(input_name, True, input[i], bdd, domains)
                for i, input_name in enumerate(input_names)
            ],
        )

    def output_equals(output: MappingTreeOutput):
        """Return condition on the output being equal to a specific output."""
        return reduce(
            lambda x, y: x & y,
            [
                comparison_condition(output_name, True, output[i], bdd, domains)
                for i, output_name in enumerate(output_names)
            ],
        )

    # Disallow inputs not from the input domain.
    input_domains = [d for n, d in domains.items() if n in input_names]
    for input in itertools.product(*input_domains):
        if input not in input_domain:
            root = root & ~input_equals(input)

    # Add the paths as restrictions.
    for comparisons, output in extract_mapping_tree_paths(tree).items():
        path_cond = bdd.true
        for left, equals, right in comparisons:
            path_cond = path_cond & comparison_condition(
                left, equals, right, bdd, domains
            )
        root = root & (~path_cond | output_equals(output))

    result: list[tuple[tuple[int | None, ...], tuple[int, ...]]] = []

    # Extract results from the BDD.
    for subset in minimal_substitution_subsets(domains, bdd, root):
        vals = {k: v for k, v in subset}
        input = tuple([vals[n] if n in vals else None for n in input_names])
        output = tuple([vals[n] for n in output_names])
        result.append((input, output))

    return result


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
            coord_mapping,
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
            corner_rotation_mapping,
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
            coord_mapping,
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
            edge_rotation_mapping,
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
            coord_mapping,
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
        mapping,
        input_names,
        input_domain,
        output_names,
        output_domain,
    ) in move_mappings(n).items():
        if len(input_domain) == 0:
            continue  # can happen for n <= 2, where there are no edges or centers

        path = file_path(n, name)
        if not overwrite and os.path.isfile(path):
            continue
        create_parent_directory(path)

        print_stamped(f"generating move mapping '{name}' for n = {n}...")
        mappings = compute_move_mapping(
            n, mapping, input_names, input_domain, output_names, output_domain
        )

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
                inputs = [None if c == "*" else int(c) for c in input_raw]
                outputs = [int(c) for c in output_raw]

                input_dict = {
                    input_names[i]: v for i, v in enumerate(inputs) if v is not None
                }
                output_dict = {output_names[i]: v for i, v in enumerate(outputs)}
                mappings.append((input_dict, output_dict))

        result[name] = mappings


# e.g. python move_mapping.py {n}
if __name__ == "__main__":
    start = datetime.now()
    generate(int(sys.argv[1]), True)
    print(f"took {datetime.now()-start} to complete!")
