import os
import sys
import ast
import inspect
import typing
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

MappingTreeCondition = tuple[str, str | int]
MappingTreeOutput = int | str | tuple[int | str, ...]
MappingTree = dict[MappingTreeCondition | None, "MappingTree"] | MappingTreeOutput


def mapping_to_tree(
    n: int, params: set[str], function: typing.Callable
) -> tuple[MappingTreeOutput, MappingTree]:
    """Convert a mapping function to a tree by parsing the function's AST."""

    def convert_value(v: ast.stmt | ast.expr) -> int | str:
        if isinstance(v, ast.Name):  # variable name
            if v.id == "n":  # n is known, so return that instead
                return n
            assert v.id in params  # variable should be a parameter
            return v.id
        elif isinstance(v, ast.Constant):  # constant integer
            return v.value
        elif isinstance(v, ast.BinOp):  # combination of two values with operator
            left = convert_value(v.left)
            right = convert_value(v.right)
            if isinstance(v.op, ast.Add):  # addition
                if isinstance(left, int) and isinstance(right, int):
                    return left + right
                return f"{left}+{right}"
            elif isinstance(v.op, ast.Sub):  # subtraction
                if isinstance(left, int) and isinstance(right, int):
                    return left - right
                return f"{left}-{right}"
            else:
                raise Exception(f"unsupported operator: {v.op}")
        else:
            raise Exception(f"unsupported value: {v}")

    def convert_return_to_output(r: ast.Return) -> MappingTreeOutput:
        assert r.value is not None
        if isinstance(r.value, ast.Tuple):
            return tuple([convert_value(v) for v in r.value.elts])
        return convert_value(r.value)

    def compare_to_condition(c: ast.Compare) -> tuple[str, int | str]:
        assert isinstance(c.left, ast.Name)  # left side is variable
        assert len(c.ops) == 1  # only one operator
        assert isinstance(c.ops[0], ast.Eq)  # operator is equals sign
        assert len(c.comparators) == 1  # left side is compared to a single value
        return (c.left.id, convert_value(c.comparators[0]))

    def recurse_nodes(nodes: list[ast.stmt]):
        if len(nodes) == 1 and isinstance(nodes[0], ast.Return):
            return convert_return_to_output(nodes[0])

        subtree = {}

        # Check for else branch and extract it.
        else_branch = None
        if not isinstance(nodes[-1], ast.If):
            else_branch_node = nodes.pop()
            if isinstance(else_branch_node, ast.Return):
                else_branch = else_branch_node
            else:
                print(f"ERROR: unexpected else branch type: {else_branch_node}")

        # Recurse on all if-branches in this branch.
        for branch in nodes:
            assert isinstance(branch, ast.If)
            assert isinstance(branch.test, ast.Compare)
            compare = compare_to_condition(branch.test)
            subtree[compare] = recurse_nodes(branch.body)

        # Add else branch if it exists.
        if else_branch is not None:
            subtree[None] = convert_return_to_output(else_branch)

        return subtree

    function_def = ast.parse(inspect.getsource(function)).body[0]
    assert isinstance(function_def, ast.FunctionDef)

    default_output_node = function_def.body.pop()
    assert isinstance(default_output_node, ast.Return)
    default_output = convert_return_to_output(default_output_node)

    return default_output, recurse_nodes(function_def.body)


def extract_tree_paths(
    tree: MappingTree,
) -> list[tuple[set[MappingTreeCondition], MappingTreeOutput]]:
    """Return all paths of conditions to the leaves in a mapping tree."""
    result: list[tuple[set[MappingTreeCondition], MappingTreeOutput]] = []

    def recurse(
        current_tree: MappingTree,
        current_conditions: set[MappingTreeCondition],
    ):
        if isinstance(current_tree, dict):
            for condition, subtree in current_tree.items():
                recurse(
                    subtree,
                    current_conditions | {condition}
                    if condition is not None
                    else current_conditions,
                )
        else:
            result.append((current_conditions, current_tree))

    recurse(tree, set())
    return result


def encode(var_name: str, var_value: int):
    """Encode a variable with its value in a string."""
    return f"{var_name}_{var_value}"


def convert_tree_condition(
    cond: MappingTreeCondition, bdd: BDD, domains: dict[str, set[int]]
) -> Function:
    left, right = cond
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


def add_composite_var(
    name: str, bdd: BDD, root: Function, domains: dict[str, set[int]]
) -> Function:
    """Add variable to the BDD consisting of two values with an operator."""
    assert name not in domains
    is_subtraction = "-" in name
    left, right = name.split("-" if is_subtraction else "+")

    domain: set[int] = set()
    equivalences: set[tuple[Function, int]] = set()

    # Exactly one is numeric and one is a variable.
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


def minimal_substitution_subsets(
    substitutions: dict[str, set[int]],
    bdd: BDD,
    root: Function,
):
    """Extract all minimal subsets of the substitutions such that, regardless of
    which values the substitutions not in the subset take on, there is always a
    truth assignment."""
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


def move_mappings(
    n: int,
) -> dict[
    str,
    tuple[
        typing.Callable,
        tuple[str, ...],  # parameter value names
        set[tuple[int, ...]],  # all possible parameter tuples
        str | tuple[str, ...],  # return value name(s)
        set[int] | set[tuple[int, ...]],  # all possible return values/tuples
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
            "r_new",
            set(range(3)),
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
            "r_new",
            set(range(3)),
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


def generate(n: int):
    for name, (
        function,
        input_names,
        input_domain,
        output_names,
        output_domain,
    ) in move_mappings(n).items():
        path = file_path(n, name)
        if os.path.isfile(path):
            return  # already generated, so skip
        create_parent_directory(path)
        print_stamped(f"generating move mappings '{name}' for n = {n}...")

        # Compute the mapping tree for this function.
        default_output, tree = mapping_to_tree(
            n, set(pn for pn in input_names), function
        )

        # Initialize separate domains for all inputs and outputs.
        domains: dict[str, set[int]] = {}
        for name in input_names:
            domains[name] = set()
        if isinstance(output_names, str):
            domains[output_names] = set()
        else:
            for name in output_names:
                domains[name] = set()

        # Compute separate domains for all variables.
        for input in input_domain:
            for i, name in enumerate(input_names):
                domains[name].add(input[i])
        for output in output_domain:
            if isinstance(output, tuple):
                for i, name in enumerate(output_names):
                    domains[name].add(output[i])
            else:
                assert isinstance(output_names, str)
                domains[output_names].add(output)

        # Initialize a BDD with a root.
        bdd = BDD()
        root = bdd.true

        # Add all variables to the BDD.
        for name, domain in domains.items():
            root = add_var(name, domain, bdd, root)

        # TODO: add all composite variables encountered in the tree
        # If this turns out to not be necessary after finishing the mappings,
        # remove the add_composite_var function and disallow non-n BinOp in
        # the mapping_to_tree function.

        def output_equals(output: MappingTreeOutput):
            """Return condition on the output being equal to a specific output."""
            return (
                equality_condition("f_out", output[0], bdd, input_domain)
                & equality_condition("y_out", output[1], bdd, input_domain)
                & equality_condition("x_out", output[2], bdd, input_domain)
            )

    # Add the paths as restrictions.
    one_path_holds = bdd.false
    for conditions, output in extract_tree_paths(n):
        one_condition_false = bdd.false
        for left, right in conditions:
            one_condition_false = one_condition_false | ~equality_condition(
                left, right, bdd, input_domain
            )
        root = root & (one_condition_false | output_equals(output))
        one_path_holds = one_path_holds | ~one_condition_false

    # If none of the paths hold, the default output of (f, y, x) holds.
    root = root & (one_path_holds | output_equals(("f_in", "y_in", "x_in")))

    mappings: list[
        tuple[
            tuple[int | None, int | None, int | None],
            tuple[int, int, int],
            tuple[int, int, int],
        ]
    ] = []

    # Extract mappings from the diagram.
    for subset in minimal_substitution_subsets(
        move_vars | input_vars | output_vars, bdd, root
    ):
        vals = {k: v for k, v in subset}
        mappings.append(
            (
                (vals.get("ma"), vals.get("mi"), vals.get("md")),
                (vals["f_in"], vals["y_in"], vals["x_in"]),
                (vals["f_out"], vals["y_out"], vals["x_out"]),
            )
        )

    # Sort to make result deterministic.
    mappings.sort(key=lambda sc: str(sc))

    with open(path, "w") as file:
        for move, input, output in mappings:
            move_str = "".join(["*" if c is None else str(c) for c in move])
            input_str = "".join([str(c) for c in input])
            output_str = "".join([str(c) for c in output])
            file.write(f"{move_str} {input_str} {output_str}\n")


def load(n: int):
    with open(file_path(n), "r") as file:
        result = {
            ma: {
                mi: {md: [] for md in list(range(3)) + [None]}
                for mi in list(range(n)) + [None]
            }
            for ma in list(range(3)) + [None]
        }

        for line in file:
            move_raw, input_raw, output_raw = line.split(" ")
            move = (
                None if move_raw[0] == "*" else int(move_raw[0]),
                None if move_raw[1] == "*" else int(move_raw[1]),
                None if move_raw[2] == "*" else int(move_raw[2]),
            )
            input = (int(input_raw[0]), int(input_raw[1]), int(input_raw[2]))
            output = (int(output_raw[0]), int(output_raw[1]), int(output_raw[2]))
            result[move[0]][move[1]][move[2]].append((input, output))

        return result


# e.g. python move_mapping.py {n}
if __name__ == "__main__":
    start = datetime.now()
    generate(int(sys.argv[1]))
    print(f"took {datetime.now()-start} to complete!")
