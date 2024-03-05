import os
import sys
import ast
import inspect
import typing
import itertools
from datetime import datetime
from functools import reduce
from dd.cudd import BDD, Function
from misc import print_stamped, create_parent_directory
from logic import (
    coord_mapping,
    corner_rotation_mapping,
    edge_rotation_mapping,
    list_centers,
    list_corners,
    list_edges,
)

MappingTreeOutput = tuple[int | str, ...]  # return tuple
MappingTreeComparison = tuple[str, bool, str | int]  # {var} =|≠ {var|const} comparison
MappingTree = dict[MappingTreeComparison | None, "MappingTree"] | MappingTreeOutput


def mapping_to_tree(n: int, mapping: typing.Callable) -> MappingTree:
    """Convert a mapping function to a tree by parsing the function's AST."""

    def convert_expression(v: ast.expr) -> int | str:
        if isinstance(v, ast.Constant):  # constant integer
            return v.value
        elif isinstance(v, ast.Name):  # variable name
            if v.id == "n":  # n is known, so return that instead
                return n
            return v.id
        elif isinstance(v, ast.BinOp):  # binary operation, replace known "n - 1" or "n"
            s = ast.unparse(v)
            s = s.replace("n - 1", str(n - 1))
            s = s.replace("n", str(n))
            return s
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


def extract_paths(tree: MappingTree):
    """Return all paths of comparisons that lead to outputs in a mapping tree."""
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


def encode(var_name: str, var_value: int | str):
    """Encode a variable with its value in a string."""
    return f"{var_name}_{var_value}"


def declare_var(name: str, domain: frozenset[int | str], bdd: BDD):
    """Add a variable to the BDD by declaring it."""
    for value in domain:
        bdd.declare(encode(name, value))


def restrict_var_domain(
    name: str,
    domain: frozenset[int | str],
    input_domains: dict[str, frozenset[int | str]],
    output_domain: frozenset[int | str],
    root: Function,
    bdd: BDD,
) -> Function:
    """Restrict a value to take exactly one of the values of the domain."""

    variant_vars = []
    variant_values = []
    for value in domain:
        variant_vars.append(bdd.var(encode(name, value)))
        variant_values.append(value)

    # At least one variable value is true.
    root = root & reduce(lambda x, y: x | y, variant_vars)

    # At most one variable value is true.
    for i in range(len(variant_vars)):
        for j in range(i + 1, len(variant_vars)):
            root = root & (
                ~variant_vars[i]
                | ~variant_vars[j]
                | equality_condition(
                    variant_values[i],
                    variant_values[j],
                    input_domains,
                    output_domain,
                    bdd,
                )
            )

    return root


def equality_condition(
    left: str | int,
    right: str | int,
    input_domains: dict[str, frozenset[int | str]],
    output_domain: frozenset[int | str],
    bdd: BDD,
) -> Function:
    """Return a condition on a variable being equal to a constant or other variable."""
    if left == "output":
        if right not in output_domain:
            return bdd.false
        return bdd.var(encode("output", right))
    elif right == "output":
        if left not in output_domain:
            return bdd.false
        return bdd.var(encode("output", left))
    elif isinstance(left, int):
        if isinstance(right, int):
            return bdd.true if left == right else bdd.false
        else:
            if left not in input_domains[right]:
                return bdd.false
            return bdd.var(encode(right, left))
    elif isinstance(right, int):
        if right not in input_domains[left]:
            return bdd.false
        return bdd.var(encode(left, right))
    else:
        if left not in input_domains or right not in input_domains:
            return bdd.false
        return reduce(
            lambda x, y: x | y,
            [
                bdd.var(encode(left, overlap)) & bdd.var(encode(right, overlap))
                for overlap in input_domains[left].intersection(input_domains[right])
            ],
        )  # exactly one of the overlap between their domains should be true.


Input = tuple[tuple[bool, str | int] | None, ...]


def minimal_inputs(
    input_names: tuple[str, ...],
    input_domains: dict[str, frozenset[int | str]],
    output_domain: frozenset[int | str],
    root: Function,
    bdd: BDD,
):
    """Extract all minimal-size inputs such that the output is deterministic."""
    result: list[tuple[Input, int | str]] = []

    def is_subset(inp1: Input, inp2: Input):
        for i in range(len(inp2)):
            if inp2[i] is None:
                if inp1[i] is not None:
                    return False
            elif inp1[i] is not None and inp1[i] != inp2[i]:
                return False
        return True

    def search_subsets(i: int, inp: Input, substituted: Function):
        if i < len(input_names):
            name = input_names[i]

            # Try skipping this input index.
            search_subsets(i + 1, inp + (None,), substituted)

            for value in input_domains[name]:
                # Try equality for this value of this input index.
                equal = bdd.let({encode(name, value): bdd.true}, substituted)
                if equal.count() > 0:  # else no use in continuing
                    search_subsets(i + 1, inp + ((True, value),), equal)

                # NOTE: commented out due to performance issues
                # # Try inequality for this value of this input.
                # inequal = bdd.let({encode(name, value): bdd.false}, substituted)
                # if inequal.count() > 0:  # else no use in continuing
                #     search_subsets(i + 1, inp + ((False, value),), inequal)

        else:
            # Only consider if no existing set is a subset.
            if all([not is_subset(ex, inp) for ex, _ in result]):
                outputs = None
                for assignment in bdd.pick_iter(substituted):
                    assignment_outputs = set()
                    for v in output_domain:
                        encoded = encode("output", v)
                        if encoded in assignment:
                            if assignment[encoded]:
                                assignment_outputs.add(v)

                    # Should hold by BDD construction.
                    assert len(assignment_outputs) > 0

                    if outputs is None:
                        outputs = assignment_outputs  # first iteration
                    elif outputs != assignment_outputs:
                        return None  # not deterministic

                # No. assignments is more than 0, so cannot be none.
                assert outputs is not None

                # All existing sets should not be supersets.
                assert all([not is_subset(inp, ex) for ex, _ in result])

                # NOTE: not sure whether this is correct
                assert len(outputs) == 1

                result.append((inp, list(outputs)[0]))

    search_subsets(0, tuple(), root)
    return result


def minimal_inputs_by_output(
    n: int,
    mapping: typing.Callable,
    input_names: tuple[str, ...],
    input_domain: frozenset[tuple[int, ...]],
    output_index: int,
):
    tree = mapping_to_tree(n, mapping)
    paths = extract_paths(tree)

    # Initialize separate domains for all inputs.
    input_domains: dict[str, frozenset[int | str]] = {}
    for i, name in enumerate(input_names):
        input_domains[name] = frozenset([inp[i] for inp in input_domain])

    # List the inputs that are allowed by the separate domains, but not by the
    # domain of all inputs together.
    banned_inputs = [
        inp
        for inp in itertools.product(*[d for d in input_domains.values()])
        if inp not in input_domain
    ]

    # NOTE: commented out due to performance issues
    # # Add symbolic domains between inputs with overlapping domains.
    # for name1, name2 in itertools.combinations(input_domains, 2):
    #     if len(input_domains[name1].intersection(input_domains[name2])) > 0:
    #         input_domains[name1] |= {name2}

    # Initialize the output domain.
    output_domain = frozenset([output[output_index] for output in paths.values()])

    # Initialize a BDD with a root.
    bdd = BDD()
    root = bdd.true

    # Declare all inputs and the output.
    for name, domain in input_domains.items():
        declare_var(name, domain, bdd)
    declare_var("output", output_domain, bdd)

    # Add all domain restrictions.
    for name, domain in input_domains.items():
        root = restrict_var_domain(
            name, domain, input_domains, output_domain, root, bdd
        )
    root = restrict_var_domain(
        "output", output_domain, input_domains, output_domain, root, bdd
    )

    # Add restrictions on symbolic equalities between inputs.
    for name, domain in input_domains.items():
        for v in domain:
            if isinstance(v, str) and v in input_domains:
                root = root & bdd.var(encode(name, v)).equiv(
                    equality_condition(name, v, input_domains, output_domain, bdd)
                )

    def input_equals(input: tuple[int | str, ...]):
        """Return condition on the input being equal to a specific input."""
        return reduce(
            lambda x, y: x & y,
            [
                equality_condition(
                    input_name, input[i], input_domains, output_domain, bdd
                )
                for i, input_name in enumerate(input_names)
            ],
        )

    # Disallow banned inputs.
    for inp in banned_inputs:
        root = root & ~input_equals(inp)

    # Add the paths as restrictions.
    for comparisons, output in paths.items():
        path_holds = bdd.true
        for left, equals, right in comparisons:
            if equals:
                path_holds = path_holds & equality_condition(
                    left, right, input_domains, output_domain, bdd
                )
            else:
                path_holds = path_holds & ~equality_condition(
                    left, right, input_domains, output_domain, bdd
                )
        root = root & (
            ~path_holds
            | equality_condition(
                "output", output[output_index], input_domains, output_domain, bdd
            )
        )

    return minimal_inputs(input_names, input_domains, output_domain, root, bdd)


def types(
    n: int,
) -> dict[
    str,
    tuple[
        typing.Callable,
        tuple[str, ...],  # input names
        frozenset[tuple[int, ...]],  # input domain
        tuple[str, ...],  # output names
    ],
]:
    corners = list_corners(n)
    centers = list_centers(n)
    edges = list_edges(n)

    return {
        "corner_coord": (
            coord_mapping,
            ("x", "y", "z", "ma", "mi", "md"),
            frozenset(
                corner + (ma, mi, md)
                for corner in corners
                for ma in range(3)
                for mi in range(n)
                for md in range(3)
            ),
            ("x_new", "y_new", "z_new"),
        ),
        "corner_rotation": (
            corner_rotation_mapping,
            ("x", "y", "z", "r", "ma", "mi", "md"),
            frozenset(
                corner + (r, ma, mi, md)
                for corner in corners
                for r in range(3)
                for ma in range(3)
                for mi in range(n)
                for md in range(3)
            ),
            ("r_new",),
        ),
        "center_coord": (
            coord_mapping,
            ("x", "y", "z", "ma", "mi", "md"),
            frozenset(
                center + (ma, mi, md)
                for center in centers
                for ma in range(3)
                for mi in range(n)
                for md in range(3)
            ),
            ("x_new", "y_new", "z_new"),
        ),
        "edge_coord": (
            coord_mapping,
            ("x", "y", "z", "ma", "mi", "md"),
            frozenset(
                edge + (ma, mi, md)
                for edge in edges
                for ma in range(3)
                for mi in range(n)
                for md in range(3)
            ),
            ("x_new", "y_new", "z_new"),
        ),
        "edge_rotation": (
            edge_rotation_mapping,
            ("x", "y", "z", "r", "ma", "mi", "md"),
            frozenset(
                edge + (r, ma, mi, md)
                for edge in edges
                for r in range(3)
                for ma in range(3)
                for mi in range(n)
                for md in range(3)
            ),
            ("r_new",),
        ),
    }


def file_path(n: int, type: str, output_name: str):
    return f"./move_mappings/n{n}-{type}-{output_name}.txt"


def generate(n: int, overwrite=False):
    for type, (mapping, input_names, input_domain, output_names) in types(n).items():
        if len(input_domain) == 0:
            continue  # can happen for n <= 2, where there are no edges or centers

        for i, output_name in enumerate(output_names):
            path = file_path(n, type, output_name)
            if not overwrite and os.path.isfile(path):
                continue
            create_parent_directory(path)

            print_stamped(
                f"computing move mapping for '{output_name}' for '{type}' with n = {n}..."
            )

            result = minimal_inputs_by_output(n, mapping, input_names, input_domain, i)
            result.sort(key=lambda x: str(x))

            with open(path, "w") as file:
                for input, output in result:
                    inputs = ["*" if v is None else str(v[1]) for v in input]
                    ops = ["*" if v is None else "=" if v[0] else "≠" for v in input]
                    file.write(f"{''.join(inputs)}\t{''.join(ops)}\t{output}\n")


def load(n: int):
    result: dict[
        str, dict[str, list[tuple[dict[str, tuple[bool, int | str]], int | str]]]
    ] = {}

    for type, (_, input_names, _, output_names) in types(n).items():
        for output_name in output_names:
            path = file_path(n, type, output_name)
            if not os.path.isfile(path):
                continue  # assume it is intentionally missing

            mappings = []
            with open(path, "r") as file:
                for line in file:
                    raw_inputs, raw_ops, output = line.rstrip("\n").split("\t")

                    inputs = {}
                    for i, input_name in enumerate(input_names):
                        input = raw_inputs[i] if raw_inputs[i] != "*" else None
                        if input is not None and input.isnumeric():
                            input = int(input)
                        equality = raw_ops[i] != "≠"
                        if input is not None:
                            inputs[input_name] = (equality, input)

                    if output.isnumeric():
                        output = int(output)

                    mappings.append((inputs, output))

            if type not in result:
                result[type] = {}
            result[type][output_name] = mappings

    return result


# e.g. python move_mapping.py {n}
if __name__ == "__main__":
    start = datetime.now()
    generate(int(sys.argv[1]), True)
    print_stamped(f"took {datetime.now()-start} to complete!")
