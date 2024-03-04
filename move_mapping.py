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


def declare_var(name: str, domain: set[int | str], bdd: BDD):
    """Add a variable to the BDD by declaring it."""
    for value in domain:
        bdd.declare(encode(name, value))


def restrict_var_domain(
    name: str,
    domain: set[int | str],
    input_domains: dict[str, set[int | str]],
    output_domain: set[int | str],
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
    input_domains: dict[str, set[int | str]],
    output_domain: set[int | str],
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


def minimal_subsets(
    input_domains: dict[str, set[int | str]],
    output_domain: set[int | str],
    root: Function,
    bdd: BDD,
):
    """Extract all minimal-size substitutions sets for the inputs such that the
    output is deterministic."""
    subsets: list[tuple[dict[str, tuple[bool, int | str]], int | str]] = []

    def extract_deterministic_output(
        remaining: list[str], substituted: Function
    ) -> None | int | str:
        # If there are no assignments, there is no use in continuing.
        if substituted.count() == 0:
            return None

        if len(remaining) > 0:
            name = remaining[0]
            encountered = None
            for v in input_domains[name]:
                output = extract_deterministic_output(
                    remaining[1:], bdd.let({encode(name, v): bdd.true}, substituted)
                )
                if output is None:
                    return None  # if one branch has no output, not deterministic
                if encountered is None:
                    encountered = output
                elif encountered != output:
                    return None  # output is not unique, not deterministic
        else:
            output = None
            for v in output_domain:
                if bdd.let({encode("output", v): bdd.true}, substituted).count() > 0:
                    if output is not None:
                        # There are at least two output values resulting in a satisfiable BDD,
                        # so output is not deterministic.
                        return None
                    output = v
            return output

    def search(
        remaining: list[str],
        chosen: dict[str, tuple[bool, int | str]],
        skipped: frozenset[str],
        substituted: Function,
    ):
        if len(remaining) > 0:
            name = remaining[0]
            search(remaining[1:], chosen, skipped | {name}, substituted)
            for value in input_domains[name]:
                equal = bdd.let({encode(name, value): bdd.true}, substituted)
                not_equal = bdd.let({encode(name, value): bdd.false}, substituted)

                if equal.count() > 0:
                    search(
                        remaining[1:], chosen | {name: (True, value)}, skipped, equal
                    )

                if not_equal.count() > 0:
                    search(
                        remaining[1:],
                        chosen | {name: (False, value)},
                        skipped,
                        not_equal,
                    )
        else:
            # Only consider is no previously found set is a subset.
            if all([not ex.items() <= chosen.items() for ex, _ in subsets]):
                output = extract_deterministic_output(list(skipped), substituted)
                if output is not None:
                    # No previously found set is a superset.
                    assert all([not chosen.items() <= ex.items() for ex, _ in subsets])
                    subsets.append((chosen, output))

    search(list(input_domains.keys()), {}, frozenset(), root)
    return subsets


def minimal_inputs_by_output(
    n: int,
    mapping: typing.Callable,
    input_names: tuple[str, ...],
    input_domain: set[tuple[int, ...]],
    output_index: int,
):
    tree = mapping_to_tree(n, mapping)
    paths = extract_paths(tree)

    # Initialize separate domains for all inputs.
    input_domains: dict[str, set[int | str]] = {}
    for i, name in enumerate(input_names):
        input_domains[name] = set()
        for inp in input_domain:
            input_domains[name].add(inp[i])

    # List the inputs that are allowed by the separate domains, but not by the
    # domain of all inputs together.
    banned_inputs = [
        inp
        for inp in itertools.product(*[d for d in input_domains.values()])
        if inp not in input_domain
    ]

    # Add symbolic domains between inputs with overlapping domains.
    for name1, name2 in itertools.combinations(input_domains, 2):
        if len(input_domains[name1].intersection(input_domains[name2])) > 0:
            input_domains[name1].add(name2)

    # Initialize the output domain.
    output_domain: set[int | str] = set()
    for output in paths.values():
        output_domain.add(output[output_index])

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

    return minimal_subsets(input_domains, output_domain, root, bdd)


def types(
    n: int,
) -> dict[
    str,
    tuple[
        typing.Callable,
        tuple[str, ...],  # input names
        set[tuple[int, ...]],  # input domain
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
            set(
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
            set(
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
            set(
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
            set(
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
            set(
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
                    input = [input[n] if n in input else None for n in input_names]
                    values = ["*" if i is None else str(i[1]) for i in input]
                    ops = ["*" if i is None else "=" if i[0] else "≠" for i in input]
                    file.write(f"{''.join(values)} {''.join(ops)} {output}\n")


def load(n: int):
    result: dict[str, dict[str, list[tuple[dict[str, int], int | str]]]] = {}

    for type, (_, input_names, _, output_names) in types(n).items():
        for output_name in output_names:
            path = file_path(n, type, output_name)
            if not os.path.isfile(path):
                continue  # assume it is intentionally missing

            mappings = []
            with open(path, "r") as file:
                for line in file:
                    raw_input, raw_output = line.split(" ")
                    mappings.append(
                        (
                            {
                                input_names[i]: int(v)
                                for i, v in enumerate(ast.literal_eval(raw_input))
                                if v != "*"
                            },
                            ast.literal_eval(raw_output),
                        )
                    )

            if type not in result:
                result[type] = {}
            result[type][output_name] = mappings

    return result


# e.g. python move_mapping.py {n}
if __name__ == "__main__":
    start = datetime.now()
    generate(int(sys.argv[1]), True)
    print_stamped(f"took {datetime.now()-start} to complete!")
