"""TT/MPS、DMRG、NN圧縮のパッケージ依存方向。"""

from __future__ import annotations

import ast
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src" / "nn_compression"
FORBIDDEN = {"compression", "training"}


def _package_context(path: Path) -> str:
    """相対importの解決に使う、``nn_compression`` からの package パス。"""
    return ".".join(path.relative_to(SRC).parent.parts)


def _resolve_import(package: str, level: int, module: str | None) -> str | None:
    """``ImportFrom.level`` を含めて絶対module名へ直す。"""
    if level == 0:
        return module
    full = f"nn_compression.{package}" if package else "nn_compression"
    bits = full.rsplit(".", level - 1)
    if len(bits) < level:
        return None
    base = bits[0]
    if module:
        return f"{base}.{module}"
    return base


def _child_module(resolved: str, name: str) -> str | None:
    """``from package import name`` の name が実在する子moduleなら絶対名を返す。"""
    if resolved == "nn_compression":
        parent = SRC
        absolute = f"nn_compression.{name}"
    elif resolved.startswith("nn_compression."):
        parent = SRC.joinpath(*resolved.removeprefix("nn_compression.").split("."))
        absolute = f"{resolved}.{name}"
    else:
        return None
    if (parent / f"{name}.py").is_file() or (parent / name / "__init__.py").is_file():
        return absolute
    return None


def _resolved_imports(path: Path, source: str | None = None) -> set[str]:
    """ファイル内のimportを、相対level込みで絶対module名へ解決する。"""
    tree = ast.parse(path.read_text(encoding="utf-8") if source is None else source)
    package = _package_context(path)
    resolved: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            resolved.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        target = _resolve_import(package, node.level, node.module)
        if target is None:
            continue
        resolved.add(target)
        for alias in node.names:
            child = _child_module(target, alias.name)
            if child is not None:
                resolved.add(child)
    return resolved


def _component(module: str) -> str | None:
    prefix = "nn_compression."
    if module == "nn_compression" or not module.startswith(prefix):
        return None
    rest = module.removeprefix(prefix)
    return rest.split(".", 1)[0] if rest else None


def _internal_parts(module: str) -> set[str]:
    prefix = "nn_compression."
    if not module.startswith(prefix):
        return set()
    return set(module.removeprefix(prefix).split("."))


def _dependency_graph() -> dict[str, set[str]]:
    """top-level package から、別の top-level package へのimport辺。"""
    graph: dict[str, set[str]] = {}
    for path in SRC.rglob("*.py"):
        source = _component(f"nn_compression.{_package_context(path)}")
        if path.name != "__init__.py":
            source = _component(
                "nn_compression." + ".".join(path.relative_to(SRC).with_suffix("").parts)
            )
        if source is None:
            continue
        graph.setdefault(source, set())
        for module in _resolved_imports(path):
            target = _component(module)
            if target is not None and target != source:
                graph[source].add(target)
    return graph


def _reachable(graph: dict[str, set[str]], start: str) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph.get(node, ()))
    seen.discard(start)
    return seen


def _find_cycle(graph: dict[str, set[str]]) -> list[str]:
    visiting: list[str] = []
    visited: set[str] = set()

    def walk(node: str) -> list[str] | None:
        if node in visiting:
            return visiting[visiting.index(node):] + [node]
        if node in visited:
            return None
        visiting.append(node)
        for nxt in sorted(graph.get(node, ())):
            cycle = walk(nxt)
            if cycle:
                return cycle
        visiting.pop()
        visited.add(node)
        return None

    for node in sorted(graph):
        cycle = walk(node)
        if cycle:
            return cycle
    return []


def test_tt_mps_does_not_import_compression_or_training():
    offenders = []
    for path in (SRC / "tt_mps").rglob("*.py"):
        for module in _resolved_imports(path):
            if _component(module) in FORBIDDEN:
                offenders.append(f"{path.name}: {module}")
    assert offenders == []


def test_dmrg_core_does_not_import_compression_training_support_or_models():
    blocked = FORBIDDEN | {"support", "models"}
    offenders = []
    for path in (SRC / "dmrg").glob("*.py"):
        for module in _resolved_imports(path):
            parts = _internal_parts(module)
            if _component(module) in blocked or parts & blocked:
                offenders.append(f"{path.name}: {module}")
    assert offenders == []


def test_metrics_does_not_import_training():
    graph = _dependency_graph()
    assert "training" not in _reachable(graph, "metrics")


def test_compression_does_not_import_training():
    graph = _dependency_graph()
    assert "training" not in _reachable(graph, "compression")


def test_training_may_import_compression():
    graph = _dependency_graph()
    assert "compression" in graph["training"]


def test_package_dependency_graph_has_no_cycles():
    cycle = _find_cycle(_dependency_graph())
    assert cycle == []


def test_detector_uses_import_level_for_parent_and_package_imports():
    """``node.module`` だけだと落ちる相対importを、level と子module名で拾う。"""
    path = SRC / "compression" / "rank_sweep.py"
    modules = _resolved_imports(
        path,
        "from ..training.loops import evaluate\nfrom .. import training\n",
    )
    assert "nn_compression.training.loops" in modules
    assert "nn_compression.training" in modules


def test_public_package_exports_match_the_new_boundary():
    import nn_compression.compression as compression
    import nn_compression.dmrg as dmrg
    import nn_compression.dmrg.models as models
    import nn_compression.dmrg.support as support
    import nn_compression.tt_mps as tt_mps

    generic = {
        "tt_unfold", "tt_svd_exact", "tt_svd", "tt_svd_error_bound", "tt_svd_ranks",
        "tt_reconstruct", "tt_num_parameters", "tt_physical_shape", "tt_ranks",
        "tt_canonicalize", "tt_canonicality_errors", "tt_move_center",
        "tt_truncate_bond", "tt_bond_singular_values", "tt_round",
        "tt_inner", "tt_fro_norm",
        "tt_matrix_ranks", "tt_matrix_num_parameters", "tt_matrix_to_dense",
        "dense_to_tt_matrix_tensor", "tt_cores_to_tt_matrix_cores",
        "tt_matrix_cores_to_tt_cores", "dense_to_tt_matrix_cores",
        "TTMatrixPairSplit", "merge_tt_matrix_cores", "merged_tt_matrix_to_matrix",
        "split_merged_tt_matrix", "canonicalize_tt_matrix_cores",
        "tt_matrix_canonicality_errors", "replace_tt_matrix_core_pair",
        "ordered_factorizations", "tt_matrix_physical_sizes",
        "tt_matrix_shape_imbalance", "tt_matrix_num_parameters_from_ranks",
    }
    assert set(tt_mps.__all__) == generic
    assert set(generic).isdisjoint(compression.__all__)
    assert {
        "tt_linear_forward",
        "tt_linear_forward_with_merged_pair",
        "assign_tt_linear_cores",
        "module_forward_with_merged_tt_linear_pair",
        "module_forward_with_tt_linear_cores",
        "TTLinear",
        "TTConv2d",
        "tt_conv2d_forward",
        "sweep_tt_matrix_weight_ranks",
    } <= set(compression.__all__)
    assert len(dmrg.__all__) == 26
    assert set(support.__all__) == {
        "make_random_mps",
        "mpo_to_dense",
        "compute_dense_residual_and_variance",
        "ResidualVarianceDiagnostics",
    }
    assert set(models.__all__) == {"build_tfim_mpo", "build_tfim_dense_hamiltonian"}
