from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
RESULTS = ROOT / "results"

NOTEBOOK_DIR_MOVES = {
    "00_fundamentals": "10_svd/00_fundamentals",
    "10_mnist_mlp": "10_svd/10_mnist_mlp",
    "20_fashion_mnist/mlp": "10_svd/20_fashion_mnist_mlp",
    "20_fashion_mnist/cnn": "10_svd/30_fashion_mnist_cnn",
    "30_cifar10": "10_svd/40_cifar10_cnn",
}

CASE_BY_TOP_DIR = {
    "10_mnist_mlp": "10_mnist_mlp",
    "20_fashion_mnist_mlp": "20_fashion_mnist_mlp",
    "30_fashion_mnist_cnn": "30_fashion_mnist_cnn",
    "40_cifar10_cnn": "40_cifar10_cnn",
}

OLD_PATH_REPLACEMENTS = {
    "notebooks/00_fundamentals/": "notebooks/10_svd/00_fundamentals/",
    "notebooks/10_mnist_mlp/": "notebooks/10_svd/10_mnist_mlp/",
    "notebooks/20_fashion_mnist/mlp/": "notebooks/10_svd/20_fashion_mnist_mlp/",
    "notebooks/20_fashion_mnist/cnn/": "notebooks/10_svd/30_fashion_mnist_cnn/",
    "notebooks/30_cifar10/": "notebooks/10_svd/40_cifar10_cnn/",
}


def move_path(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def remove_if_empty(path: Path) -> None:
    if path.exists() and path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def move_notebooks() -> None:
    for old_rel, new_rel in NOTEBOOK_DIR_MOVES.items():
        move_path(NOTEBOOKS / old_rel, NOTEBOOKS / new_rel)

    # 旧 placeholder は複数手法を混在させていたため、手法ごとの入口へ分ける。
    shutil.rmtree(NOTEBOOKS / "40_tensor_network_dmrg", ignore_errors=True)
    for method_dir in ["20_tucker", "30_tt_mps", "40_dmrg"]:
        directory = NOTEBOOKS / method_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch()

    old_fashion = NOTEBOOKS / "20_fashion_mnist"
    if old_fashion.exists():
        gitkeep = old_fashion / ".gitkeep"
        if gitkeep.exists():
            gitkeep.unlink()
        remove_if_empty(old_fashion)


def classify_fashion_result(name: str) -> str:
    if "_mlp_" in name:
        return "20_fashion_mnist_mlp"
    if "_cnn_" in name:
        return "30_fashion_mnist_cnn"
    raise ValueError(f"cannot classify Fashion-MNIST result directory: {name}")


def move_result_children(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        if child.name == ".gitkeep":
            child.unlink()
            continue
        move_path(child, destination / child.name)
    remove_if_empty(source)


def move_results() -> None:
    move_result_children(
        RESULTS / "10_mnist_mlp",
        RESULTS / "10_svd" / "10_mnist_mlp",
    )

    fashion_source = RESULTS / "20_fashion_mnist"
    if fashion_source.exists():
        for child in sorted(fashion_source.iterdir()):
            if child.name == ".gitkeep":
                child.unlink()
                continue
            case_name = classify_fashion_result(child.name)
            move_path(child, RESULTS / "10_svd" / case_name / child.name)
        remove_if_empty(fashion_source)

    move_result_children(
        RESULTS / "30_cifar10",
        RESULTS / "10_svd" / "40_cifar10_cnn",
    )


def write_paths_module() -> None:
    path = ROOT / "src" / "nn_compression" / "utils" / "paths.py"
    path.write_text(
        '''"""Notebook 実験で共通利用するパス処理。

Notebook と成果物の分類軸をそろえるため、実験成果物は
``method -> case -> experiment`` の順で保存する。
"""

from pathlib import Path


def find_project_root(start_path: Path | str) -> Path:
    """上位ディレクトリを探索し、``.git`` が存在する場所を返す。

    Notebook の作業ディレクトリが ``notebooks/...`` でも、data・models・
    results を常にプロジェクトルート基準で参照するために使う。
    """
    start_path = Path(start_path).resolve()

    for candidate in [start_path, *start_path.parents]:
        if (candidate / ".git").exists():
            return candidate

    raise FileNotFoundError(
        "プロジェクトルートを特定できませんでした。"
        "上位ディレクトリに.gitが存在するか確認してください。"
    )


def get_experiment_dirs(
    project_root: Path | str,
    method_name: str,
    case_name: str,
    experiment_name: str,
    *,
    create: bool = True,
) -> tuple[Path, Path, Path]:
    """data・models・results の実験用ディレクトリを返す。

    ``method_name`` / ``case_name`` / ``experiment_name`` を出力パスに含め、
    SVD・Tucker・TT/MPS・DMRG の各実験を同じ規則で整理する。
    """
    project_root = Path(project_root)
    data_dir = project_root / "data"

    # notebooks/ と results/models の分類軸を同じにするため、
    # method -> case -> experiment の順で保存する。
    models_dir = (
        project_root / "models" / method_name / case_name / experiment_name
    )
    results_dir = (
        project_root / "results" / method_name / case_name / experiment_name
    )

    if create:
        for directory in [data_dir, models_dir, results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    return data_dir, models_dir, results_dir
''',
        encoding="utf-8",
    )


def find_experiment_name(nb: dict, fallback: str) -> str:
    patterns = [
        re.compile(r'(?m)^\s*EXPERIMENT_NAME\s*=\s*["\']([^"\']+)["\']'),
        re.compile(r'(?m)^\s*experiment_name\s*=\s*["\']([^"\']+)["\']'),
    ]
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        for pattern in patterns:
            match = pattern.search(source)
            if match:
                return match.group(1)
    return fallback


def standard_path_cell(case_name: str, experiment_name: str) -> list[str]:
    text = f'''# ============================================================
# 保存先
# ============================================================
import sys
from pathlib import Path

# Notebook の配置が深くなっても src/ を見つけられるよう、
# プロジェクトルート候補を上へたどって import path を補う。
for _candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
    _src = _candidate / "src"
    if (_src / "nn_compression").is_dir():
        if str(_src) not in sys.path:
            sys.path.insert(0, str(_src))
        break

# 保存先の組み立ては Notebook 内へ重複実装せず、src の共通処理を使う。
from nn_compression.utils import find_project_root, get_experiment_dirs

# method-first のディレクトリ構成に合わせて実験を識別する。
METHOD_NAME = "10_svd"
CASE_NAME = "{case_name}"
EXPERIMENT_NAME = "{experiment_name}"

project_root = find_project_root(Path.cwd())
data_dir, models_dir, results_dir = get_experiment_dirs(
    project_root,
    METHOD_NAME,
    CASE_NAME,
    EXPERIMENT_NAME,
)

print(f"project_root: {{project_root}}")
print(f"data_dir:     {{data_dir}}")
print(f"models_dir:   {{models_dir}}")
print(f"results_dir:  {{results_dir}}")
'''
    return text.splitlines(keepends=True)


def transform_common_path_cell(source: str, case_name: str) -> str:
    replacement = (
        '# method-first の新しい構成に合わせて保存先を識別する。\n'
        'METHOD_NAME = "10_svd"\n'
        f'CASE_NAME = "{case_name}"\n'
    )
    source, count_upper = re.subn(
        r'(?m)^\s*DATASET_NAME\s*=\s*["\'][^"\']+["\']\s*\n',
        replacement,
        source,
        count=1,
    )
    if count_upper == 0:
        source, _ = re.subn(
            r'(?m)^\s*dataset_name\s*=\s*["\'][^"\']+["\']\s*\n',
            replacement,
            source,
            count=1,
        )

    source = re.sub(
        r'(get_experiment_dirs\(\s*\n\s*project_root,\s*\n)\s*(?:DATASET_NAME|dataset_name),\s*\n\s*EXPERIMENT_NAME,',
        r'\1    METHOD_NAME,\n    CASE_NAME,\n    EXPERIMENT_NAME,',
        source,
    )
    source = re.sub(
        r'get_experiment_dirs\(project_root,\s*(?:DATASET_NAME|dataset_name),\s*EXPERIMENT_NAME\)',
        'get_experiment_dirs(project_root, METHOD_NAME, CASE_NAME, EXPERIMENT_NAME)',
        source,
    )
    return source


def notebook_replacements(case_name: str) -> dict[str, str]:
    replacements = dict(OLD_PATH_REPLACEMENTS)
    old_case_roots = {
        "10_mnist_mlp": "10_mnist_mlp",
        "20_fashion_mnist_mlp": "20_fashion_mnist",
        "30_fashion_mnist_cnn": "20_fashion_mnist",
        "40_cifar10_cnn": "30_cifar10",
    }
    old_root = old_case_roots[case_name]
    replacements[f"results/{old_root}/"] = f"results/10_svd/{case_name}/"
    replacements[f"models/{old_root}/"] = f"models/10_svd/{case_name}/"
    replacements[
        "get_experiment_dirs(project_root, dataset_name, experiment_name)"
    ] = "get_experiment_dirs(project_root, method_name, case_name, experiment_name)"
    return replacements


def apply_string_replacements(value: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def update_notebook(path: Path, case_name: str) -> None:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    indent = 1
    if len(lines) > 1:
        match = re.match(r'^(\s+)"(?:cells|metadata|nbformat)"', lines[1])
        if match:
            indent = len(match.group(1))
    trailing_newline = raw.endswith("\n")

    nb = json.loads(raw)
    experiment_name = find_experiment_name(nb, path.stem)
    replacements = notebook_replacements(case_name)

    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        original_source = source

        is_local_path_definition = (
            cell.get("cell_type") == "code"
            and "def find_project_root" in source
            and "results_dir" in source
            and ("dataset_name" in source or "DATASET_NAME" in source)
        )
        is_common_path_setup = (
            cell.get("cell_type") == "code"
            and "get_experiment_dirs(" in source
            and ("DATASET_NAME" in source or "dataset_name" in source)
            and "def get_experiment_dirs" not in source
        )

        if is_local_path_definition:
            cell["source"] = standard_path_cell(case_name, experiment_name)
            cell["execution_count"] = None
            cell["outputs"] = []
            continue

        if is_common_path_setup:
            source = transform_common_path_cell(source, case_name)
            if source != original_source:
                cell["execution_count"] = None
                cell["outputs"] = []

        # cwd 相対の ./data は、深い階層へ移動すると別 data/ を作るため共通 data_dir へ寄せる。
        source = source.replace('root="./data"', "root=data_dir")
        source = source.replace("root='./data'", "root=data_dir")
        source = apply_string_replacements(source, replacements)

        if source != original_source:
            cell["source"] = source.splitlines(keepends=True)

    serialized = json.dumps(nb, ensure_ascii=False, indent=indent)
    if trailing_newline:
        serialized += "\n"
    path.write_text(serialized, encoding="utf-8")


def update_all_notebooks() -> None:
    svd_root = NOTEBOOKS / "10_svd"
    for case_dir_name, case_name in CASE_BY_TOP_DIR.items():
        case_dir = svd_root / case_dir_name
        if not case_dir.exists():
            raise FileNotFoundError(case_dir)
        for notebook in sorted(case_dir.rglob("*.ipynb")):
            update_notebook(notebook, case_name)


def update_text_references() -> None:
    candidates = [ROOT / "README.md"]
    candidates.extend((ROOT / ".cursor" / "rules").glob("*.mdc"))
    candidates.extend((ROOT / "plan").glob("*.md"))
    candidates.extend((ROOT / "docs").rglob("*.md"))

    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        updated = apply_string_replacements(text, OLD_PATH_REPLACEMENTS)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def cleanup_code_and_tools() -> None:
    tools = ROOT / "tools"
    tools.mkdir(exist_ok=True)
    source = NOTEBOOKS / "regen_notebook_cell_ids.py"
    destination = tools / "regen_notebook_cell_ids.py"
    if source.exists():
        if destination.exists():
            raise FileExistsError(destination)
        shutil.move(str(source), str(destination))

    # code/ は Notebook 実験の重複コピーなので canonical な置き場から外す。
    shutil.rmtree(ROOT / "code", ignore_errors=True)

    gitignore = ROOT / ".gitignore"
    if gitignore.exists():
        text = gitignore.read_text(encoding="utf-8")
        text = re.sub(
            r'\n# Legacy artifacts under code/.*?\n/code/data/MNIST/\n/code/mnist_mlp_baseline\.pth\n',
            "\n",
            text,
            flags=re.DOTALL,
        )
        gitignore.write_text(text, encoding="utf-8")


def write_path_tests() -> None:
    test_path = ROOT / "tests" / "test_paths.py"
    test_path.write_text(
        '''from pathlib import Path

from nn_compression.utils.paths import get_experiment_dirs


def test_get_experiment_dirs_uses_method_first_layout(tmp_path: Path) -> None:
    # notebooks/ と results/models が同じ分類軸になることを固定する回帰テスト。
    data_dir, models_dir, results_dir = get_experiment_dirs(
        tmp_path,
        "10_svd",
        "40_cifar10_cnn",
        "02_svd_global_compression",
    )

    assert data_dir == tmp_path / "data"
    assert models_dir == (
        tmp_path
        / "models"
        / "10_svd"
        / "40_cifar10_cnn"
        / "02_svd_global_compression"
    )
    assert results_dir == (
        tmp_path
        / "results"
        / "10_svd"
        / "40_cifar10_cnn"
        / "02_svd_global_compression"
    )
    assert data_dir.is_dir()
    assert models_dir.is_dir()
    assert results_dir.is_dir()
''',
        encoding="utf-8",
    )


def verify_notebook_sources() -> None:
    old_fragments = [
        "results/10_mnist_mlp/",
        "results/20_fashion_mnist/",
        "results/30_cifar10/",
        "models/10_mnist_mlp/",
        "models/20_fashion_mnist/",
        "models/30_cifar10/",
        'DATASET_NAME = "10_mnist_mlp"',
        'DATASET_NAME = "20_fashion_mnist"',
        'DATASET_NAME = "30_cifar10"',
        'dataset_name = "10_mnist_mlp"',
        'dataset_name = "20_fashion_mnist"',
        'dataset_name = "30_cifar10"',
        'root="./data"',
        "root='./data'",
    ]

    failures: list[str] = []
    for case_dir_name in CASE_BY_TOP_DIR:
        for notebook in (NOTEBOOKS / "10_svd" / case_dir_name).rglob("*.ipynb"):
            nb = json.loads(notebook.read_text(encoding="utf-8"))
            sources = "\n".join(
                "".join(cell.get("source", [])) for cell in nb.get("cells", [])
            )
            for fragment in old_fragments:
                if fragment in sources:
                    failures.append(f"{notebook.relative_to(ROOT)}: {fragment}")
            if "get_experiment_dirs(" in sources:
                if 'METHOD_NAME = "10_svd"' not in sources or "CASE_NAME = " not in sources:
                    failures.append(
                        f"{notebook.relative_to(ROOT)}: missing METHOD_NAME/CASE_NAME"
                    )

    if failures:
        raise AssertionError("stale Notebook path settings:\n" + "\n".join(failures))


def verify_layout() -> None:
    expected = [
        NOTEBOOKS / "10_svd" / "00_fundamentals",
        NOTEBOOKS / "10_svd" / "10_mnist_mlp",
        NOTEBOOKS / "10_svd" / "20_fashion_mnist_mlp",
        NOTEBOOKS / "10_svd" / "30_fashion_mnist_cnn",
        NOTEBOOKS / "10_svd" / "40_cifar10_cnn",
        NOTEBOOKS / "20_tucker",
        NOTEBOOKS / "30_tt_mps",
        NOTEBOOKS / "40_dmrg",
        RESULTS / "10_svd" / "10_mnist_mlp",
        RESULTS / "10_svd" / "20_fashion_mnist_mlp",
        RESULTS / "10_svd" / "30_fashion_mnist_cnn",
        RESULTS / "10_svd" / "40_cifar10_cnn",
        ROOT / "tools" / "regen_notebook_cell_ids.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
    if missing:
        raise AssertionError("missing expected paths:\n" + "\n".join(missing))

    forbidden = [
        NOTEBOOKS / "00_fundamentals",
        NOTEBOOKS / "10_mnist_mlp",
        NOTEBOOKS / "20_fashion_mnist",
        NOTEBOOKS / "30_cifar10",
        NOTEBOOKS / "40_tensor_network_dmrg",
        RESULTS / "10_mnist_mlp",
        RESULTS / "20_fashion_mnist",
        RESULTS / "30_cifar10",
        ROOT / "code",
    ]
    leftovers = [str(path.relative_to(ROOT)) for path in forbidden if path.exists()]
    if leftovers:
        raise AssertionError("old paths still exist:\n" + "\n".join(leftovers))


def main() -> None:
    move_notebooks()
    move_results()
    cleanup_code_and_tools()
    write_paths_module()
    update_all_notebooks()
    update_text_references()
    write_path_tests()
    verify_layout()
    verify_notebook_sources()
    print("repository reorganization checks passed")


if __name__ == "__main__":
    main()
