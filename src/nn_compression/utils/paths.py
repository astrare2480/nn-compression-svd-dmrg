"""Notebook 実験で共通利用するパス処理。

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
