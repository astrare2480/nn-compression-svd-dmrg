"""Notebook 実験で共通利用するパス処理。

参照元:
    notebooks/20_fashion_mnist/mlp/00_mlp_baseline_fixed_50epochs.ipynb
    notebooks/20_fashion_mnist/mlp/01_mlp_baseline_early_stopping.ipynb
    notebooks/20_fashion_mnist/mlp/02_mlp_svd_rank_selection.ipynb
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
"""

from pathlib import Path


def find_project_root(start_path: Path | str) -> Path:
    """上位ディレクトリを探索し、``.git`` が存在する場所を返す。

    Notebook の作業ディレクトリが ``notebooks/...`` でも、data・models・
    results を常にプロジェクトルート基準で参照するために使う。
    """
    start_path = Path(start_path).resolve()

    for candidate in [
        start_path,
        *start_path.parents,
    ]:
        if (candidate / ".git").exists():
            return candidate

    raise FileNotFoundError(
        "プロジェクトルートを特定できませんでした。"
        "上位ディレクトリに.gitが存在するか確認してください。"
    )


def get_experiment_dirs(
    project_root: Path | str,
    dataset_name: str,
    experiment_name: str,
    *,
    create: bool = True,
) -> tuple[Path, Path, Path]:
    """data・models・results の実験用ディレクトリを返す。

    ``dataset_name`` と ``experiment_name`` を出力パスに含めるので、
    別実験が既存のCSV・図・重みを誤って上書きしない。
    """
    project_root = Path(project_root)
    data_dir = project_root / "data"
    models_dir = project_root / "models" / dataset_name / experiment_name
    results_dir = project_root / "results" / dataset_name / experiment_name

    if create:
        for directory in [
            data_dir,
            models_dir,
            results_dir,
        ]:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    return data_dir, models_dir, results_dir
