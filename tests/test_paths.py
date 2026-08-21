from pathlib import Path

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
