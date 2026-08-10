"""2つの最小化目的に対する Pareto frontier 抽出。

参照元:
    notebooks/20_fashion_mnist/mlp/02_mlp_svd_rank_selection.ipynb
    notebooks/20_fashion_mnist/mlp/03_mlp_svd_finetuning.ipynb
"""


def is_pareto_candidate(df, row, x_column, y_column):
    """row が他の点に劣後していなければ ``True`` を返す。

    2目的とも小さいほど良い前提で、他の候補が両方以上に良く、少なくとも
    一方で厳密に良い場合、その候補は支配されている。
    """
    dominated = (
        (df[x_column] <= row[x_column])
        & (df[y_column] <= row[y_column])
        & (
            (df[x_column] < row[x_column])
            | (df[y_column] < row[y_column])
        )
    )
    return not dominated.any()


def extract_pareto_frontier(df, x_column, y_column):
    """両列とも小さいほど良い場合の Pareto frontier を返す。

    Fashion-MNIST実験では ``parameters`` と ``validation_loss`` に用い、
    圧縮率と性能のどちらも一方的に劣る候補を除外する。
    """
    keep_indices = []

    for index, row in df.iterrows():
        if is_pareto_candidate(
            df,
            row,
            x_column,
            y_column,
        ):
            keep_indices.append(index)

    pareto = df.loc[keep_indices].copy()
    return pareto.sort_values(x_column).reset_index(drop=True)
