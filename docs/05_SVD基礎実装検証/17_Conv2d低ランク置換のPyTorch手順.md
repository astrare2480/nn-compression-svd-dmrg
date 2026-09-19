# Conv2d低ランク置換のPyTorch手順

数式と考え方は [[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]] を参照する。このノートはPyTorch・Pythonの操作、コード例、実装上の注意を扱う。教材のコード例と現行の共通関数は区別し、公開APIの仕様は [[07_src設計/05_Core_API_v1]] を参照する。

---

## 12. optimizerを作り直す理由

SVD前は、

```text
1つのConv2d Parameter群
```

だった。

SVD後は、

```text
1層目ConvのParameter群
+
2層目ConvのParameter群
```

という新しい `Parameter` オブジェクトになる。

元optimizerは、作成時に登録された古いParameterを参照している。

したがって、構造変更後は、

```python
optimizer = torch.optim.Adam(
    compressed_model.parameters(),
    lr=...,
)
```

と作り直す。

Adamなら、parameterごとに、

- `step`
- `exp_avg`
- `exp_avg_sq`

などの内部状態も持つ。

したがって、

```text
weightはSVDから継承
optimizer stateは継承しない
```

と考えると分かりやすい。

詳細は [[05_SVD基礎実装検証/12_PyTorch学習と評価の基礎]]。

---

## 14. `copy.deepcopy` と層置換

元モデルを残したまま候補モデルを作るなら、

```python
compressed_model = copy.deepcopy(model)
```

してから置換する。

現在の便利関数、

```python
factorize_named_conv2d(
    model,
    "conv2",
    rank,
)
```

も、元モデルをdeepcopyし、コピー側だけを変更する。

```text
元モデル
→ 変更しない

圧縮モデル
→ 独立したModule / Parameter群
```

となる。

`model.conv2` へ `(64,288)` のTensorそのものを代入してはいけない。

`model.conv2` は `nn.Module` としてforward時に呼び出されるため、置換後も、

```text
nn.Conv2d
または
nn.Sequential
```

などの `nn.Module` である必要がある。

---

## 15. deviceとdtypeを維持する

新しいConv層はデフォルトではCPU / default dtypeで作られる。

元モデルがGPU上や別dtypeにある場合に備えて、

```python
return nn.Sequential(
    first_layer,
    second_layer,
).to(
    device=conv.weight.device,
    dtype=conv.weight.dtype,
)
```

のように、元weightのdevice / dtypeを引き継ぐ。

---

## 16. `groups` の扱い

現在の `factorize_conv2d_layer()` は、

```text
groups = 1
```

のConv2dだけを対象とする。

```python
if conv.groups != 1:
    raise ValueError(...)
```

として、未対応の構造を明示する。

Depthwise Convなどへ同じfactorizationをそのまま適用するのは別問題である。

一方、MACsを数える汎用関数はgrouped Convのweight shapeを考慮している。詳しくは [[00_基礎理論/04_実験設計/11_理論計算量とベンチマーク]]。

---

## 20. コメント付き処理を、一つの関数で追う

ここまでの行列化、打ち切りSVD、reshape、2層生成を一つの教材用関数へまとめる。
これは上記srcの公開関数を変更・再定義するものではなく、各処理とshapeの対応を
一か所で確認するための独立した例である。対象は実数float32/float64の通常Conv2d、
`groups=1` とする。

Linear層の属性は `in_features` / `out_features` だが、Conv2dでは
`in_channels` / `out_channels` を使う。引数にはモデル全体や取り出した行列ではなく、
設定とweightを持つConv層一つを渡す。戻り値はTensorではなく、forwardを実行できる
`nn.Sequential` である。

```python
import copy
import numbers
import torch
from torch import nn


def factorize_conv2d_svd_example(conv: nn.Conv2d, rank: int) -> nn.Sequential:
    """行列化→打ち切りSVD→reshape→2層生成を追う教材用の例。"""
    if not isinstance(conv, nn.Conv2d):
        raise TypeError("通常のnn.Conv2dを渡してください。")
    if conv.groups != 1:
        raise ValueError("この教材例はgroups=1だけを対象にします。")
    if conv.weight.dtype not in (torch.float32, torch.float64):
        raise TypeError("この教材例は実数float32/float64を対象にします。")

    # 各出力filterを1行にする。detachは元weightを変更する処理ではない。
    out_ch, in_ch, k_h, k_w = conv.weight.shape
    max_rank = min(out_ch, in_ch * k_h * k_w)
    if isinstance(rank, bool) or not isinstance(rank, numbers.Integral):
        raise TypeError("rankはboolでない整数にしてください。")
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rankは1〜{max_rank}にしてください。")
    rank = int(rank)
    weight_matrix = conv.weight.detach().flatten(start_dim=1)

    # Reduced SVD。特異値は大きい順で、上位rank成分だけを残す。
    U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
    first_factor = Vh_r       # (rank, in_ch*k_h*k_w)
    second_factor = U_r * S_r # (out_ch, rank)、各列へS_rを掛ける。

    # 元dtype/deviceで最初から作り、copy_時にfloat32を経由させない。
    factory = {"device": conv.weight.device, "dtype": conv.weight.dtype}
    first_layer = nn.Conv2d(
        conv.in_channels,
        rank,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        padding_mode=conv.padding_mode,
        groups=1,
        bias=False,
        **factory,
    )
    # 位置ごとのchannel混合。空間サイズをさらに変えない。
    second_layer = nn.Conv2d(
        rank,
        conv.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        bias=(conv.bias is not None),
        **factory,
    )

    # SVD行列の要素を、Convが要求する4階weightの添字へ戻して代入する。
    with torch.no_grad():
        first_layer.weight.copy_(
            first_factor.reshape(rank, in_ch, k_h, k_w)
        )
        second_layer.weight.copy_(
            second_factor.reshape(out_ch, rank, 1, 1)
        )
        if conv.bias is not None:
            # 元biasは最終出力へ一度だけ加える。
            second_layer.bias.copy_(conv.bias)

    # freeze設定はweight/bias別に引き継ぎ、train/eval状態も合わせる。
    first_layer.weight.requires_grad_(conv.weight.requires_grad)
    second_layer.weight.requires_grad_(conv.weight.requires_grad)
    if conv.bias is not None:
        second_layer.bias.requires_grad_(conv.bias.requires_grad)
    return nn.Sequential(first_layer, second_layer).train(conv.training)


# 小さい独立層で確認する。学習モデルの書き換えやfine-tuningは行わない。
original = nn.Conv2d(1, 2, kernel_size=2, bias=True, dtype=torch.float64)
original.eval()
source_copy = copy.deepcopy(original)
factorized = factorize_conv2d_svd_example(source_copy, rank=2)
images = torch.arange(9, dtype=torch.float64).reshape(1, 1, 3, 3)

with torch.no_grad():
    # full rankは圧縮率ではなく、向き・reshape・bias・forwardの等価性を確認する。
    torch.testing.assert_close(
        factorized(images),
        original(images),
        rtol=1e-10,
        atol=1e-10,
    )

assert factorized[0].weight.shape == (2, 1, 2, 2)
assert factorized[1].weight.shape == (2, 2, 1, 1)
assert not factorized.training
```

生成後に `.to(dtype=...)` するだけでは、copy先がdefault float32だった場合に
float64因子が一度丸められ、その後float64へ戻しても失われた桁は回復しない。
そのため、層の生成時点で元weightと同じdevice/dtypeを指定する。
この例は通常のConv演算を対象とし、独自forwardを持つsubclassや既存hookの移植は扱わない。
[Conv2dの設定とweight shape](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)。

reshape前後のscalar対応は

$$
(W^{(1)})_{\alpha,c,a,b}
=(F_1)_{\alpha,(cK_h+a)K_w+b},
\qquad
(W^{(2)})_{o,\alpha,0,0}=(F_2)_{o,\alpha}
$$

である。例えば $K_h=K_w=3$ では

$$
(c,a,b)=(17,2,1)
\quad\Longrightarrow\quad
q=(17\times3+2)3+1=160.
$$

また、$(F_2)_{3,5}$ は `second_layer.weight[3,5,0,0]` に対応する。
要素数が同じでも `copy_` は軸を自動的にConvの意味へ読み替えない。
reshapeしない代入は一般のshapeでは失敗し、broadcast可能な偶然のshapeでも
意図した添字対応を保証しない。行列形の因子をそのまま `model.conv2` へ
代入することとも区別する。

### 関数を分割するなら、shapeと責務の境界で分ける

一つの大きい関数と分割した関数は、別の圧縮手法ではなく同じ処理の整理方法である。
学習時は上の一関数でshapeの流れを追い、再利用する段階では次の3責務に分けられる。

1. weightからSVD因子を求める。
2. Conv設定と因子から2層Moduleを作る。
3. 引数を検査し、1と2を呼び出す。

一行のflattenやsliceをすべて関数に分割する必要はない。分割してもrank検査、
`groups` 条件、dtype/device、`padding_mode`、bias、freeze設定を落とさない。
同じ完成コードを複数箇所へ重複掲載せず、この節を処理全体の対応先にする。
Moduleを実際に差し替える場合はコピー先へ登録し、その後でoptimizerを再作成する。

---

---

## 理論ノート中の短い確認コード

## PyTorch確認-001

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]（対応コード）

```python
from nn_compression.compression import (
    factorize_conv2d_layer,
    factorize_named_conv2d,
)
```

## PyTorch確認-002

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]（1層目）

```python
nn.Conv2d(
    C_in,
    r,
    kernel_size=(K_h, K_w),
)
```

## PyTorch確認-003

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]（2層目）

```python
nn.Conv2d(
    r,
    C_out,
    kernel_size=1,
)
```

## PyTorch確認-004

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]（3. stride / padding / dilationの引き継ぎ）

```python
first_layer = nn.Conv2d(
    in_channels=conv.in_channels,
    out_channels=rank,
    kernel_size=conv.kernel_size,
    stride=conv.stride,
    padding=conv.padding,
    dilation=conv.dilation,
    bias=False,
)
```

## PyTorch確認-005

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/17_Conv2dの低ランク2層置換]]（3. stride / padding / dilationの引き継ぎ）

```python
second_layer = nn.Conv2d(
    in_channels=rank,
    out_channels=conv.out_channels,
    kernel_size=1,
    stride=1,
    padding=0,
    bias=...,
)
```

