# `tucker2_hooi_sweep`

**Stability:** A  
**定義:** `src/nn_compression/compression/conv_tucker.py`

## 責務

Conv2d weightのchannel mode 0/1に限定したpartial HOOIを1 sweepだけ実行するwrapper。

## Signature

```python
tucker2_hooi_sweep(
    weight: torch.Tensor,
    factors: dict[int, torch.Tensor],
    rank_out: int,
    rank_in: int,
) -> dict[int, torch.Tensor]
```

## 引数

- `weight`: factorを1 sweep更新する対象のConv2d weight。shapeは`(C_out, C_in, kH, kW)`。
- `factors`: sweep開始時点のchannel factor。`{0: U_out, 1: U_in}`で、各factorは現在の近似基底を表す。
- `rank_out`: mode 0で更新後に保持する出力channel rank。
- `rank_in`: mode 1で更新後に保持する入力channel rank。

## 戻り値

mode 0→1の1 sweepを終えた新しいfactor dict`{0: U_out_new, 1: U_in_new}`。入力`factors`をそのまま破壊して返すのではなく、generic `hooi_sweep()`の更新結果を返す。

## 使用場面

Tucker-2 HOOIの更新過程を1 sweep単位で確認したいとき、学習Notebookでgeneric HOOIとの対応を追うとき。

## 処理概要

1. **weight shapeとchannel rankをTucker-2規約で検証する。**
2. **generic HOOI用rank Mappingを作る。**  
   `{0: rank_out, 1: rank_in}`とし、kernel mode 2/3は更新対象にしない。
3. **`hooi_sweep(weight, factors, ranks)`へ処理を委譲する。**
4. **generic HOOIがmode 0 → mode 1の順でfactorを更新する。**  
   mode 1更新時には同sweepで更新済みの`U_out`を使う。
5. **更新後factor dictをそのまま返す。**

## 主なcontract / 注意事項

Tucker-2固有の新しいsweepアルゴリズムを持たず、generic HOOIのsemanticsをそのまま再利用する。

## 関連API

`hooi_sweep`, `tucker2_hooi`
