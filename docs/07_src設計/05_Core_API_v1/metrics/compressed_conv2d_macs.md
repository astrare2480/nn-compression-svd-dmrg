# `compressed_conv2d_macs`

**Stability:** A  
**定義:** `src/nn_compression/metrics/macs.py`

## 責務

`factorize_conv2d_layer()`が作るSVD 2層Convと同じ構造について理論MACsを計算する。

## Signature

```python
compressed_conv2d_macs(
    conv,
    rank: int,
    out_h: int,
    out_w: int,
) -> int
```

## 引数

- `conv`: 圧縮前の元Conv2d。`C_in / C_out / kH / kW`を取り出し、SVD 2層化後MACsの基準形状として使う。
- `rank`: 2層Conv間の中間channel数。`factorize_conv2d_layer()`で使うSVD rankと同じ意味。
- `out_h`: 対象Convが生成する出力feature mapの高さ。
- `out_w`: 対象Convが生成する出力feature mapの幅。

## 戻り値

SVD 2層化後の**1 output batch/sampleあたりの理論MAC（multiply-accumulate）総数**を表す整数。

```text
out_h*out_w*rank*in_ch*kH*kW
+
out_h*out_w*out_ch*rank
```

前半が元kernelを持つ`C_in → rank` Conv、後半が`rank → C_out`の1x1 Convの計算量。

## 使用場面

Conv SVD candidateの理論計算量比較。

## 処理概要

1. **元Convが`groups=1`か確認する。**  
   圧縮実装自体が通常Convだけを正式対応としているため、MACsだけgrouped Convの仮想値を返さない。
2. **元weight shapeからSVD最大rankを求め、rankを検証する。**
3. **`out_h/out_w`を正整数として検証する。**
4. **1層目の空間Conv MACsを計算する。**  
   `C_in → rank`、元kernelを使うので`out_h*out_w*rank*in_ch*kH*kW`。
5. **2層目の1x1 Conv MACsを計算する。**  
   `rank → C_out`なので`out_h*out_w*out_ch*rank`。
6. **2層分を加算して返す。**

## 主なcontract / 注意事項

- compressed Convは`groups=1`のみ。
- 実際に生成できないrankに対して「もっともらしいMACs値」だけ返さない。

## 関連API

`conv2d_macs`, `estimate_conv2d_macs`, `factorize_conv2d_layer`
