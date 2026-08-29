# `hooi_sweep`

**Stability:** A  
**定義:** `src/nn_compression/compression/hooi.py`

## 責務

現在のfactor集合を初期値として、`ranks`で指定された各modeのfactorを1回ずつ更新するHOOI 1 sweepを実行する。

## Signature

```python
hooi_sweep(
    X: torch.Tensor,
    factors: dict[int, torch.Tensor],
    ranks: dict[int, int],
) -> dict[int, torch.Tensor]
```

## 引数

- `X`: 元Tensor。2階以上の実数浮動小数点Tensor。
- `factors`: 現在の `{mode: U}`。
- `ranks`: 更新対象modeと、そのfactor rank。

## 戻り値

1 sweep更新後のfactor dict。入力`factors`とは別オブジェクト。

## 使用場面

HOOI反復の1単位をテスト・可視化したいとき、Tucker-2 HOOIをgeneric HOOIへ委譲するとき。

## 処理の流れ（日本語）

1. **HOOI入力全体を検証する。**  
   `X`のndim/dtype、`ranks`、factor key集合、factor shape/dtype/device、各target modeのprojected unfoldingでrankが実現可能かを確認する。
2. **入力factorをcloneする。**  
   呼び出し元の辞書・Tensorを直接上書きせず、更新先`updated_factors`を作る。
3. **`ranks`の挿入順でtarget modeを1つ選ぶ。**  
   Tucker-2で`{0: rank_out, 1: rank_in}`ならmode 0 → mode 1の順。
4. **元Tensor`X`からprojectionを作り始める。**
5. **target以外の圧縮modeを最新factorで射影する。**  
   `U_other.T`を`mode_dot`する。sweep前半ですでに更新されたfactorは、後半modeの更新でその**新しい値**を使う。
6. **projected Tensorをtarget modeでunfoldする。**
7. **unfoldingをtruncated SVDする。**  
   左特異ベクトルの上位rankを、新しいtarget factorとして採用する。
8. **`updated_factors[target_mode]`を更新する。**
9. **全target modeについて4〜8を繰り返す。**
10. **1 sweep後のfactor辞書を返す。**

### Gauss-Seidel型更新

この実装はsweep途中で更新したfactorを後続modeが利用する。すべてを旧factorから同時更新するJacobi型ではない。

### 処理フロー（短縮版）

```text
X / factors / ranks
→ 全contract検証
→ factors clone
→ target modeを順に選択
   → target以外を「最新factor」でprojection
   → target mode unfold
   → truncated_svd
   → target factor更新
→ updated_factors
```

## 主なcontract / 注意事項

- 入力`factors`を破壊しない。
- 更新順は`ranks` Mappingの挿入順。
- factor keys/shape/dtype/deviceだけでなくprojected rank feasibilityも事前検証する。

## 関連API

`hooi`, `tucker2_hooi_sweep`, `core_from_factors`
