# Tucker-2 Conv2dのPyTorch契約

数式と考え方は [[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]] を参照する。このノートはPyTorch・Pythonの操作、コード例、実装上の注意を扱う。教材のコード例と現行の共通関数は区別し、公開APIの仕様は [[07_src設計/05_Core_API_v1]] を参照する。

---

## 10. PyTorch実装上の契約

今回のsrcでは、元Convの意味を保つため次を維持する。

- `groups=1` の通常Conv2dのみ対応
- 中央core Convが元の `stride / padding / dilation / padding_mode` を継承
- 前後1x1 Convはstride 1 / padding 0 / dilation 1
- device / dtypeを維持
- 元weight / biasの `requires_grad` を維持
- 元biasは最後の1x1へコピー
- 元Conv自体を破壊せず、置換後Parameterとstorageを共有しない
- `rank_out / rank_in` はboolを拒否し、`1 <= rank_out <= C_out`, `1 <= rank_in <= C_in`
- 現行HOSVD/HOOI経路は実数dtype限定で、複素Tensorは明示的に拒否

重みコピーは学習演算ではなく初期化なので、

```python
with torch.no_grad():
    parameter.copy_(value)
```

を使う。

`no_grad()` はコピー操作の履歴を記録しないだけで、コピー後のParameterを学習不能にするものではない。

### autograd境界はAPI層で分ける

すべてのTucker分解で入力weightを一律 `detach()` するわけではない。

```text
tucker2_hooi(weight, ...)
→ Tensor-level decomposition
→ 入力weightをdetachしない

build_tucker2_conv(conv, ...)
→ Module構築・初期化
→ conv.weightをdetachして分解し、新しいleaf Parameterへcopy
```

この分離により、低レベルTensor APIでは必要以上にautograd graphを切らず、Module置換時には元モデルと新しいParameterを独立させる。

`tucker2_effective_weight()` は評価用helperなので、3層のweightをdetachして等価な4階weightを再構成する。

---

---

## 理論ノート中の短い確認コード

## PyTorch確認-001

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]（5. `[:, :, None, None]` の意味）

```python
u_in.T[:, :, None, None]
```

## PyTorch確認-002

対応する数式・説明：[[00_基礎理論/03_モデル圧縮理論/24_Conv2dのTucker2圧縮]]（5. `[:, :, None, None]` の意味）

```python
u_out[:, :, None, None]
```

