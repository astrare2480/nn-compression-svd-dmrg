# `MNISTMLP`

**Stability:** B  
**定義:** `src/nn_compression/models/mlp.py`

## 責務

MNIST / Fashion-MNISTのMLP SVD実験で共通利用するbaseline architectureを提供する。

## Signature

```python
MNISTMLP()
```

## 引数

なし。

## 戻り値

`nn.Module` instance。

## 使用場面

MNIST / Fashion-MNIST MLP baseline、Linear SVD、rank sweep、圧縮後Fine-tuning。

## 処理の流れ（日本語）

### 初期化時

1. **1層目`fc1`を作る。**  
   28×28画像をflattenした784特徴を512特徴へ変換する`Linear(784, 512)`を定義する。
2. **1層目の活性化`relu1`を作る。**
3. **2層目`fc2`を作る。**  
   `Linear(512, 256)`で中間特徴をさらに圧縮・変換する。
4. **2層目の活性化`relu2`を作る。**
5. **分類層`fc3`を作る。**  
   `Linear(256, 10)`で10クラスのlogitsへ変換する。

### forward時

1. **入力画像をbatch dimensionを残してflattenする。**  
   `x.view(x.size(0), -1)`により`(N,1,28,28)`等を`(N,784)`へする。
2. **`fc1 → relu1`を通す。**
3. **`fc2 → relu2`を通す。**
4. **`fc3`へ通して10クラスlogitsを作る。**
5. **softmaxはかけずlogitsを返す。**  
   CrossEntropyLoss等がlogitsを直接受け取る前提。

### 処理フロー（短縮版）

```text
画像batch
→ flatten 784
→ fc1: 784→512
→ ReLU
→ fc2: 512→256
→ ReLU
→ fc3: 256→10
→ logits
```

## 主なcontract / 注意事項

Core API v1では、既存checkpoint・Notebook・named compressionが依存する主要layer名を安定化する。

```text
fc1, relu1, fc2, relu2, fc3
```

## 関連API

`make_one_layer_svd_model`, `make_two_layer_svd_model`, `estimate_mlp_macs`
