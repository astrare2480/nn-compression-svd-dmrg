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

なし。architectureは`784 → 512 → 256 → 10`の固定baselineとして定義する。

## 戻り値

28×28画像をflattenして10クラスlogitsへ変換する`nn.Module` instance。主要named layerは`fc1 / fc2 / fc3`で、Linear SVD APIがこれらのpathを安定した圧縮対象として参照する。

## 使用場面

MNIST / Fashion-MNIST MLP baseline、Linear SVD、rank sweep、圧縮後Fine-tuning。

## 処理概要

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

### 構造図

```mermaid
flowchart LR
    X["画像入力<br/>(N,1,28,28)"] --> F["Flatten<br/>784"]
    F --> FC1["fc1<br/>784 → 512"]
    FC1 --> R1["ReLU"]
    R1 --> FC2["fc2<br/>512 → 256"]
    FC2 --> R2["ReLU"]
    R2 --> FC3["fc3<br/>256 → 10"]
    FC3 --> Y["class logits<br/>(N,10)"]
```

この図は制御フローではなく、**baseline MLPのlayer構造とfeature dimensionの変化**を示す。

## 主なcontract / 注意事項

Core API v1では、既存checkpoint・Notebook・named compressionが依存する主要layer名を安定化する。

```text
fc1, relu1, fc2, relu2, fc3
```

## 関連API

`make_one_layer_svd_model`, `make_two_layer_svd_model`, `estimate_mlp_macs`
