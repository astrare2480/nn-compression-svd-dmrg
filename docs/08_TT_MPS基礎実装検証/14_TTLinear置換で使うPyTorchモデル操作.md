---
title: TTLinear置換で使うPyTorchモデル操作
tags:
  - PyTorch
  - state-dict
  - eval
  - no-grad
  - module-replacement
  - TTLinear
---

# TTLinear置換で使うPyTorchモデル操作

## 1. このノートの目的

TTLinearの数学とは別に、学習済みPyTorchモデルの一層を安全に置換し、同じ条件で比較するために必要な操作を整理する。

## 2. state_dictとは

<code>state_dict()</code> は、moduleに登録されたparameterとbufferを名前からtensorへ対応させる辞書である。

~~~text
fc1.weight
fc1.bias
fc2.weight
fc2.bias
...
~~~

<code>state_dict</code> は値を保存するが、module class、forwardの演算順序、activationなどのネットワーク構造そのものは保存しない。

従ってstate dict形式のcheckpointを読むときは、先に同じ構造のinstanceを作る。

~~~python
# まずネットワーク構造を作る
model = MNISTMLP().to(device=device, dtype=dtype)

# checkpointファイルからparameter / bufferの辞書を読む
state_dict = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)

# 同じ名前とshapeを持つ場所へ値を入れる
model.load_state_dict(state_dict, strict=True)
model.eval()
~~~

<code>load_state_dict()</code> にファイルpathを直接渡すのではなく、<code>torch.load()</code> が返した辞書を渡す。

モデルobject全体を

~~~python
torch.save(model, checkpoint_path)
~~~

として保存した場合は、読込時に別途class instanceを作らない形もある。しかしPython classのmodule pathや実行環境へ依存しやすい。再利用性と移植性のため、通常はmodel構造をコードで定義し、<code>state_dict</code> を保存する。

## 3. strict=Trueとstrict=False

既定の <code>strict=True</code> はkeyの不足と余剰を拒否する。等価性検証ではコピー漏れを隠さないため、原則としてこちらを使う。

<code>strict=False</code> はmissing keyとunexpected keyを許容し、返り値から確認できる。

~~~python
result = target.load_state_dict(source_state, strict=False)
print(result.missing_keys)
print(result.unexpected_keys)
~~~

ただし <code>strict=False</code> は、同名tensorのshape不一致まで自動変換する機能ではない。

## 4. dense Linearのstate dictをTTLinearへ直接読めない

dense Linearは通常、

~~~text
weight: (out_features, in_features)
bias:   (out_features,)
~~~

を持つ。一方、TTLinearは

~~~text
cores.0: (1, m1, n1, r1)
cores.1: (r1, m2, n2, r2)
...
bias:    (out_features,)
~~~

のようにparameter構造が異なる。

そのため、

~~~python
# keyもshapeも異なるため、dense -> TTの変換にはならない
tt_linear.load_state_dict(dense_linear.state_dict())
~~~

とはできない。dense weightはtensorizationとTT-SVDでcore列へ変換し、biasだけを引き継ぐ。

## 5. evalとno_gradは別の操作

<code>model.eval()</code> はmoduleを評価modeへ切り替える。

- Dropoutを無効化する
- BatchNormにrunning statisticsを使わせる
- autograd自体は無効化しない

<code>torch.no_grad()</code> は、そのblock内の演算についてautograd graphを記録しない。

- memory使用量を抑える
- 出力の <code>requires_grad</code> を通常falseにする
- DropoutやBatchNormのmodeは変更しない

従って評価・等価性検証では両方を使う。

~~~python
dense_model.eval()
tt_model.eval()

with torch.no_grad():
    logits_dense = dense_model(x_batch)
    logits_tt = tt_model(x_batch)
~~~

trainingと <code>backward()</code> を行う処理を <code>no_grad()</code> 内へ入れてはいけない。

## 6. detachとの違い

<code>tensor.detach()</code> は、そのtensorだけを現在のautograd graphから切り離したviewとして扱う。一方、<code>torch.no_grad()</code> はblock内で新たに行う演算全体のgraph記録を止める。

初期化用に学習済み重みを読むだけなら、

~~~python
weight = dense_linear.weight.detach()
~~~

とできる。モデル評価全体なら <code>no_grad()</code> の方が意図を表しやすい。

## 7. named_modulesで置換候補を探す

<code>named_modules()</code> はmodel自身と、再帰的に含まれる全submoduleを

$$
\text{module path}
\longmapsto
\text{module object}
$$

として列挙する。

~~~python
for path, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(
            path,
            tuple(module.weight.shape),
            module.in_features,
            module.out_features,
        )
~~~

先頭にはpathが空文字のmodel自身も含まれる。<code>named_parameters()</code> はparameter、<code>named_buffers()</code> はbufferを列挙するAPIであり、module構造を探す <code>named_modules()</code> とは役割が異なる。

## 8. getattrでpathを辿る

pathが <code>encoder.block1.linear</code> なら、

~~~python
parent = model
for part in ["encoder", "block1"]:
    parent = getattr(parent, part)
~~~

の後、<code>parent</code> は <code>model.encoder.block1</code> を指す。

最後の <code>linear</code> まで <code>getattr</code> しない理由は、置換には対象module自身ではなく、それを所有する親moduleと属性名が必要だからである。

## 9. setattrでmoduleを置換する

~~~python
setattr(parent, "linear", new_module)
~~~

は、

~~~python
parent.linear = new_module
~~~

と同じ意味である。<code>setattr</code> の戻り値は <code>None</code> だが、parent object自体が変更される。

汎用的なpath置換は次のように書ける。

~~~python
def set_module_by_path(
    model: nn.Module,
    path: str,
    new_module: nn.Module,
) -> None:
    """ドット区切りpathで指定したsubmoduleを置換する。"""
    parts = path.split(".")
    if not path or any(part == "" for part in parts):
        raise ValueError(f"invalid module path: {path!r}")

    parent: nn.Module = model

    # 最後の子名を残し、その親まで降りる
    for part in parts[:-1]:
        parent = getattr(parent, part)

    child_name = parts[-1]
    old_module = getattr(parent, child_name)

    # 置換前後で入出力shapeが互換かは呼び出し側でも確認する
    setattr(parent, child_name, new_module)

    if getattr(parent, child_name) is not new_module:
        raise RuntimeError(
            f"failed to replace {path!r}: old={type(old_module).__name__}"
        )
~~~

<code>nn.Sequential</code> の子名は文字列の <code>"0"</code>、<code>"1"</code> などとして登録されるため、同じ <code>getattr</code> / <code>setattr</code> の考え方で扱える。

## 10. deepcopyして比較基準を守る

置換はmodelをin-placeに変更する。元モデルを比較基準として残すには、

~~~python
import copy

tt_model = copy.deepcopy(dense_model)
set_module_by_path(tt_model, target_path, tt_layer)
~~~

とする。

置換後は少なくとも次を検査する。

~~~python
dense_modules = dict(dense_model.named_modules())
tt_modules = dict(tt_model.named_modules())

assert isinstance(dense_modules[target_path], nn.Linear)
assert isinstance(tt_modules[target_path], TTLinear)

for path, dense_module in dense_modules.items():
    if path in ("", target_path):
        continue
    assert type(tt_modules[path]) is type(dense_module)
~~~

## 11. Parameterとbuffer

学習するTT coreは通常 <code>nn.Parameter</code> として登録する。

~~~python
self.cores = nn.ParameterList(
    [nn.Parameter(core.clone()) for core in cores]
)
~~~

学習しない参照値ならbufferとして登録する選択肢がある。ただし複数coreをbufferとして持つ場合、各coreへ固有名を付けて登録する設計が必要である。

~~~python
for index, core in enumerate(cores):
    self.register_buffer(f"core_{index}", core.detach().clone())
~~~

<code>requires_grad=False</code> のParameterはbufferと同じではない。optimizerへ渡すparameter集合、<code>named_parameters()</code>、設計上の意味が異なる。

## 12. argmaxのdim

logitsが

$$
L\in\mathbb{R}^{B\times C}
$$

なら、各sampleの予測classは

~~~python
prediction = logits.argmax(dim=1)
~~~

でshape $(B,)$ になる。

<code>dim=0</code> はclassごとに、batch中のどのsampleが最大かを返すため、分類予測ではない。

softmaxは単調なので、

$$
\operatorname*{arg\,max}_k L_{b,k}
=
\operatorname*{arg\,max}_k
\operatorname{softmax}(L_b)_k.
$$

prediction比較だけならlogitsから直接argmaxしてよい。

## 13. 比較時の安全な順序

~~~text
同じmodel classを生成
→ state_dictをstrictにロード
→ eval modeへ切替
→ Linear層とshapeを列挙
→ dense重みをTT coreへ変換
→ deepcopyしたmodelの対象層だけを置換
→ 同じinput tensorを両モデルへ渡す
→ no_grad内で中間出力とlogitsを比較
→ argmaxを補助確認
~~~

この順序により、checkpoint読込、module置換、mode順序、direct contraction、rank近似誤差を混同しにくくなる。
