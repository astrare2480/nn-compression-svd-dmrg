# Compatibility API

**Stability:** C

historical Notebook互換のため残している名前。新規コードではprimary APIを使う。

| Compatibility | Primary | 備考 |
|---|---|---|
| `SVD` | `truncated_svd` | 単純alias |
| `RebuildSVD` | `rebuild_linear_from_svd` | 単純alias |
| `factorize_Conv2d_layer` | `factorize_conv2d_layer` | 単純alias |
| `factorized_linear_macs` | `compressed_linear_macs` | 単純alias |
| `factorized_conv2d_macs` | `compressed_conv2d_macs` | 単純alias |
| `sweep_conv_svd_ranks()` | `sweep_conv2d_ranks()` | `rank_list`名を維持するwrapper |
| `r1/r2` keyword | `fc1_rank/fc2_rank` | MLP historical Notebook向け |

## 使用方針

- existing Notebookの再現では利用可。
- 新しいNotebook/src実装ではprimary名を使用する。
- Compatibility APIへ新機能は追加しない。
- Core API v1中は既存互換を不用意に削除しない。
