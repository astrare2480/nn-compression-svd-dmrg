---
title: Core API v1
aliases:
  - Public API freeze
  - API仕様
---

# Core API v1

Public APIの詳細仕様は、関数・クラス単位のファイルへ分割した。

入口：[[07_src設計/05_Core_API_v1/README]]

各APIファイルには原則として次を記載する。

```text
責務
Signature
引数
戻り値
使用場面
処理概要
フローチャート / 構造図（必要な場合のみ）
主なcontract / 注意事項
関連API
```

`処理概要`では、英語identifierだけの短い矢印列にせず、**現行srcが何をどの順番で行い、なぜその処理が必要かを日本語で追える粒度**にする。

Mermaid図は全関数へ機械的に付けない。反復・分岐・複数APIをまたぐ状態遷移・Module構造変換など、文章だけより図の方が理解しやすいAPIに限定する。

Public APIの基本境界は各sub-packageの `__all__`。private helperはCore API v1の固定対象にしない。

Compatibility aliasは [[07_src設計/05_Core_API_v1/Compatibility_API]] にまとめる。
