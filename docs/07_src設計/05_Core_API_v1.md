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
処理の流れ（日本語）
処理フロー（短縮版。必要な場合）
主なcontract / 注意事項
関連API
```

処理説明は英語identifierだけの短い矢印列にせず、**現行srcが何をどの順番で行い、なぜその処理が必要かを日本語で追える粒度**にする。

Public APIの基本境界は各sub-packageの `__all__`。private helperはCore API v1の固定対象にしない。

Compatibility aliasは [[07_src設計/05_Core_API_v1/Compatibility_API]] にまとめる。
