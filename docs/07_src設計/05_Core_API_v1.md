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
ざっくりした処理
主なcontract / 注意事項
関連API
```

Public APIの基本境界は各sub-packageの `__all__`。private helperはCore API v1の固定対象にしない。

Compatibility aliasは [[07_src設計/05_Core_API_v1/Compatibility_API]] にまとめる。
