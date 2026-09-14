"""Core API v1設計書とPublic API境界の同期を検証する。"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"
CORE_API_ROOT = DOCS_ROOT / "07_src設計" / "05_Core_API_v1"
SRC_ROOT = PROJECT_ROOT / "src"

PUBLIC_PACKAGES = (
    "compression",
    "datasets",
    "metrics",
    "models",
    "selection",
    "tensor",
    "training",
    "utils",
)

SPECIAL_DOCUMENTS = {"README.md", "Compatibility_API.md", "Internal_API.md"}
COMPATIBILITY_EXCEPTIONS = {
    ("compression", "SVD"),
    ("compression", "RebuildSVD"),
    ("compression", "factorize_Conv2d_layer"),
    ("metrics", "factorized_linear_macs"),
    ("metrics", "factorized_conv2d_macs"),
}
METHOD_DOCUMENTS = {("models", "FashionMNISTCNN.inspect_shapes")}

REQUIRED_HEADINGS = (
    "## 責務",
    "## Signature",
    "## 引数",
    "## 戻り値",
    "## 使用場面",
    "## 処理概要",
    "## 主なcontract / 注意事項",
    "## 関連API",
)


def _individual_documents() -> list[Path]:
    """READMEと集約仕様を除く、関数・クラス単位の設計書を返す。"""
    return sorted(
        path
        for path in CORE_API_ROOT.rglob("*.md")
        if path.name not in SPECIAL_DOCUMENTS
    )


def _public_exports() -> set[tuple[str, str]]:
    """subpackageの__all__からPublic API境界を収集する。"""
    exports: set[tuple[str, str]] = set()
    for package_name in PUBLIC_PACKAGES:
        package = importlib.import_module(f"nn_compression.{package_name}")
        exports.update((package_name, name) for name in package.__all__)
    return exports


def _parse_documented_signature(text: str) -> tuple[str, ast.FunctionDef]:
    """MarkdownのSignatureをPython ASTとして解析する。"""
    match = re.search(r"## Signature\s*```python\s*(.*?)```", text, re.DOTALL)
    assert match is not None, "Signature code blockがありません。"

    signature = match.group(1).strip()
    dotted_name = signature.split("(", 1)[0].strip()
    safe_name = dotted_name.replace(".", "_")
    function_source = f"def {safe_name}{signature[len(dotted_name):]}:\n    pass\n"
    node = ast.parse(function_source).body[0]
    assert isinstance(node, ast.FunctionDef)
    return dotted_name, node


def _documented_parameter_contract(
    arguments: ast.arguments,
) -> list[tuple[str, inspect._ParameterKind, object]]:
    """annotation表記に依存せず、引数区分・名前・既定値を抽出する。"""
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(arguments.defaults)
    ) + list(arguments.defaults)

    contract: list[tuple[str, inspect._ParameterKind, object]] = []
    for index, (argument, default_node) in enumerate(zip(positional, defaults)):
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < len(arguments.posonlyargs)
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        default = (
            inspect.Parameter.empty
            if default_node is None
            else ast.literal_eval(default_node)
        )
        contract.append((argument.arg, kind, default))

    if arguments.vararg is not None:
        contract.append(
            (
                arguments.vararg.arg,
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.empty,
            )
        )

    for argument, default_node in zip(arguments.kwonlyargs, arguments.kw_defaults):
        default = (
            inspect.Parameter.empty
            if default_node is None
            else ast.literal_eval(default_node)
        )
        contract.append((argument.arg, inspect.Parameter.KEYWORD_ONLY, default))

    if arguments.kwarg is not None:
        contract.append(
            (
                arguments.kwarg.arg,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.empty,
            )
        )

    return contract


def _resolve_documented_object(source_path: Path, dotted_name: str):
    """定義元moduleから文書化対象の関数・クラス・methodを取得する。"""
    module_path = source_path.relative_to(SRC_ROOT).with_suffix("")
    module = importlib.import_module(".".join(module_path.parts))

    obj = module
    name_parts = dotted_name.split(".")
    for name_part in name_parts:
        obj = getattr(obj, name_part)
    return obj, len(name_parts) > 1


def _runtime_parameter_contract(
    obj,
    *,
    drop_bound_receiver: bool,
) -> list[tuple[str, inspect._ParameterKind, object]]:
    """実装側のinspect.signatureから同じcontractを抽出する。"""
    parameters = list(inspect.signature(obj).parameters.values())
    if (
        drop_bound_receiver
        and parameters
        and parameters[0].name in {"self", "cls"}
    ):
        parameters = parameters[1:]
    return [(parameter.name, parameter.kind, parameter.default) for parameter in parameters]


def test_every_public_export_has_a_documented_boundary():
    """Public APIを個別仕様かCompatibility集約のどちらかへ収載する。"""
    documented = {
        (path.parent.name, path.stem) for path in _individual_documents()
    }
    expected_individual = _public_exports() - COMPATIBILITY_EXCEPTIONS

    assert documented - METHOD_DOCUMENTS == expected_individual

    compatibility_text = (CORE_API_ROOT / "Compatibility_API.md").read_text(
        encoding="utf-8"
    )
    for _, name in COMPATIBILITY_EXCEPTIONS:
        assert f"`{name}`" in compatibility_text


def test_individual_documents_match_source_contracts():
    """必須章、定義元、引数contractと引数説明を実装へ同期させる。"""
    for document_path in _individual_documents():
        text = document_path.read_text(encoding="utf-8")
        for heading in REQUIRED_HEADINGS:
            assert heading in text, f"{document_path}: {heading}がありません。"

        definition = re.search(r"\*\*定義:\*\* `([^`]+)`", text)
        assert definition is not None, f"{document_path}: 定義元がありません。"
        source_path = PROJECT_ROOT / definition.group(1)
        assert source_path.is_file(), f"{document_path}: {source_path}が存在しません。"

        dotted_name, signature_node = _parse_documented_signature(text)
        obj, is_method = _resolve_documented_object(source_path, dotted_name)
        documented_contract = _documented_parameter_contract(signature_node.args)
        runtime_contract = _runtime_parameter_contract(
            obj,
            drop_bound_receiver=is_method,
        )
        assert documented_contract == runtime_contract, (
            f"{document_path}: 引数contractが実装と一致しません。\n"
            f"docs={documented_contract}\n"
            f"src={runtime_contract}"
        )

        argument_section = text.split("## 引数", 1)[1].split("\n## ", 1)[0]
        for name, _, _ in documented_contract:
            assert f"`{name}`" in argument_section, (
                f"{document_path}: 引数{name}の説明がありません。"
            )


def test_visible_internal_helpers_are_listed_in_internal_boundary():
    """非公開でもPublic APIに見える名前のhelperを境界仕様へ収載する。"""
    public_exports = _public_exports()
    visible_internal_helpers: list[str] = []

    for package_name in PUBLIC_PACKAGES:
        package_root = SRC_ROOT / "nn_compression" / package_name
        for source_path in sorted(package_root.glob("*.py")):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    continue
                if node.name.startswith("_"):
                    continue
                if (package_name, node.name) in public_exports:
                    continue
                visible_internal_helpers.append(node.name)

    internal_text = (CORE_API_ROOT / "Internal_API.md").read_text(encoding="utf-8")
    for helper_name, definition_count in Counter(visible_internal_helpers).items():
        assert internal_text.count(f"`{helper_name}`") >= definition_count, (
            f"Internal_API.mdに{helper_name}の定義元ごとの説明がありません。"
        )


def test_core_api_markdown_links_and_blocks_are_balanced():
    """Core API v1内のWikiリンクとMarkdown blockの破損を検出する。"""
    for document_path in sorted(CORE_API_ROOT.rglob("*.md")):
        text = document_path.read_text(encoding="utf-8")
        assert text.count("```") % 2 == 0, (
            f"{document_path}: code block delimiterが対応していません。"
        )
        assert text.count("$$") % 2 == 0, (
            f"{document_path}: math block delimiterが対応していません。"
        )

        link_targets = re.findall(r"\[\[([^\]|#]+)", text)
        for link_target in link_targets:
            target_path = DOCS_ROOT / f"{link_target}.md"
            assert target_path.is_file(), (
                f"{document_path}: Wikiリンク{link_target}を解決できません。"
            )
