"""Stable family recipe identities from explicit numerical dependencies."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .registry import FamilySpec


class RecipeDependencyError(ValueError):
    """A declared recipe dependency is missing, ambiguous, or incomplete."""


@dataclass(frozen=True)
class RecipeIdentity:
    family_id: str
    family_version: str
    recipe_id: str
    material_json: str
    provenance: Mapping[str, str]


@dataclass(frozen=True)
class _ModuleIndex:
    module: str
    path: Path
    tree: ast.Module
    definitions: Mapping[str, ast.AST]
    constants: Mapping[str, ast.AST]
    imported_symbols: Mapping[str, str]
    imported_modules: Mapping[str, str]


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise RecipeDependencyError("Nonfinite recipe material is forbidden")
        return value
    raise RecipeDependencyError(f"Unsupported recipe material type: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _normalize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RecipeDependencyError("Recipe material is not canonical JSON") from exc


def _module_path(code_root: Path, module: str) -> Path:
    relative = Path(*module.split("."))
    file_path = code_root / relative.with_suffix(".py")
    package_path = code_root / relative / "__init__.py"
    if file_path.is_file():
        return file_path
    if package_path.is_file():
        return package_path
    raise RecipeDependencyError(f"Cannot resolve module: {module}")


def _index_module(code_root: Path, module: str, cache: dict[str, _ModuleIndex]) -> _ModuleIndex:
    if module in cache:
        return cache[module]
    path = _module_path(code_root, module)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise RecipeDependencyError(f"Cannot parse module: {module}") from exc

    definitions: dict[str, ast.AST] = {}
    constants: dict[str, ast.AST] = {}
    imported_symbols: dict[str, str] = {}
    imported_modules: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            constants[node.target.id] = node.value
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = module if path.name == "__init__.py" else module.rpartition(".")[0]
                try:
                    imported_from = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), package
                    )
                except (ImportError, ValueError):
                    continue
            else:
                imported_from = node.module
            if not imported_from:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                target = f"{imported_from}.{alias.name}"
                if node.module is None:
                    imported_modules[local_name] = target
                else:
                    imported_symbols[local_name] = target
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules[alias.asname or alias.name.split(".")[0]] = alias.name

    index = _ModuleIndex(
        module=module,
        path=path,
        tree=tree,
        definitions=definitions,
        constants=constants,
        imported_symbols=imported_symbols,
        imported_modules=imported_modules,
    )
    cache[module] = index
    return index


def _split_symbol(symbol: str) -> tuple[str, str]:
    if "." not in symbol:
        raise RecipeDependencyError(f"Symbol must be module-qualified: {symbol}")
    return symbol.rsplit(".", 1)


def _resolve_symbol(
    code_root: Path,
    symbol: str,
    cache: dict[str, _ModuleIndex],
) -> tuple[_ModuleIndex, ast.AST, str]:
    module, name = _split_symbol(symbol)
    index = _index_module(code_root, module, cache)
    if name in index.definitions:
        return index, index.definitions[name], "definition"
    if name in index.constants:
        return index, index.constants[name], "constant"
    raise RecipeDependencyError(f"Cannot resolve symbol: {symbol}")


def _without_docstring(node: ast.AST) -> ast.AST:
    normalized = copy.deepcopy(node)
    body = getattr(normalized, "body", None)
    if (
        isinstance(body, list)
        and body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        normalized.body = body[1:]
    return normalized


def _definition_fingerprint(node: ast.AST) -> str:
    normalized = _without_docstring(node)
    material = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _constant_value(node: ast.AST, symbol: str) -> Any:
    try:
        return _normalize(ast.literal_eval(node))
    except (ValueError, TypeError) as exc:
        raise RecipeDependencyError(f"Constant is not a literal: {symbol}") from exc


def _internal_kind(
    code_root: Path,
    symbol: str,
    cache: dict[str, _ModuleIndex],
) -> str | None:
    try:
        _, _, kind = _resolve_symbol(code_root, symbol, cache)
    except RecipeDependencyError:
        return None
    return kind


def _qualified_reference(index: _ModuleIndex, node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        if node.id in index.definitions or node.id in index.constants:
            return f"{index.module}.{node.id}"
        return index.imported_symbols.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        module = index.imported_modules.get(node.value.id)
        if module:
            return f"{module}.{node.attr}"
    return None


def _audit_definition_dependencies(
    code_root: Path,
    index: _ModuleIndex,
    node: ast.AST,
    declared_symbols: set[str],
    declared_constants: set[str],
    cache: dict[str, _ModuleIndex],
) -> None:
    called_nodes = {id(call.func) for call in ast.walk(node) if isinstance(call, ast.Call)}
    checked: set[tuple[str, str]] = set()
    for child in ast.walk(node):
        target = _qualified_reference(index, child)
        if target is None:
            continue
        kind = _internal_kind(code_root, target, cache)
        if kind is None or (target, kind) in checked:
            continue
        checked.add((target, kind))
        if kind == "definition" and id(child) in called_nodes and target not in declared_symbols:
            raise RecipeDependencyError(f"Undeclared numerical helper: {target}")
        if kind == "constant" and target not in declared_constants:
            raise RecipeDependencyError(f"Missing constant fingerprint: {target}")


def _git_head(code_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"
    return completed.stdout.strip() or "UNAVAILABLE"


def build_recipe_identity(
    spec: FamilySpec,
    config: Mapping[str, object],
    code_root: Path,
) -> RecipeIdentity:
    root = Path(code_root).resolve(strict=True)
    if spec.extractor_symbol not in spec.numerical_symbols:
        raise RecipeDependencyError("Family extractor must be a declared numerical symbol")
    if len(set(spec.numerical_symbols)) != len(spec.numerical_symbols):
        raise RecipeDependencyError("Duplicate numerical symbol declaration")
    if len(set(spec.numerical_constants)) != len(spec.numerical_constants):
        raise RecipeDependencyError("Duplicate numerical constant declaration")

    cache: dict[str, _ModuleIndex] = {}
    symbol_fingerprints: dict[str, str] = {}
    declared_symbols = set(spec.numerical_symbols)
    declared_constants = set(spec.numerical_constants)
    for symbol in sorted(declared_symbols):
        index, node, kind = _resolve_symbol(root, symbol, cache)
        if kind != "definition":
            raise RecipeDependencyError(f"Numerical symbol is not callable code: {symbol}")
        symbol_fingerprints[symbol] = _definition_fingerprint(node)
        _audit_definition_dependencies(
            root,
            index,
            node,
            declared_symbols,
            declared_constants,
            cache,
        )

    constant_fingerprints: dict[str, dict[str, Any]] = {}
    for symbol in sorted(declared_constants):
        _, node, kind = _resolve_symbol(root, symbol, cache)
        if kind != "constant":
            raise RecipeDependencyError(f"Numerical constant is not a constant: {symbol}")
        value = _constant_value(node, symbol)
        constant_fingerprints[symbol] = {
            "value": value,
            "sha256": hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest(),
        }

    parameters = dict(spec.default_config)
    parameters.update(config)
    runtime_dependencies = dict(spec.numerical_dependencies)
    if "numpy" in runtime_dependencies:
        runtime_dependencies["numpy"] = np.__version__
    material = {
        "family_id": spec.family_id,
        "family_version": spec.family_version,
        "extractor_symbol": spec.extractor_symbol,
        "processing_parameters": parameters,
        "input_contract": spec.input_contract.material(),
        "feature_definitions": [item.material() for item in spec.feature_definitions],
        "numerical_symbols": symbol_fingerprints,
        "numerical_constants": constant_fingerprints,
        "numerical_dependencies": runtime_dependencies,
    }
    material_json = _canonical_json(material)
    return RecipeIdentity(
        family_id=spec.family_id,
        family_version=spec.family_version,
        recipe_id=hashlib.sha256(material_json.encode("utf-8")).hexdigest(),
        material_json=material_json,
        provenance={"git_commit": _git_head(root)},
    )
