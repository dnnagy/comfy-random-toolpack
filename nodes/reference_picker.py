"""Versioned reference selection; the resolver/CLI needs only the standard library.

ComfyUI, Pillow, NumPy and torch are imported only when needed by the node.
See docs/reference-picker.md for the manifest and job integration contract.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MAX_SELECTION_SEED = 2**53 - 1  # Exact integers in browser/JSON clients, too.


class ReferenceCatalogError(ValueError):
    """An invalid catalog, selection, or asset, suitable for a node error."""


def _text(value, label, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ReferenceCatalogError(f"{label} must be a {'non-empty ' if not allow_empty else ''}string.")
    return value


def _relative_path(base: Path, value: str) -> Path:
    _text(value, "Asset path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value or ":" in value:
        raise ReferenceCatalogError(f"Asset path must be relative and stay inside the catalog: {value!r}")
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ReferenceCatalogError(f"Asset path escapes the catalog: {value!r}")
    return path


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReferenceCatalogError(f"Duplicate manifest key: {key!r}")
        result[key] = value
    return result


def _read_catalog(catalog, catalog_path):
    _text(catalog, "catalog")
    parts = catalog.split("@")
    if len(parts) != 2 or not all(_TOKEN.fullmatch(part) for part in parts):
        raise ReferenceCatalogError("catalog must be name@version, for example portraits@1.0.0.")
    name, version = parts
    _text(catalog_path, "catalog_path")
    directory = Path(catalog_path).expanduser()
    if not directory.is_absolute():
        raise ReferenceCatalogError("catalog_path must be an absolute directory path.")
    manifest = directory.resolve() / "manifest.json"
    try:
        raw = manifest.read_bytes()
        data = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReferenceCatalogError(f"Cannot read catalog {catalog!r} at {manifest}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReferenceCatalogError("Manifest must be a JSON object.")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ReferenceCatalogError("Manifest schema_version must be 1.")
    if data.get("name") != name or data.get("version") != version:
        raise ReferenceCatalogError(f"Manifest name/version must match {catalog!r}.")
    assets = data.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ReferenceCatalogError("Manifest assets must be a non-empty array.")
    seen = set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise ReferenceCatalogError("Each asset must be a JSON object.")
        asset_id = _text(asset.get("id"), "Asset id")
        if asset_id in seen:
            raise ReferenceCatalogError(f"Duplicate asset id: {asset_id!r}")
        seen.add(asset_id)
        _relative_path(manifest.parent, asset.get("path"))
        _text(asset.get("description"), f"Description for {asset_id!r}")
        attributes = asset.get("attributes")
        if not isinstance(attributes, dict) or not attributes:
            raise ReferenceCatalogError(f"Attributes for {asset_id!r} must be a non-empty object.")
        for category, values in attributes.items():
            _text(category, "Attribute category")
            if isinstance(values, str):
                _text(values, "Attribute value")
            elif isinstance(values, list) and values:
                for value in values:
                    _text(value, "Attribute value")
            else:
                raise ReferenceCatalogError(f"Attribute {category!r} must be a string or non-empty string array.")
        if "sha256" in asset and (
            not isinstance(asset["sha256"], str) or not _SHA256.fullmatch(asset["sha256"])
        ):
            raise ReferenceCatalogError(f"sha256 for {asset_id!r} must be 64 lowercase hex characters.")
    return manifest, data, hashlib.sha256(raw).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_reference(catalog_path, catalog, category, value, selection_seed, asset_id=""):
    """Return JSON-serializable selection metadata, without loading image tensors.

    Persist ``catalog`` and ``asset_id`` in the job and pass them to the picker
    at execution. ``path`` is absolute on this host; ``relative_path`` is portable.
    Exact IDs override category/value filters and the seed (including empty filters).
    """
    _text(category, "category", allow_empty=True)
    _text(value, "value", allow_empty=True)
    _text(asset_id, "asset_id", allow_empty=True)
    if type(selection_seed) is not int or not 0 <= selection_seed <= MAX_SELECTION_SEED:
        raise ReferenceCatalogError(f"selection_seed must be an integer from 0 to {MAX_SELECTION_SEED}.")
    manifest, data, manifest_hash = _read_catalog(catalog, catalog_path)
    if asset_id:
        selected = next((asset for asset in data["assets"] if asset["id"] == asset_id), None)
        if selected is None:
            raise ReferenceCatalogError(f"Asset {asset_id!r} does not exist in {catalog!r}.")
    else:
        _text(category, "category")
        _text(value, "value")
        candidates = []
        for asset in data["assets"]:
            values = asset["attributes"].get(category, [])
            if value == values or (isinstance(values, list) and value in values):
                candidates.append(asset)
        if not candidates:
            raise ReferenceCatalogError(f"No references match {category}={value!r} in {catalog!r}.")
        candidates.sort(key=lambda asset: asset["id"])
        key = json.dumps(
            ["crtp-reference-v1", catalog, category, value, selection_seed],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        index = int.from_bytes(hashlib.sha256(key).digest(), "big") % len(candidates)
        selected = candidates[index]
    path = _relative_path(manifest.parent, selected["path"])
    try:
        image_hash = _file_hash(path)
    except OSError as exc:
        raise ReferenceCatalogError(f"Cannot read asset {selected['id']!r} at {path}: {exc}") from exc
    if selected.get("sha256", image_hash) != image_hash:
        raise ReferenceCatalogError(f"SHA-256 mismatch for asset {selected['id']!r} in {catalog!r}.")
    return {
        "catalog": catalog,
        "catalog_name": data["name"],
        "catalog_version": data["version"],
        "asset_id": selected["id"],
        "description": selected["description"],
        "relative_path": selected["path"],
        "path": str(path),
        "manifest_sha256": manifest_hash,
        "sha256": image_hash,
    }


class CRTP_ReferenceImagePicker:
    CATEGORY = "image/reference"
    FUNCTION = "pick"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "asset_id", "description")
    DESCRIPTION = "Load an exact reference or deterministically select from a versioned catalog."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "catalog_path": ("STRING", {"default": "", "tooltip": "Absolute folder path containing manifest.json and reference images."}),
                "catalog": ("STRING", {"default": "portraits@1.0.0", "tooltip": "Catalog name@version."}),
                "category": ("STRING", {"default": "hair_color"}),
                "value": ("STRING", {"default": "brown"}),
                "selection_seed": ("INT", {
                    "default": 0, "min": 0, "max": MAX_SELECTION_SEED,
                    "control_after_generate": False,
                    "tooltip": "Reference selection only. Keep fixed when reseeding generation.",
                }),
            },
            "optional": {
                "asset_id": ("STRING", {"default": "", "tooltip": "Exact asset ID; overrides filters and seed."}),
            },
        }

    def pick(self, catalog_path, catalog, category, value, selection_seed, asset_id=""):
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        resolved = resolve_reference(catalog_path, catalog, category, value, selection_seed, asset_id)
        try:
            raw = Path(resolved["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != resolved["sha256"]:
                raise ReferenceCatalogError("Reference changed while loading; retry with a stable catalog.")
            with Image.open(io.BytesIO(raw)) as source:
                # One reference always produces one image, even for animated files.
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                pixels = np.array(rgb, dtype=np.float32) / 255.0
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ReferenceCatalogError(f"Cannot load reference image {resolved['asset_id']!r}: {exc}") from exc
        return (torch.from_numpy(pixels).unsqueeze(0), resolved["asset_id"], resolved["description"])

    @classmethod
    def IS_CHANGED(cls, catalog_path, catalog, category, value, selection_seed, asset_id=""):
        try:
            resolved = resolve_reference(catalog_path, catalog, category, value, selection_seed, asset_id)
        except (ReferenceCatalogError, OSError):
            # Never reuse a cached image after deletion/invalid edits. Execution
            # reports the actionable error; this also works with lazy branches.
            return float("nan")
        return (resolved["manifest_sha256"], resolved["asset_id"], resolved["sha256"])


NODE_CLASS_MAPPINGS = {"CRTP_ReferenceImagePicker": CRTP_ReferenceImagePicker}
NODE_DISPLAY_NAME_MAPPINGS = {"CRTP_ReferenceImagePicker": "CRTP Reference Image Picker"}


def main():
    parser = argparse.ArgumentParser(description="Resolve a CRTP reference for a saved job; outputs JSON.")
    parser.add_argument("catalog", help="name@version")
    parser.add_argument("--catalog-path", required=True)
    parser.add_argument("--category", default="")
    parser.add_argument("--value", default="")
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--asset-id", default="")
    args = parser.parse_args()
    try:
        resolved = resolve_reference(
            args.catalog_path, args.catalog, args.category, args.value,
            args.selection_seed, args.asset_id,
        )
    except ReferenceCatalogError as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(resolved, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
