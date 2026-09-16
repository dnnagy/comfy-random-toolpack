# CRTP Reference Image Picker

`CRTP_ReferenceImagePicker` selects one image from a manifest and returns
`image` (`IMAGE`), `asset_id` (`STRING`), and `description` (`STRING`).
Restart ComfyUI after installing the node, then find **CRTP Reference Image
Picker** under `image/reference`.

## Inputs

| Input | Type | Meaning |
| --- | --- | --- |
| `catalog_path` | STRING | Absolute directory containing `manifest.json`, e.g. `/path/to/faces`. `~` is supported. Path is on the ComfyUI server. |
| `catalog` | STRING | Exact manifest name and version, e.g. `portraits@1.0.0`. |
| `category` | STRING | Attribute name, e.g. `hair_color`. |
| `value` | STRING | Exact, case-sensitive value, e.g. `brown`. |
| `selection_seed` | INT | Selection seed, 0 through 9007199254740991. No automatic after-generation randomization. |
| `asset_id` | STRING, optional | Empty selects by attributes; a non-empty ID loads that exact asset, overriding filters and seed. |

Each node can use a different directory: `/path/to/faces`, `/path/to/clothes`,
or `/path/to/backgrounds`. There is no global catalog directory to configure.
Both `catalog_path` and `catalog` are explicit, so relocating a collection
does not change seeded selection. A mismatched manifest name/version is an error.

## Create a catalog

Copy `examples/reference-catalog/manifest.json` into your chosen directory
and add your own images. The example is a template; portrait images are not
included. A complete directory might be:

```text
/path/to/faces/
  manifest.json
  images/
    portrait-001.png
    portrait-002.png
```

The manifest contains `schema_version: 1`, `name`, `version`, and a non-empty
`assets` array. Every asset has a unique stable `id`, a `path` relative to
the manifest directory, a short `description`, and an `attributes` object.
Attribute values may be strings or non-empty lists of strings. Names and
versions use ASCII letters, digits, dots, underscores, or hyphens, beginning
with a letter or digit.

Paths use forward slashes. Absolute paths, parent traversal, and symlinks
escaping the catalog directory are rejected. Keep stable IDs independent of
filenames; never recycle an ID for a different subject. An optional `sha256`
per asset (64 lowercase hex characters) enforces the image's exact file bytes.

Publish a new version when changing catalog membership, attributes, descriptions,
or images. Keep prior versions in separate directories while jobs refer to them.
The name/version check pins the declared version; immutable version directories
and optional asset checksums are how you preserve the original contents.

## Determinism, loading, and caching

Candidates are sorted by stable asset ID, never filesystem or manifest order.
Selection uses SHA-256 of UTF-8 JSON with compact separators and unescaped Unicode:

```json
["crtp-reference-v1","portraits@1.0.0","hair_color","brown",0]
```

Interpret the digest as an unsigned big-endian integer and take its remainder
modulo candidate count. This is independent of Python's random state and the
generation seed. Different seeds can select the same candidate; changing a
seed does not guarantee a different reference.

An explicit ID ignores filter matching (filters may then be empty). Unknown IDs,
missing files, invalid manifests, checksum mismatches, and unreadable images
raise errors rather than selecting replacements. An animated image contributes
only its first frame. EXIF orientation is applied and pixels are converted to
RGB float32 `[1, H, W, 3]` in `[0, 1]`; alpha is discarded.

`IS_CHANGED` fingerprints the full manifest bytes, selected asset ID, and
selected file bytes with SHA-256, following the
[ComfyUI cache hook](https://docs.comfy.org/custom-nodes/backend/server_overview#is-changed).
Same-size or same-timestamp replacements are detected. Invalid/deleted files
invalidate cached outputs. Unselected files are not decoded or hashed.

## Resolve once when creating a job

The standalone resolver uses only the Python standard library. It validates
the manifest, selects the reference, checks file readability/checksums, and
returns JSON metadata without loading torch or decoding an image:

```bash
python /path/to/comfy-random-toolpack/nodes/reference_picker.py portraits@1.0.0 \
  --catalog-path /path/to/faces \
  --category hair_color --value brown --selection-seed 42
```

The same file exposes `resolve_reference(catalog_path, catalog, category, value,
selection_seed, asset_id="")` for Python callers. Results include `catalog`,
`catalog_name`, `catalog_version`, `asset_id`, `description`, `relative_path`,
absolute `path`, `manifest_sha256`, and image `sha256`.

Save the selected `catalog`, `asset_id`, and checksums with the job. At execution,
pass the saved ID and catalog to the picker with the directory on that execution
host. The same resolver can feed existing loaders using its resolved path.
Image decoding errors are reported by the loader at execution.

For Los Feliz, declare `selection_seed` as an ordinary `int` parameter. Its
Reseed action changes parameters marked `random_seed`; keep that kind for the
generation seed. A separate Shuffle references action should change selection
seeds, re-resolve assets, and persist the new IDs. Changing selection_seed while
keeping an explicit asset_id intentionally keeps the same image.

This pack provides the resolver and picker; job persistence and the Shuffle
references UI belong in the calling application.

## H3 wiring and prompt order

In `MiniMax-H3-Ref2VA-full_api.json`, H3 node `18` receives nine image slots
from nodes `26`, `27`, `28`, `29`, `506`, `507`, `508`, `509`, and `510`.
Replace a chosen loader with `CRTP_ReferenceImagePicker` and keep output index
`0` connected to the same H3 reference slot. For example, replace node `26`:

```json
{
  "class_type": "CRTP_ReferenceImagePicker",
  "inputs": {
    "catalog_path": "/path/to/faces",
    "catalog": "portraits@1.0.0",
    "category": "hair_color",
    "value": "brown",
    "selection_seed": 42,
    "asset_id": "portrait-001"
  }
}
```

Keep optional, unused references on the existing lazy loaders or disconnect
those slots. A picker represents a required, real reference, so an empty ID
means seeded selection, not an absent reference.

Build the prompt and links from the same ordered list of resolved references.
For example, wire identity to `ref_images.ref_image_0` and clothing to
`ref_images.ref_image_1`, then say: “Use <Picture 1> for identity and
<Picture 2> for clothing.” Pack supplied images into contiguous slots and
number only those actual images, starting from 1. The picker description
does not invent a picture number because it cannot know its downstream slot.

Start with one primary portrait and a few focused clothing, pose, or scene
references. The node does not guarantee that H3 separates attributes from
different people; that requires a model experiment with your references.

## Tests

Run with the ComfyUI Python environment (Pillow, NumPy, and torch installed):

```bash
python -m unittest discover -s tests -p 'test_reference_picker.py' -v
```
