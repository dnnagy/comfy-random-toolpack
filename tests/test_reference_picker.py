import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "reference_picker.py"
spec = importlib.util.spec_from_file_location("reference_picker", MODULE_PATH)
picker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(picker)


class ReferencePickerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog_path = self.root / "faces"
        self.catalog_path.mkdir()
        self.manifest = self.catalog_path / "manifest.json"
        self.data = {
            "schema_version": 1, "name": "portraits", "version": "1.0.0",
            "assets": [
                {"id": "portrait-b", "path": "b.png", "description": "Second portrait",
                 "attributes": {"hair_color": ["brown", "auburn"]}},
                {"id": "portrait-a", "path": "a.png", "description": "First portrait",
                 "attributes": {"hair_color": "brown"}},
                {"id": "portrait-c", "path": "c.png", "description": "Third portrait",
                 "attributes": {"hair_color": "blonde"}},
            ],
        }
        for filename, color in [("a.png", "red"), ("b.png", "green"), ("c.png", "blue")]:
            Image.new("RGB", (4, 3), color).save(self.catalog_path / filename)
        self.write_manifest()
        self.inputs = dict(catalog_path=str(self.catalog_path), catalog="portraits@1.0.0",
                           category="hair_color", value="brown", selection_seed=0)

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")

    def resolve(self, **changes):
        return picker.resolve_reference(**(self.inputs | changes))

    def fingerprint(self, **changes):
        return picker.CRTP_ReferenceImagePicker.IS_CHANGED(**(self.inputs | changes))

    def test_determinism_and_order_independence(self):
        before = [self.resolve(selection_seed=i)["asset_id"] for i in range(30)]
        self.assertEqual(set(before), {"portrait-a", "portrait-b"})
        self.data["assets"].reverse()
        self.write_manifest()
        after = [self.resolve(selection_seed=i)["asset_id"] for i in range(30)]
        self.assertEqual(before, after)
        # Freeze the cross-language selection contract with an independent vector.
        key = b'["crtp-reference-v1","portraits@1.0.0","hair_color","brown",0]'
        expected = ["portrait-a", "portrait-b"][int(hashlib.sha256(key).hexdigest(), 16) % 2]
        self.assertEqual(before[0], expected)

    def test_exact_id_overrides_filters_and_seed(self):
        for seed in [0, 40, picker.MAX_SELECTION_SEED]:
            result = self.resolve(asset_id="portrait-c", category="", value="", selection_seed=seed)
            self.assertEqual(result["asset_id"], "portrait-c")
            self.assertEqual(result["catalog_version"], "1.0.0")
            self.assertEqual(result["description"], "Third portrait")

    def test_matching_is_exact_and_supports_lists(self):
        self.assertEqual(self.resolve(value="auburn")["asset_id"], "portrait-b")
        for changes in [{"value": "Brown"}, {"category": "missing"}, {"value": ""}, {"asset_id": "missing"}]:
            with self.subTest(changes=changes), self.assertRaises(picker.ReferenceCatalogError):
                self.resolve(**changes)

    def test_plain_integer_input_and_optional_id(self):
        schema = picker.CRTP_ReferenceImagePicker.INPUT_TYPES()
        kind, config = schema["required"]["selection_seed"]
        self.assertEqual(kind, "INT")
        self.assertFalse(config["control_after_generate"])
        self.assertIn("catalog_path", schema["required"])
        self.assertIn("asset_id", schema["optional"])
        for seed in [-1, True, 0.5, "1", picker.MAX_SELECTION_SEED + 1]:
            with self.subTest(seed=seed), self.assertRaises(picker.ReferenceCatalogError):
                self.resolve(selection_seed=seed)

    def test_catalog_identity_and_version_are_pinned(self):
        for catalog in ["portraits", "portraits@2.0.0", "other@1.0.0", "../portraits@1.0.0"]:
            with self.subTest(catalog=catalog), self.assertRaises(picker.ReferenceCatalogError):
                self.resolve(catalog=catalog)

    def test_independent_catalog_paths(self):
        clothes = self.root / "clothes"
        clothes.mkdir()
        data = copy.deepcopy(self.data)
        data["name"] = "clothes"
        data["assets"] = [{"id": "shirt", "path": "shirt.png", "description": "White shirt",
                           "attributes": {"garment": "shirt"}}]
        (clothes / "manifest.json").write_text(json.dumps(data))
        Image.new("RGB", (2, 2), "white").save(clothes / "shirt.png")
        result = self.resolve(catalog_path=str(clothes), catalog="clothes@1.0.0",
                              category="garment", value="shirt")
        self.assertEqual(result["asset_id"], "shirt")
        self.assertEqual(self.resolve(asset_id="portrait-a")["asset_id"], "portrait-a")

    def test_invalid_manifest(self):
        original = copy.deepcopy(self.data)
        variants = [[], {}, original | {"schema_version": True}, original | {"assets": []},
                    original | {"assets": original["assets"] * 2},
                    original | {"assets": [{"id": "bad"}]}]
        for data in variants:
            with self.subTest(data=data), self.assertRaises(picker.ReferenceCatalogError):
                self.data = data
                self.write_manifest()
                self.resolve()
        for raw in ['{', '{"schema_version":1,"schema_version":1}']:
            self.manifest.write_text(raw)
            with self.assertRaises(picker.ReferenceCatalogError):
                self.resolve()

    def test_paths_cannot_escape_catalog(self):
        for path in ["../outside.png", "/tmp/outside.png", "C:\\outside.png", "foo\\bar.png"]:
            self.data["assets"][0]["path"] = path
            self.write_manifest()
            with self.subTest(path=path), self.assertRaises(picker.ReferenceCatalogError):
                self.resolve()
        outside = self.root / "outside.png"
        Image.new("RGB", (1, 1)).save(outside)
        (self.catalog_path / "escape.png").symlink_to(outside)
        self.data["assets"][0]["path"] = "escape.png"
        self.write_manifest()
        with self.assertRaises(picker.ReferenceCatalogError):
            self.resolve()

    def test_invalid_catalog_path(self):
        for path in ["", "relative/faces", str(self.root / "missing")]:
            with self.subTest(path=path), self.assertRaises(picker.ReferenceCatalogError):
                self.resolve(catalog_path=path)

    def test_cache_tracks_manifest_and_image_bytes_with_same_stat(self):
        initial = self.fingerprint(asset_id="portrait-a")
        self.assertEqual(initial, self.fingerprint(asset_id="portrait-a"))
        self.data["assets"][1]["description"] = "Updated description"
        self.write_manifest()
        updated = self.fingerprint(asset_id="portrait-a")
        self.assertNotEqual(initial, updated)
        path = self.catalog_path / "a.png"
        stat = path.stat()
        contents = bytearray(path.read_bytes())
        contents[-1] ^= 1
        path.write_bytes(contents)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(updated, self.fingerprint(asset_id="portrait-a"))

    def test_deleted_selected_asset_never_falls_back(self):
        selected = self.resolve()
        Path(selected["path"]).unlink()
        with self.assertRaisesRegex(picker.ReferenceCatalogError, "Cannot read asset"):
            self.resolve()
        self.assertTrue(math.isnan(self.fingerprint()))

    def test_optional_sha256_enforces_content(self):
        asset = self.data["assets"][1]
        asset["sha256"] = self.resolve(asset_id=asset["id"])["sha256"]
        self.write_manifest()
        self.resolve(asset_id=asset["id"])
        Image.new("RGB", (4, 3), "white").save(self.catalog_path / asset["path"])
        with self.assertRaisesRegex(picker.ReferenceCatalogError, "SHA-256 mismatch"):
            self.resolve(asset_id=asset["id"])

    def test_output_is_comfy_rgb_tensor(self):
        image, asset_id, description = picker.CRTP_ReferenceImagePicker().pick(
            **self.inputs, asset_id="portrait-a")
        self.assertEqual(tuple(image.shape), (1, 3, 4, 3))
        self.assertEqual(image.dtype, torch.float32)
        np.testing.assert_array_equal(image[0, 0, 0].numpy(), [1, 0, 0])
        self.assertEqual((asset_id, description), ("portrait-a", "First portrait"))

    def test_exif_orientation_and_non_rgb_images(self):
        path = self.catalog_path / "a.png"
        source = Image.new("RGB", (4, 3), "red")
        exif = source.getexif()
        exif[274] = 6
        source.save(path, exif=exif)
        result = picker.CRTP_ReferenceImagePicker().pick(**self.inputs, asset_id="portrait-a")[0]
        self.assertEqual(tuple(result.shape), (1, 4, 3, 3))
        for mode in ["L", "RGBA", "P"]:
            Image.new(mode, (4, 3)).save(path)
            result = picker.CRTP_ReferenceImagePicker().pick(**self.inputs, asset_id="portrait-a")[0]
            self.assertEqual(tuple(result.shape), (1, 3, 4, 3))

    def test_animation_loads_only_first_frame_and_corrupt_image_errors(self):
        path = self.catalog_path / "a.png"
        Image.new("RGB", (4, 3), "red").save(
            path, format="GIF", save_all=True, append_images=[Image.new("RGB", (4, 3), "blue")])
        result = picker.CRTP_ReferenceImagePicker().pick(**self.inputs, asset_id="portrait-a")[0]
        self.assertEqual(tuple(result.shape), (1, 3, 4, 3))
        np.testing.assert_array_equal(result[0, 0, 0].numpy(), [1, 0, 0])
        path.write_text("not an image")
        with self.assertRaisesRegex(picker.ReferenceCatalogError, "Cannot load reference image"):
            picker.CRTP_ReferenceImagePicker().pick(**self.inputs, asset_id="portrait-a")

    def test_cli_works_without_comfy_or_site_packages(self):
        result = subprocess.run([
            sys.executable, "-S", str(MODULE_PATH), "portraits@1.0.0",
            "--catalog-path", str(self.catalog_path), "--category", "hair_color",
            "--value", "brown", "--selection-seed", "0",
        ], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout), self.resolve())


if __name__ == "__main__":
    unittest.main()
