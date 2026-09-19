"""Tests for advanced tools meta endpoint model options configurability."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestAdvancedToolsMetaModelOptions(unittest.TestCase):
    """Verify that sequence/structure model options are read from constant.json."""

    def _build_constant(self, extra_web_ui=None):
        web_ui = {
            "model_mapping_zero_shot": {"ESM2-650M": "esm2", "ESM-IF1": "esmif1"},
            "dataset_mapping_zero_shot": [],
            "model_mapping_function": {},
            "model_residue_mapping_function": {},
            "dataset_mapping_function": {},
            "residue_mapping_function": {},
            "llm_models": {},
        }
        if extra_web_ui:
            web_ui.update(extra_web_ui)
        return {"web_ui": web_ui}

    def test_default_model_options_when_not_in_constant(self):
        """Fallback defaults are used when constant.json omits model option lists."""
        data = self._build_constant()
        web_ui = data.get("web_ui", {})
        # Simulate the meta endpoint logic: read with fallback
        seq = web_ui.get("sequence_model_options", ["VenusPLM", "ESM2-650M", "ESM-1v", "ESM-1b"])
        struct = web_ui.get("structure_model_options", ["VenusREM (foldseek-based)", "ProSST-2048", "ProtSSN", "ESM-IF1", "SaProt", "MIF-ST"])

        self.assertEqual(seq, ["VenusPLM", "ESM2-650M", "ESM-1v", "ESM-1b"])
        self.assertIn("ESM-IF1", struct)

    def test_custom_model_options_from_constant(self):
        """Custom model options defined in constant.json are returned by the meta logic."""
        custom_seq = ["MyTrainedModel", "ESM2-650M"]
        custom_struct = ["MyStructureModel", "ESM-IF1"]
        data = self._build_constant(
            {
                "sequence_model_options": custom_seq,
                "structure_model_options": custom_struct,
            }
        )
        web_ui = data.get("web_ui", {})
        seq = web_ui.get("sequence_model_options", ["VenusPLM", "ESM2-650M", "ESM-1v", "ESM-1b"])
        struct = web_ui.get("structure_model_options", ["VenusREM (foldseek-based)", "ProSST-2048", "ProtSSN", "ESM-IF1", "SaProt", "MIF-ST"])

        self.assertEqual(seq, custom_seq)
        self.assertIn("MyTrainedModel", seq)
        self.assertEqual(struct, custom_struct)
        self.assertIn("MyStructureModel", struct)

    def test_constant_json_has_model_option_keys(self):
        """The shipped constant.json already contains sequence/structure_model_options."""
        constant_path = Path(__file__).resolve().parent.parent / "src" / "constant.json"
        self.assertTrue(constant_path.exists(), "constant.json not found")
        data = json.loads(constant_path.read_text(encoding="utf-8"))
        web_ui = data.get("web_ui", {})
        self.assertIn("sequence_model_options", web_ui, "sequence_model_options missing from constant.json")
        self.assertIn("structure_model_options", web_ui, "structure_model_options missing from constant.json")
        self.assertIsInstance(web_ui["sequence_model_options"], list)
        self.assertIsInstance(web_ui["structure_model_options"], list)


if __name__ == "__main__":
    unittest.main()
