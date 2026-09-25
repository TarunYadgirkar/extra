import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from timm.models import _hub

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features


class BackboneRevisionTests(unittest.TestCase):
    def test_default_backbone_routes_pinned_revision_to_hub(self):
        with patch.object(features.timm, "create_model") as create, \
                patch.object(features, "device", return_value="cpu"):
            result = features.load_model()

        self.assertIs(result, create.return_value.eval.return_value.to.return_value)
        create.return_value.eval.return_value.to.assert_called_once_with("cpu")
        model_name, = create.call_args.args
        self.assertEqual(model_name, "vit_small_patch14_reg4_dinov2.lvd142m")
        kwargs = create.call_args.kwargs
        self.assertTrue(kwargs["pretrained"])
        self.assertTrue(kwargs["dynamic_img_size"])
        self.assertEqual(kwargs["num_classes"], 0)
        hub_id = kwargs["pretrained_cfg_overlay"]["hf_hub_id"]
        with patch.object(_hub, "hf_hub_download", return_value="offline.safetensors") as download, \
                patch.object(_hub.safetensors.torch, "load_file", return_value={}) as load:
            _hub.load_state_dict_from_hf(hub_id)

        download.assert_called_once_with(
            repo_id="timm/vit_small_patch14_reg4_dinov2.lvd142m",
            filename="model.safetensors",
            revision="c04b5193082a8d5b0c4856c7937384a48136c5de",
            cache_dir=None,
        )
        load.assert_called_once_with("offline.safetensors", device="cpu")

    def test_base_backbone_retains_existing_loading_behavior(self):
        with patch.object(features.timm, "create_model") as create, \
                patch.object(features, "device", return_value="cpu"):
            features.load_model("b")

        create.assert_called_once_with(
            "vit_base_patch14_reg4_dinov2.lvd142m",
            pretrained=True,
            dynamic_img_size=True,
            num_classes=0,
        )


if __name__ == "__main__":
    unittest.main()
