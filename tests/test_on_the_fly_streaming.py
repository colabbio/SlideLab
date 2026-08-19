import unittest
import tempfile
import os
import json
import sys

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Add SlideLab root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SlideLab"))

from integrations.omero_writer import OmeroWriter
if HAS_TORCH:
    from tiling.no_saving.cpu import skip_none_collate, StreamingTileDataset, CPUTileDataset

class TestOnTheFlyStreaming(unittest.TestCase):
    
    @unittest.skipUnless(HAS_TORCH, "PyTorch is required for tensor collation tests")
    def test_skip_none_collate_normal_batch(self):
        t1 = torch.ones((3, 256, 256))
        c1 = torch.tensor([0.0, 0.0])
        t2 = torch.ones((3, 256, 256)) * 2
        c2 = torch.tensor([256.0, 0.0])
        
        batch = [(t1, c1), (t2, c2)]
        imgs, coords = skip_none_collate(batch)
        
        self.assertIsNotNone(imgs)
        self.assertIsNotNone(coords)
        self.assertEqual(imgs.shape, (2, 3, 256, 256))
        self.assertEqual(coords.shape, (2, 2))

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required for tensor collation tests")
    def test_skip_none_collate_with_dropped_tiles(self):
        t1 = torch.ones((3, 256, 256))
        c1 = torch.tensor([0.0, 0.0])
        
        # Simulated blurry/glass tile filtered to None
        batch = [(t1, c1), (None, torch.tensor([256.0, 0.0]))]
        imgs, coords = skip_none_collate(batch)
        
        self.assertIsNotNone(imgs)
        self.assertIsNotNone(coords)
        self.assertEqual(imgs.shape, (1, 3, 256, 256))
        self.assertEqual(coords.shape, (1, 2))

    @unittest.skipUnless(HAS_TORCH, "PyTorch is required for tensor collation tests")
    def test_skip_none_collate_all_dropped(self):
        batch = [(None, torch.tensor([0.0, 0.0])), (None, torch.tensor([256.0, 0.0]))]
        imgs, coords = skip_none_collate(batch)
        
        self.assertIsNone(imgs)
        self.assertIsNone(coords)

    def test_omero_writer_offline_mode(self):
        writer = OmeroWriter()
        metrics = {
            "patient_id": "TEST_PATIENT_001",
            "num_tiles": 450,
            "elapsed_cpu_seconds": 12.34,
            "elapsed_user_seconds": 3.45,
            "feature_extractor": "resnet50",
            "device": "cpu"
        }
        
        # Test offline mode returns True without throwing exception
        success = writer.write_pac_and_qc_metrics(
            image_id=9999,
            metrics=metrics,
            pac_score=0.8542,
            prediction_label="MSI-High",
            prediction_confidence=0.94
        )
        self.assertTrue(success)

if __name__ == "__main__":
    unittest.main()
