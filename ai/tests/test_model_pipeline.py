"""
Automated Test Suite for Smart City AI Microservice & Heuristic Vision Pipeline
Covers:
1. Model checkpoint output classes & architecture verification.
2. Incompatible checkpoint detection & graceful handling.
3. Image classification via Heuristic Vision engine (predictionSource: "heuristic").
4. Verification that /analyze exclusively uses Heuristic Vision and NEVER calls MobileNetV3.
5. Verification that heuristic failure (e.g. corrupt image) returns handled Other and NEVER calls MobileNetV3.
6. Emergency hazard escalation to CRITICAL in heuristic vision.
7. Explicit fallback metadata and diagnostic tracking.
8. Verification that all pre-existing civic categories continue working without regression.
"""

import os
import sys
import io
import json
import torch
import unittest
from unittest.mock import patch
from PIL import Image

# Ensure ai directory is on python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.dirname(CURRENT_DIR)
if AI_DIR not in sys.path:
    sys.path.insert(0, AI_DIR)

from schemas import CIVIC_TAXONOMY
from classifier import (
    MOBILENET_LOADER,
    classify_local,
    classify_complaint,
    classify_heuristic_vision,
    verify_model_architecture,
)
from app import app


class TestAiModelPipeline(unittest.TestCase):

    def setUp(self):
        self.model_dir = os.path.join(AI_DIR, "models")
        self.checkpoint_path = os.path.join(self.model_dir, "best_model.pth")
        self.config_path = os.path.join(self.model_dir, "model_config.json")
        self.client = app.test_client()

    def test_checkpoint_classes_and_street_light_mapping(self):
        """Test 1: Checkpoint output dimensions and Street Light mapping."""
        self.assertTrue(os.path.exists(self.checkpoint_path), "best_model.pth must exist")
        self.assertTrue(os.path.exists(self.config_path), "model_config.json must exist")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        classes = config.get("classes", [])
        self.assertEqual(len(classes), 6, "Model config must define exactly 6 classes")
        self.assertIn("Street Light", classes, "Street Light must be in model classes")
        self.assertEqual(classes[3], "Street Light", "Street Light must be at index 3")

        # Load weights and verify classifier layer shape
        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        weight_key = "classifier.3.weight"
        self.assertIn(weight_key, state_dict, f"State dict must contain {weight_key}")
        weight_shape = state_dict[weight_key].shape
        self.assertEqual(weight_shape[0], 6, "Classifier output classes must equal 6")
        self.assertEqual(weight_shape[1], 1024, "Classifier input features must equal 1024")

    def test_incompatible_checkpoint_fails_safely(self):
        """Test 2: Incompatible checkpoint produces clear error / disabled status."""
        import torchvision.models as tv_models
        dummy_model = tv_models.mobilenet_v3_small(weights=None)
        in_features = dummy_model.classifier[3].in_features
        dummy_model.classifier[3] = torch.nn.Linear(in_features, 99)

        is_valid = verify_model_architecture(dummy_model, expected_classes=6)
        self.assertFalse(is_valid, "Architecture verification must reject 99-class model when expecting 6")

    def test_heuristic_vision_image_classification(self):
        """Test 3: Image classification exclusively uses Heuristic Vision engine."""
        sample_img_path = os.path.join(AI_DIR, "data", "Street Light", "street_light_001.jpg")
        self.assertTrue(os.path.exists(sample_img_path), f"Sample image {sample_img_path} must exist")

        with patch.object(MOBILENET_LOADER, "predict") as mock_mobilenet:
            result = classify_complaint(
                image_path=sample_img_path,
                description="Street light pole not working at night",
                request_id="test-req-001"
            )

            # Proves MobileNetV3 was NOT invoked
            mock_mobilenet.assert_not_called()

        self.assertIsNotNone(result, "classify_complaint must return a valid dict")
        self.assertEqual(result["predictedCategory"], "Street Light", "Sample must predict Street Light")
        self.assertEqual(result["predictionSource"], "heuristic", "Prediction source must be 'heuristic'")
        self.assertEqual(result["source"], "heuristic", "Source must be 'heuristic'")
        self.assertEqual(result["model"], "SmartCity-HeuristicVision-v1.0")
        self.assertFalse(result["fallbackUsed"], "fallbackUsed must be False on successful heuristic vision inference")
        self.assertGreater(result["confidence"], 0.5, "Confidence must be reasonable for sample")
        self.assertEqual(result["requestId"], "test-req-001", "requestId must match")
        self.assertEqual(result["department"], "Electrical Engineering & Public Lighting Division")
        self.assertIsNotNone(result.get("features"), "Visual features must be present")

    def test_analyze_endpoint_uses_heuristic_and_does_not_call_mobilenet(self):
        """Test 4: /analyze endpoint uses heuristic vision and does NOT invoke MobileNetV3."""
        sample_img_path = os.path.join(AI_DIR, "data", "Street Light", "street_light_001.jpg")
        self.assertTrue(os.path.exists(sample_img_path), f"Sample image {sample_img_path} must exist")

        with open(sample_img_path, "rb") as img_file:
            img_bytes = img_file.read()

        with patch.object(MOBILENET_LOADER, "predict") as mock_mobilenet:
            response = self.client.post(
                "/analyze",
                data={
                    "image": (io.BytesIO(img_bytes), "street_light.jpg"),
                    "description": "Street light damaged on main cross road",
                    "requestId": "req-analyze-test-01"
                },
                content_type="multipart/form-data"
            )

            # Assert MobileNetV3 was NEVER called in this flow
            mock_mobilenet.assert_not_called()

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNotNone(data)
        self.assertEqual(data["predictionSource"], "heuristic", "predictionSource must be 'heuristic'")
        self.assertEqual(data["source"], "heuristic", "source must be 'heuristic'")
        self.assertEqual(data["predictedCategory"], "Street Light")
        self.assertEqual(data["department"], "Electrical Engineering & Public Lighting Division")
        self.assertFalse(data["fallbackUsed"])
        self.assertIn("confidence", data)
        self.assertIn("reason", data)

    def test_analyze_corrupt_image_fails_heuristically_and_never_calls_mobilenet(self):
        """Test 5: Corrupt image returns handled Other result with clear reason and NEVER calls MobileNetV3."""
        corrupt_bytes = b"CORRUPTED_NON_IMAGE_BINARY_BYTES_XYZ_9876543210"

        with patch.object(MOBILENET_LOADER, "predict") as mock_mobilenet:
            response = self.client.post(
                "/analyze",
                data={
                    "image": (io.BytesIO(corrupt_bytes), "corrupt_test.jpg"),
                    "description": "Random unreadable submission",
                    "requestId": "req-corrupt-002"
                },
                content_type="multipart/form-data"
            )

            # Crucial: MobileNetV3 is NOT invoked even when heuristic analysis fails
            mock_mobilenet.assert_not_called()

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNotNone(data)
        self.assertEqual(data["predictedCategory"], "Other", "Corrupt image must return 'Other'")
        self.assertEqual(data["subcategory"], "General Issue")
        self.assertEqual(data["predictionSource"], "heuristic", "Must never label MobileNetV3 as heuristic")
        self.assertEqual(data["source"], "heuristic")
        self.assertTrue(data["fallbackUsed"])
        self.assertTrue(data["isFallback"])
        self.assertIn("Heuristic vision analysis failed", data["reason"])
        self.assertIsNotNone(data.get("error"))

    def test_emergency_hazard_escalation_in_heuristic_vision(self):
        """Test 6: Emergency hazard detection escalates priority to CRITICAL."""
        sample_img_path = os.path.join(AI_DIR, "data", "Street Light", "street_light_001.jpg")
        with open(sample_img_path, "rb") as img_file:
            img_bytes = img_file.read()

        with patch.object(MOBILENET_LOADER, "predict") as mock_mobilenet:
            response = self.client.post(
                "/analyze",
                data={
                    "image": (io.BytesIO(img_bytes), "street_light.jpg"),
                    "description": "Live wire sparking and hanging from pole accident hazard fatal danger",
                    "requestId": "req-hazard-003"
                },
                content_type="multipart/form-data"
            )
            mock_mobilenet.assert_not_called()

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["priority"], "CRITICAL", "Hazard keywords must escalate to CRITICAL")
        self.assertIn("CRITICAL", data["reason"])
        self.assertEqual(data["predictionSource"], "heuristic")
        self.assertEqual(data["source"], "heuristic")

    def test_rule_based_fallback_metadata(self):
        """Test 7: Local heuristic classifier metadata for text-only reports."""
        text = "The street light pole is broken and flickering in the dark"
        result = classify_local(
            description=text,
            fallback_reason="Text-only report",
            request_id="test-fallback-002"
        )

        self.assertEqual(result["predictedCategory"], "Street Light")
        self.assertEqual(result["predictionSource"], "heuristic")
        self.assertEqual(result["source"], "heuristic")
        self.assertTrue(result["fallbackUsed"])
        self.assertEqual(result["fallbackReason"], "Text-only report")
        self.assertEqual(result["requestId"], "test-fallback-002")
        self.assertIn(result["predictedSubcategory"], [
            "Street Light Not Working",
            "Street Light Damaged",
            "Street Light Missing",
            "Street Light Flickering",
            "Street Light Pole Damaged",
            "Multiple Street Lights Not Working"
        ])

    def test_existing_categories_continue_working(self):
        """Test 8: Existing categories continue working without regression."""
        expected_categories = [
            "Waste Management",
            "Roads",
            "Drainage",
            "Water Supply",
            "Street Light",
            "Street Infrastructure",
            "Public Property",
            "Other"
        ]

        for cat in expected_categories:
            self.assertIn(cat, CIVIC_TAXONOMY, f"{cat} must exist in CIVIC_TAXONOMY")
            self.assertGreater(len(CIVIC_TAXONOMY[cat]["subcategories"]), 0, f"{cat} must have subcategories")

        drainage_result = classify_local(description="Severe sewage overflow and blocked drainage pipe", request_id="test-drainage")
        self.assertEqual(drainage_result["predictedCategory"], "Drainage")
        self.assertEqual(drainage_result["predictionSource"], "heuristic")

        roads_result = classify_local(description="Dangerous pothole on the main road", request_id="test-roads")
        self.assertEqual(roads_result["predictedCategory"], "Roads")
        self.assertEqual(roads_result["predictionSource"], "heuristic")

        waste_result = classify_local(description="Garbage dump overflowing with trash", request_id="test-waste")
        self.assertEqual(waste_result["predictedCategory"], "Waste Management")
        self.assertEqual(waste_result["predictionSource"], "heuristic")

        water_result = classify_local(description="Drinking water supply pipeline burst and leakage leaving residents with no water", request_id="test-water")
        self.assertEqual(water_result["predictedCategory"], "Water Supply")
        self.assertEqual(water_result["predictionSource"], "heuristic")


if __name__ == "__main__":
    unittest.main()
