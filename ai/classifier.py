"""
Smart City Civic AI Classifier
Core classification engine integrating fine-tuned PyTorch MobileNetV3 Small computer vision,
safety hazard detection, natural language processing, and optional Gemini LLM intelligence.
"""

import os
import re
import json
import time
from typing import Optional, Dict, Any, List, Tuple

from schemas import (
    CIVIC_TAXONOMY,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW,
    ClassificationResult,
    get_department_for_category,
    validate_category,
    validate_subcategory
)
from utils import (
    clean_text,
    detect_hazard_keywords,
    extract_image_features,
    encode_image_to_base64
)
from prompt import build_civic_analysis_prompt

# Optional PyTorch & Torchvision imports for fine-tuned MobileNetV3
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try importing google-genai for optional LLM reasoning if configured
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def verify_model_architecture(model: Any, expected_classes: int = 6, image_size: int = 224, device: str = "cpu") -> bool:
    """
    Validates model forward pass and verifies output classes match expectations.
    Returns True if valid, False if dimensions mismatch or forward pass fails.
    """
    if not TORCH_AVAILABLE or model is None:
        return False
    try:
        dummy_input = torch.zeros(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            dummy_output = model(dummy_input)
        return dummy_output.shape == (1, expected_classes)
    except Exception as e:
        print(f"[Model Verification Failure] {e}")
        return False


class MobileNetV3ModelLoader:
    """
    Singleton loader for the fine-tuned MobileNetV3 Small civic classification model.
    Loads weights once at service startup and performs thread-safe, fast CPU inference.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), "models")
        self.weights_path = os.path.join(self.models_dir, "best_model.pth")
        self.config_path = os.path.join(self.models_dir, "model_config.json")
        self.is_loaded = False
        self.model = None
        self.config: Dict[str, Any] = {}
        self.classes: List[str] = []
        self.num_classes: int = 0
        self.load_error: Optional[str] = None
        self.load_time_ms: float = 0.0
        self.device = torch.device("cpu") if TORCH_AVAILABLE else None
        self.transform = None
        self.softmax = None

        self._load()

    def _load(self) -> None:
        if not TORCH_AVAILABLE:
            self.load_error = "PyTorch or Torchvision is not installed in the Python environment."
            print(f"[MobileNetV3 Loader Warning] {self.load_error}")
            return

        if not os.path.exists(self.weights_path):
            self.load_error = f"Model weights not found at: {self.weights_path}"
            print(f"[MobileNetV3 Loader Info] {self.load_error}")
            return

        if not os.path.exists(self.config_path):
            self.load_error = f"Model configuration not found at: {self.config_path}"
            print(f"[MobileNetV3 Loader Info] {self.load_error}")
            return

        t0 = time.perf_counter()
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)

            self.classes = self.config.get("classes", [])
            self.num_classes = len(self.classes)
            image_size = self.config.get("image_size", 224)
            mean = self.config.get("input_mean", [0.485, 0.456, 0.406])
            std = self.config.get("input_std", [0.229, 0.224, 0.225])

            # Build MobileNetV3 Small architecture
            from torchvision.models import mobilenet_v3_small
            model = mobilenet_v3_small(weights=None)
            in_features = model.classifier[3].in_features
            model.classifier[2] = nn.Dropout(p=0.2, inplace=True)
            model.classifier[3] = nn.Linear(in_features=in_features, out_features=self.num_classes)

            # Load trained weights state_dict
            state_dict = torch.load(self.weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()

            # Architecture verification: ensure forward pass works and matches class dimensions
            dummy_input = torch.zeros(1, 3, image_size, image_size, device=self.device)
            with torch.no_grad():
                dummy_output = model(dummy_input)
            if dummy_output.shape != (1, self.num_classes):
                raise ValueError(
                    f"Model output dimension mismatch: expected (1, {self.num_classes}), got {dummy_output.shape}"
                )

            self.model = model.to(self.device)

            self.transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            self.softmax = nn.Softmax(dim=1)
            self.is_loaded = True
            self.load_error = None
            self.load_time_ms = round((time.perf_counter() - t0) * 1000, 2)
            print(f"[MobileNetV3 Loader] Successfully loaded and verified '{self.config.get('model_name', 'MobileNetV3-Small')}' "
                  f"({self.num_classes} classes) in {self.load_time_ms} ms on {self.device}.")

        except Exception as e:
            self.is_loaded = False
            self.model = None
            self.load_error = f"Error during model loading or verification: {str(e)}"
            print(f"[MobileNetV3 Loader Error] {self.load_error}")

    def reload(self) -> Dict[str, Any]:
        """Reloads configuration and checkpoint weights from disk."""
        self._load()
        return self.get_status()

    @torch.inference_mode()
    def predict(self, image_path: str, top_k: int = 3) -> Tuple[Optional[str], float, List[Dict[str, Any]], float]:
        """
        Runs CPU inference on an input image.
        Returns:
            (predicted_category, confidence, top_predictions, latency_ms)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"MobileNetV3 model is not loaded: {self.load_error}")

        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            raise ValueError(f"Invalid or empty image file: {image_path}")

        t0 = time.perf_counter()
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = self.softmax(logits)[0]

        k = min(top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probs, k=k)

        top_preds = []
        for p, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            top_preds.append({
                "category": self.classes[idx],
                "confidence": round(float(p), 4)
            })

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        best_cat = top_preds[0]["category"]
        best_conf = top_preds[0]["confidence"]

        return best_cat, best_conf, top_preds, latency_ms

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model_name": self.config.get("model_name", "MobileNetV3-Small-Civic") if self.is_loaded else None,
            "architecture": "mobilenet_v3_small",
            "version": self.config.get("version", "1.0.0") if self.is_loaded else None,
            "device": str(self.device) if self.device else None,
            "classes": self.classes,
            "num_classes": self.num_classes,
            "load_time_ms": self.load_time_ms,
            "error": self.load_error,
            "verified": self.is_loaded and (self.model is not None)
        }


# Initialize model loader once during module import
MOBILENET_LOADER = MobileNetV3ModelLoader()


def get_model_status() -> Dict[str, Any]:
    """Exposes current model loading status and diagnostics for the health check."""
    return MOBILENET_LOADER.get_status()


def reload_model() -> Dict[str, Any]:
    """Reloads the model weights and config from disk."""
    return MOBILENET_LOADER.reload()


def classify_with_gemini(
    image_path: Optional[str] = None,
    description: str = "",
    api_key: Optional[str] = None,
    request_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Attempts multimodal classification using Gemini if API key and package are available.
    Returns None if unavailable or on error, allowing graceful fallback to local engine.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key or not GENAI_AVAILABLE:
        return None

    try:
        client = genai.Client(api_key=key)
        has_image = bool(image_path and os.path.exists(image_path))
        prompt_text = build_civic_analysis_prompt(description=description, has_image=has_image)

        contents = [prompt_text]

        if has_image:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            ext = os.path.splitext(image_path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "image/jpeg"
            contents.append(
                genai.types.Part.from_bytes(data=img_bytes, mime_type=mime)
            )

        candidate_models = [
            os.environ.get("GEMINI_MODEL"),
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
            "gemini-flash-latest"
        ]
        candidate_models = [m for m in candidate_models if m]

        response = None
        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config={"response_mime_type": "application/json"}
                )
                if response and response.text:
                    break
            except Exception:
                continue

        if not response or not response.text:
            return None

        data = json.loads(response.text)
        cat = data.get("category", "Other")
        if not validate_category(cat):
            cat = "Other"

        sub = data.get("subcategory", "")
        if not validate_subcategory(cat, sub):
            sub = CIVIC_TAXONOMY[cat]["subcategories"][0]

        conf = float(data.get("confidence", 0.92))
        priority = data.get("priority", "MEDIUM")
        reason = data.get("reason", "Analyzed by multimodal civic vision AI")
        dept = data.get("department") or get_department_for_category(cat)

        return {
            "category": cat,
            "subcategory": sub,
            "predictedCategory": cat,
            "predictedSubcategory": sub,
            "priority": priority,
            "confidence": conf,
            "reason": reason,
            "department": dept,
            "model": "Gemini-Multimodal-Civic",
            "modelVersion": None,
            "isFallback": False,
            "fallbackUsed": False,
            "fallbackReason": None,
            "modelLoaded": MOBILENET_LOADER.is_loaded,
            "source": "gemini",
            "predictionSource": "gemini_vision",
            "requestId": request_id,
        }
    except Exception as e:
        print(f"[SmartCity Classifier] Gemini call warning (using local fallback): {e}")
        return None


def classify_heuristic_vision(
    image_path: Optional[str] = None,
    description: str = "",
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Primary and exclusive heuristic vision classification engine for civic complaints.
    Analyzes visual features (brightness, contrast, dominant tint: greyish asphalt,
    dark nighttime scene, wet bluish-green liquid, greenish foliage) combined with
    lexical keyword matching and emergency safety hazard escalation.

    Operates 100% locally on CPU without neural network overhead or paid external APIs.
    MobileNetV3 is NEVER invoked in this classification pipeline.
    """
    has_image = bool(image_path)
    text = clean_text(description)
    is_hazard, hazard_priority, matched_hazards = detect_hazard_keywords(text)

    # 1. Feature extraction & image inspection
    img_features = extract_image_features(image_path)

    # Handled failure case: image was provided but failed decoding or verification
    if has_image and not img_features.valid:
        error_msg = img_features.error or "Corrupt or unreadable image file"
        print(f"[Heuristic Vision Error] [{request_id}] Image analysis failed: {error_msg}")

        # Preserve emergency hazard escalation if text description has hazard keywords
        priority = hazard_priority if is_hazard else PRIORITY_MEDIUM
        reason = f"Heuristic vision analysis failed: {error_msg}. Categorized as Other."
        if is_hazard:
            reason += f" Escalated to {priority} due to hazard indicators: [{', '.join(matched_hazards)}]."

        result = ClassificationResult(
            category="Other",
            subcategory="General Issue",
            priority=priority,
            confidence=0.50,
            reason=reason,
            department=get_department_for_category("Other"),
            model="SmartCity-HeuristicVision-v1.0",
            isFallback=True,
            source="heuristic",
            predictionSource="heuristic",
            predictedCategory="Other",
            predictedSubcategory="General Issue",
            modelLoaded=MOBILENET_LOADER.is_loaded,
            modelVersion=None,
            fallbackUsed=True,
            fallbackReason=f"Heuristic vision error: {error_msg}",
            requestId=request_id,
            features=None,
            error=error_msg,
        )
        res_dict = result.to_dict()
        res_dict["top_predictions"] = [
            {"category": "Other", "confidence": 0.50}
        ]
        return res_dict

    # 2. Heuristic scoring across civic taxonomy
    scores: Dict[str, float] = {}

    for cat, data in CIVIC_TAXONOMY.items():
        score = 0.0

        # Lexical keyword matching
        for kw in data["keywords"]:
            if re.search(r"\b" + re.escape(kw) + r"\b", text):
                score += 3.0
            elif kw in text:
                score += 1.5

        # Category-specific hazard keyword bonus
        for h_kw in data.get("hazard_keywords", []):
            if h_kw in text:
                score += 3.5

        # Multimodal computer vision feature correlation
        if cat in ["Street Light", "Street Infrastructure"] and img_features.is_dark:
            score += 2.5

        if cat == "Roads" and img_features.is_greyish:
            score += 2.0

        if cat in ["Water Supply", "Drainage"] and img_features.is_wet_blue:
            score += 2.5

        if cat == "Public Property" and img_features.is_greenish:
            score += 2.0

        scores[cat] = score

    best_cat = max(scores, key=lambda c: scores[c])
    highest_score = scores[best_cat]

    if highest_score <= 0:
        # Visual heuristic defaults when no keywords matched
        if img_features.is_dark and not text:
            best_cat = "Street Light"
            confidence = 0.60
        elif img_features.is_greyish and not text:
            best_cat = "Roads"
            confidence = 0.60
        elif img_features.is_wet_blue and not text:
            best_cat = "Drainage"
            confidence = 0.60
        elif img_features.is_greenish and not text:
            best_cat = "Public Property"
            confidence = 0.60
        else:
            best_cat = "Waste Management" if not text else "Other"
            confidence = 0.60
    else:
        confidence = round(min(0.85, 0.65 + (highest_score * 0.02)), 2)

    cat_data = CIVIC_TAXONOMY[best_cat]

    # 3. Subcategory mapping based on text tokens
    selected_sub = cat_data["subcategories"][0]
    for sub in cat_data["subcategories"]:
        sub_tokens = [w.lower() for w in re.findall(r"\w+", sub)]
        if any(w in text for w in sub_tokens if len(w) > 3):
            selected_sub = sub
            break

    # 4. Priority evaluation and emergency safety hazard escalation
    priority = cat_data["default_priority"]
    if is_hazard:
        priority = hazard_priority

    # 5. Department routing
    department = get_department_for_category(best_cat)

    # 6. Explanation construction
    visual_indicators = []
    if img_features.is_dark:
        visual_indicators.append("dark/nighttime illumination")
    if img_features.is_greyish:
        visual_indicators.append("greyish asphalt/road surface")
    if img_features.is_wet_blue:
        visual_indicators.append("wet/fluid reflection")
    if img_features.is_greenish:
        visual_indicators.append("greenery/vegetation")

    reason = f"Classified by heuristic vision engine: {best_cat} ({selected_sub})."
    if visual_indicators:
        reason += f" Visual indicators: [{', '.join(visual_indicators)}]."
    if is_hazard:
        hazard_list_str = ", ".join(matched_hazards)
        reason += f" Escalated to {priority} due to critical hazard indicators: [{hazard_list_str}]."

    # 7. Top predictions formatted from heuristic scoring matrix
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top_preds = []
    for cat_name, sc in sorted_cats:
        c_val = round(confidence, 4) if cat_name == best_cat else round(max(0.10, min(0.60, 0.20 + max(0.0, sc) * 0.04)), 4)
        top_preds.append({"category": cat_name, "confidence": c_val})

    result = ClassificationResult(
        category=best_cat,
        subcategory=selected_sub,
        priority=priority,
        confidence=confidence,
        reason=reason,
        department=department,
        model="SmartCity-HeuristicVision-v1.0",
        isFallback=False,
        source="heuristic",
        predictionSource="heuristic",
        predictedCategory=best_cat,
        predictedSubcategory=selected_sub,
        modelLoaded=MOBILENET_LOADER.is_loaded,
        modelVersion=None,
        fallbackUsed=False,
        fallbackReason=None,
        requestId=request_id,
        features=img_features.to_dict() if img_features.valid else None,
        error=None,
    )

    res_dict = result.to_dict()
    res_dict["top_predictions"] = top_preds
    return res_dict


def classify_local(
    image_path: Optional[str] = None,
    description: str = "",
    fallback_reason: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Keyword and sensory heuristic classifier.
    Preserves backwards-compatibility with existing tests and scripts.
    """
    result = classify_heuristic_vision(
        image_path=image_path,
        description=description,
        request_id=request_id
    )
    if fallback_reason:
        result["fallbackReason"] = fallback_reason
        result["fallbackUsed"] = True
        result["isFallback"] = True
    return result


def classify_complaint(
    image_path: Optional[str] = None,
    description: str = "",
    use_llm: bool = False,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main classification orchestrator:
    Heuristic vision is the PRIMARY and EXCLUSIVE classification engine
    for image-based civic complaint analysis.

    Requirements enforced:
    - Analyzed exclusively using the heuristic vision engine.
    - MobileNetV3 is NEVER invoked for classification in this flow.
    - Heuristic classification failures do NOT silently call MobileNetV3;
      instead, an explicit handled Other result with clear reason is returned.
    - Response schema preserves category, subcategory, priority, department,
      confidence, reason, source, predictionSource, and features.
    - Deployed configuration cannot override heuristic classification.
    """
    t0 = time.perf_counter()

    try:
        result = classify_heuristic_vision(
            image_path=image_path,
            description=description,
            request_id=request_id
        )
    except Exception as exc:
        print(f"[SmartCity Classifier Error] [{request_id}] Heuristic classification exception: {exc}")
        text = clean_text(description)
        is_hazard, hazard_priority, matched_hazards = detect_hazard_keywords(text)
        priority = hazard_priority if is_hazard else PRIORITY_MEDIUM
        reason = f"Heuristic vision analysis failed: {str(exc)}. Categorized as Other."
        if is_hazard:
            reason += f" Escalated to {priority} due to hazard indicators: [{', '.join(matched_hazards)}]."

        result = {
            "category": "Other",
            "subcategory": "General Issue",
            "predictedCategory": "Other",
            "predictedSubcategory": "General Issue",
            "priority": priority,
            "confidence": 0.50,
            "reason": reason,
            "department": get_department_for_category("Other"),
            "model": "SmartCity-HeuristicVision-v1.0",
            "modelVersion": None,
            "isFallback": True,
            "fallbackUsed": True,
            "fallbackReason": f"Heuristic exception: {str(exc)}",
            "modelLoaded": MOBILENET_LOADER.is_loaded,
            "source": "heuristic",
            "predictionSource": "heuristic",
            "top_predictions": [{"category": "Other", "confidence": 0.50}],
            "requestId": request_id,
            "error": str(exc),
        }

    result["inference_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result
