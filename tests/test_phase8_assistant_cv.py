"""
Test Suite for Phase 8: LandslideGuard AI Assistant & Computer Vision.

Tests:
1. Knowledge engine retrieval accuracy and provenance citations.
2. Anti-hallucination guards (unknown districts, missing data, out-of-scope queries).
3. Multilingual safety guidance (English, Hindi, Punjabi) and formatters.
4. Computer vision feature extraction (tension cracks, bare soil exposure, deformation).
5. Computer vision input validation (corrupt bytes, oversized payloads, non-image files).
6. Phase 8 FastAPI REST API endpoints.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from src.assistant.knowledge_engine import KnowledgeEngine, get_knowledge_engine, ASSISTANT_DISCLAIMER
from src.assistant.safety_guidance import (
    get_safety_guidance,
    format_safety_markdown,
    normalize_language_code,
    get_supported_languages,
    SUPPORTED_LANGUAGES,
)
from src.geospatial.terrain_cv import (
    TerrainCVAnalyzer,
    get_cv_analyzer,
    CV_SCIENTIFIC_DISCLAIMER,
    MAX_IMAGE_BYTES,
)


def _run_coro_sync(coro):
    """Run an async coroutine synchronously without event loop conflicts."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    raise RuntimeError("Coroutine did not complete in one step.")


# ─────────────────────────────────────────────────────────────
# 1. Knowledge Engine Retrieval & Provenance Tests
# ─────────────────────────────────────────────────────────────

def test_knowledge_engine_retrieval_and_provenance():
    """Verify that the assistant answers core questions using actual project data with source citations."""
    engine = get_knowledge_engine()

    # Query 1: Active alerts
    r_alerts = engine.answer_query("Which locations have active experimental alerts?")
    assert r_alerts["intent"] == "ACTIVE_ALERTS"
    assert "active" in r_alerts["answer"].lower() and "alert" in r_alerts["answer"].lower()
    assert any("alerts" in src.lower() for src in r_alerts["sources"])
    assert r_alerts["is_experimental"] is True

    # Query 2: Data sources
    r_sources = engine.answer_query("What data sources does LandslideGuard use?")
    assert r_sources["intent"] == "DATA_SOURCES"
    assert "Open-Meteo" in r_sources["answer"]
    assert "NASA" in r_sources["answer"] or "Copernicus" in r_sources["answer"]
    assert len(r_sources["sources"]) >= 1

    # Query 3: Methodology
    r_method = engine.answer_query("How does the machine learning risk engine work?")
    assert r_method["intent"] == "METHODOLOGY_MODEL"
    assert "Logistic Regression" in r_method["answer"] or "PR-AUC" in r_method["answer"]
    assert "60%" in r_method["answer"] and "20%" in r_method["answer"]

    # Query 4: Rainfall information
    r_rain = engine.answer_query("What rainfall data is available?")
    assert r_rain["intent"] == "RAINFALL_WEATHER"
    assert "24-Hour Accumulation" in r_rain["answer"]
    assert "Open-Meteo" in r_rain["answer"]


def test_knowledge_engine_anti_hallucination_and_out_of_scope():
    """Verify anti-hallucination safeguards on unknown districts, empty queries, and out-of-scope topics."""
    engine = get_knowledge_engine()

    # Out of scope 1: Crypto / stocks
    r_crypto = engine.answer_query("What is the current price of Bitcoin?")
    assert r_crypto["intent"] == "OUT_OF_SCOPE"
    assert "cannot answer queries outside this domain" in r_crypto["answer"].lower()

    # Out of scope 2: Sports / pop culture
    r_sports = engine.answer_query("Who won the cricket world cup?")
    assert r_sports["intent"] == "OUT_OF_SCOPE"
    assert "specialized exclusively" in r_crypto["answer"].lower()

    # Unknown fictional district
    r_unknown = engine.answer_query("What is the landslide risk in Atlantis?")
    assert r_unknown["intent"] == "DISTRICT_LOOKUP"
    assert "Atlantis" in r_unknown["answer"]
    assert "was not found in LandslideGuard's India boundary database" in r_unknown["answer"]
    assert r_unknown["retrieved_facts"]["found"] is False

    # Empty query
    r_empty = engine.answer_query("   ")
    assert r_empty["intent"] == "EMPTY_QUERY"
    assert "Please ask a question" in r_empty["answer"]


def test_knowledge_engine_context_awareness_and_disclaimer():
    """Verify context-aware queries for specific districts and check mandatory disclaimers."""
    engine = get_knowledge_engine()

    # Context query for known district
    r_dist = engine.answer_query("What is the risk here?", context={"district": "Idukki", "state": "Kerala"})
    assert "Idukki" in r_dist["answer"]
    assert ASSISTANT_DISCLAIMER in r_dist["disclaimer"]

    # Why high risk explanation
    r_why = engine.answer_query("Why is a location classified as high risk?")
    assert r_why["intent"] == "RISK_EXPLANATION"
    assert "SHAP" in r_why["answer"]
    assert "24h Accumulated Rainfall" in r_why["answer"]
    assert "Causal Humility" in r_why["answer"]


# ─────────────────────────────────────────────────────────────
# 2. Multilingual Safety Guidance Tests
# ─────────────────────────────────────────────────────────────

def test_multilingual_safety_guidance_all_languages():
    """Verify completeness, verified authority, emergency contacts, and formatting for all 3 languages."""
    languages = ["en", "hi", "pa"]
    for lang in languages:
        guide = get_safety_guidance(lang)
        assert guide["language"] == lang
        assert "authority" in guide and "NDMA" in guide["authority"]
        assert "before" in guide and len(guide["before"]["warning_signs"]) >= 5
        assert "during" in guide and len(guide["during"]["indoor_actions"]) >= 2
        assert "after" in guide and len(guide["after"]["safety_rules"]) >= 4
        assert "emergency_contacts" in guide and len(guide["emergency_contacts"]) >= 5

        # Check critical emergency numbers present
        nums = [ec["number"] for ec in guide["emergency_contacts"]]
        assert any("112" in n for n in nums)
        assert any("1078" in n for n in nums)
        assert any("1070" in n for n in nums)

        # Test Markdown formatting
        md = format_safety_markdown(guide)
        assert len(md) > 300
        assert guide["title"] in md
        assert "112" in md


def test_safety_language_normalization_and_descriptors():
    """Verify language code normalization and supported language helper."""
    assert normalize_language_code("English") == "en"
    assert normalize_language_code("hindi") == "hi"
    assert normalize_language_code("हिन्दी") == "hi"
    assert normalize_language_code("Punjabi") == "pa"
    assert normalize_language_code("ਪੰਜਾਬੀ") == "pa"
    assert normalize_language_code("french") == "en"  # fallback to en
    assert normalize_language_code(None) == "en"

    supported = get_supported_languages()
    assert len(supported) == 3
    codes = [s["code"] for s in supported]
    assert "en" in codes and "hi" in codes and "pa" in codes


# ─────────────────────────────────────────────────────────────
# 3. Computer Vision Feature Analysis Tests
# ─────────────────────────────────────────────────────────────

def test_cv_analyzer_synthetic_cracks_and_scarp():
    """Verify crack detection, bare soil exposure, and composite scoring on synthetic slope image."""
    analyzer = get_cv_analyzer()

    # Generate synthetic slope with green grass, brown soil patch, and dark tension crack
    h, w = 240, 320
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = [34, 139, 34]  # Forest green
    arr[60:180, 80:240] = [140, 90, 45]  # Brown soil patch

    # Draw dark linear crack fissure
    for y in range(80, 160):
        x = int(150 + 8 * np.sin(y / 10.0))
        arr[y, max(0, x-1):min(w, x+2)] = [15, 10, 5]

    res = analyzer.analyze(Image.fromarray(arr))
    assert res["available"] is True
    assert res["crack_detection"]["crack_candidate_count"] >= 1
    assert res["surface_composition"]["soil_exposure_ratio"] > 0.05
    assert res["composite_assessment"]["anomaly_score"] > 0.0
    assert "disclaimer" in res and CV_SCIENTIFIC_DISCLAIMER in res["disclaimer"]
    assert "overlay_base64" in res and len(res["overlay_base64"]) > 50


def test_cv_analyzer_stable_dense_forest():
    """Verify that a uniform dense forest image produces normal/stable appearance and zero cracks."""
    analyzer = get_cv_analyzer()

    h, w = 180, 240
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = [40, 150, 40]  # Uniform green canopy

    res = analyzer.analyze(Image.fromarray(arr))
    assert res["available"] is True
    assert res["crack_detection"]["crack_candidate_count"] == 0
    assert res["surface_composition"]["vegetation_cover_ratio"] > 0.90
    assert res["composite_assessment"]["anomaly_tier"] == "Normal / Stable Appearance"


def test_cv_analyzer_validation_and_error_handling():
    """Verify input validation, corrupted file rejection, and oversized payload guards."""
    analyzer = get_cv_analyzer()

    # Empty payload
    r_empty = analyzer.analyze(b"")
    assert r_empty["available"] is False
    assert "empty" in r_empty["error"].lower()

    # Corrupted / invalid image bytes
    r_corrupt = analyzer.analyze(b"NOT_AN_IMAGE_STRING_BYTES_12345")
    assert r_corrupt["available"] is False
    assert "failed to decode" in r_corrupt["error"].lower()

    # Oversized payload guard (> 10 MB)
    oversized_bytes = b"0" * (MAX_IMAGE_BYTES + 1024)
    r_large = analyzer.analyze(oversized_bytes)
    assert r_large["available"] is False
    assert "exceeds maximum allowed limit" in r_large["error"]


# ─────────────────────────────────────────────────────────────
# 4. Phase 8 FastAPI Endpoint Tests
# ─────────────────────────────────────────────────────────────

def test_phase8_fastapi_endpoints():
    """Verify Phase 8 REST endpoints in api.py: query_assistant, analyze_terrain_image, get_safety_guidance."""
    from api import (
        query_assistant,
        analyze_terrain_image,
        get_safety_guidance_endpoint,
        AssistantQueryRequest,
        CVAnalysisRequest,
    )

    # 1. Assistant query endpoint
    req_q = AssistantQueryRequest(query="What data sources are integrated?", language="en")
    res_q = _run_coro_sync(query_assistant(req_q))
    assert res_q["intent"] == "DATA_SOURCES"
    assert "Open-Meteo" in res_q["answer"]
    assert len(res_q["sources"]) >= 1

    # 2. Safety guidance endpoint
    lang_val = "hi"
    res_safety = _run_coro_sync(get_safety_guidance_endpoint(language=lang_val))
    assert res_safety["language"] == "hi"
    assert "राष्ट्रीय आपदा प्रबंधन प्राधिकरण" in res_safety["authority"]
    assert len(res_safety["emergency_contacts"]) >= 5

    # 3. CV analysis endpoint
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :] = [30, 140, 30]
    bio = io.BytesIO()
    Image.fromarray(arr).save(bio, format="PNG")
    b64_str = base64.b64encode(bio.getvalue()).decode("ascii")

    req_cv = CVAnalysisRequest(image_base64=b64_str, filename="test_slope.png")
    res_cv = _run_coro_sync(analyze_terrain_image(req_cv))
    assert res_cv["available"] is True
    assert "composite_assessment" in res_cv
    assert res_cv["composite_assessment"]["anomaly_tier"] == "Normal / Stable Appearance"
