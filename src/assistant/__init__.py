"""
LandslideGuard AI Assistant & Safety Module (Phase 8).

Provides:
- KnowledgeEngine: Context-aware retrieval engine answering questions based strictly on actual project data.
- SafetyGuidance: Multilingual safety guidance in English, Hindi, and Punjabi.
- TerrainCVAnalyzer: Experimental computer vision analysis for terrain images.
"""

from __future__ import annotations

from src.assistant.safety_guidance import get_safety_guidance, format_safety_markdown
from src.assistant.knowledge_engine import KnowledgeEngine, get_knowledge_engine
from src.geospatial.terrain_cv import TerrainCVAnalyzer, get_cv_analyzer

__all__ = [
    "get_safety_guidance",
    "format_safety_markdown",
    "KnowledgeEngine",
    "get_knowledge_engine",
    "TerrainCVAnalyzer",
    "get_cv_analyzer",
]

