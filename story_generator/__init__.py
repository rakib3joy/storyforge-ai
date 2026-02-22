# story_generator package - v3.0.0
from .logging_config import setup_logging
from .vision import extract_scene_from_images
from .character import extract_main_characters
from .narrative import generate_story_from_images, generate_story_with_critic
from .tts import narrate_story
from .monitoring import token_monitor
from .coherence import build_emotional_arc, score_story_coherence
from .hallucination import parse_cited_story, generate_hallucination_report, strip_citation_tags
from .memory import story_memory, build_memory_context
from .structured_output import parse_structured_output, render_structured_story, STORY_SCHEMA
from .streaming import stream_story_token_level, stream_story_ui_level, StreamingMode, get_streaming_explanation
from .research import run_attention_analysis, format_analysis_report
from .critic import format_critique_report, REGENERATION_THRESHOLD

# Initialize logging as soon as package is imported
setup_logging()

__version__ = "3.0.0"
__all__ = [
    # Core pipeline
    "extract_scene_from_images",
    "extract_main_characters",
    "generate_story_from_images",
    "generate_story_with_critic",
    "narrate_story",
    "token_monitor",
    # Level 2
    "build_emotional_arc",
    "score_story_coherence",
    "parse_cited_story",
    "generate_hallucination_report",
    "strip_citation_tags",
    "story_memory",
    "build_memory_context",
    # Level 3
    "parse_structured_output",
    "render_structured_story",
    "STORY_SCHEMA",
    "stream_story_token_level",
    "stream_story_ui_level",
    "StreamingMode",
    "get_streaming_explanation",
    "run_attention_analysis",
    "format_analysis_report",
    # Level 4 — Critic Loop
    "format_critique_report",
    "REGENERATION_THRESHOLD",
    # Utilities
    "setup_logging",
]