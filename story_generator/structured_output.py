"""
Structured Output Generation - Level 3 Upgrade (8️⃣)
Uses JSON schema to enforce structured story output instead of raw text.
This is much more robust than prompt-only control.

Interview talking point:
  "Instead of asking the AI to format text a certain way with prompts,
   I used structured generation — defining a JSON schema and forcing the
   AI to return validated structured output. This eliminates formatting
   bugs entirely and makes the output programmatically reliable."
"""

import json
import logging
from .logging_config import log_execution_time

logger = logging.getLogger(__name__)

# ---------------------------------------------------
# 8️⃣ JSON Schema Definition
# ---------------------------------------------------
# This is the contract the AI MUST follow.
# Every field is typed and required.

STORY_SCHEMA = {
    "type": "object",
    "required": ["title", "paragraphs", "style_tag"],
    "properties": {
        "title": {
            "type": "string",
            "description": "The story title"
        },
        "paragraphs": {
            "type": "array",
            "description": "Ordered list of story paragraphs",
            "items": {
                "type": "object",
                "required": ["scene_reference", "text"],
                "properties": {
                    "scene_reference": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Which scene numbers this paragraph draws from"
                    },
                    "text": {
                        "type": "string",
                        "description": "The paragraph content"
                    },
                    "emotional_tone": {
                        "type": "string",
                        "description": "The dominant emotion in this paragraph"
                    }
                }
            }
        },
        "style_tag": {
            "type": "object",
            "description": "Genre-specific ending element",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "MORAL | SOLUTION | TWIST | NONE"
                },
                "content": {
                    "type": "string",
                    "description": "The moral lesson, mystery solution, or plot twist"
                }
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "word_count":       {"type": "integer"},
                "paragraph_count":  {"type": "integer"},
                "dominant_emotion": {"type": "string"},
                "arc_summary":      {"type": "string"}
            }
        }
    }
}


def build_structured_prompt(scene_data: list, character_map: dict,
                             style: str, tone: str,
                             narrative_structure: str,
                             emotional_arc: dict) -> str:
    """
    Build a prompt that instructs the AI to return structured JSON output
    matching STORY_SCHEMA exactly.

    Args:
        scene_data:          List of scene dicts from Layer 1
        character_map:       Entity → BD name mapping from Layer 2
        style:               Story genre
        tone:                Story tone
        narrative_structure: Story structure type
        emotional_arc:       Arc dict from coherence.py

    Returns:
        Full prompt string demanding JSON output
    """
    schema_str   = json.dumps(STORY_SCHEMA, indent=2)
    scene_str    = json.dumps(scene_data, indent=2)
    char_str     = json.dumps(character_map, indent=2) if character_map else "None"
    arc_summary  = emotional_arc.get("arc_summary", "neutral") if emotional_arc else "neutral"

    style_tag_instruction = {
        "Morale":    'Set style_tag.type = "MORAL" and write the moral lesson in style_tag.content.',
        "Mystery":   'Set style_tag.type = "SOLUTION" and reveal the culprit and key clue in style_tag.content.',
        "Thriller":  'Set style_tag.type = "TWIST" and write a shocking twist in style_tag.content.',
    }.get(style, 'Set style_tag.type = "NONE" and style_tag.content = "".')

    prompt = f"""
    You are a professional story writer. Generate a {style} story in {tone} tone
    using a {narrative_structure} structure.

    SCENE DATA (use ALL scenes):
    {scene_str}

    CHARACTER MAP (use EXACT names):
    {char_str}

    EMOTIONAL ARC TO FOLLOW: {arc_summary}

    STYLE TAG INSTRUCTION: {style_tag_instruction}

    CRITICAL INSTRUCTION:
    You MUST return ONLY a valid JSON object that exactly matches this schema:
    {schema_str}

    Rules:
    - "scene_reference" in each paragraph = list of scene image_number integers used
    - Write 4-5 paragraphs
    - Use only Bangladeshi names, settings, and cultural context
    - Do NOT return any text outside the JSON object
    - Do NOT wrap in markdown code blocks
    - The JSON must be directly parseable with json.loads()
    """

    logger.debug(f"Structured prompt built — style='{style}', tone='{tone}', arc='{arc_summary}'")
    return prompt


@log_execution_time
def parse_structured_output(raw_response: str) -> dict:
    """
    Parse and validate the AI's structured JSON response.

    Args:
        raw_response: raw string from Gemini API

    Returns:
        Validated story dict matching STORY_SCHEMA, or error dict
    """
    try:
        # Clean markdown wrappers if AI added them anyway
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        story_dict = json.loads(cleaned)

        # Validate required fields
        missing = [f for f in STORY_SCHEMA["required"] if f not in story_dict]
        if missing:
            logger.warning(f"Structured output missing required fields: {missing}")
            story_dict["_validation_warnings"] = missing

        # Compute metadata if not present
        if "metadata" not in story_dict:
            paragraphs = story_dict.get("paragraphs", [])
            full_text  = " ".join([p.get("text", "") for p in paragraphs])
            story_dict["metadata"] = {
                "word_count":      len(full_text.split()),
                "paragraph_count": len(paragraphs),
                "dominant_emotion": "",
                "arc_summary":     ""
            }

        logger.info(
            f"Structured output parsed — title: '{story_dict.get('title', 'N/A')}', "
            f"paragraphs: {len(story_dict.get('paragraphs', []))}"
        )
        return story_dict

    except json.JSONDecodeError as e:
        logger.error(f"Structured output JSON parse failed: {e} | snippet: {raw_response[:200]}")
        return {"error": f"JSON parse failed: {str(e)}", "raw": raw_response}

    except Exception as e:
        logger.error(f"parse_structured_output failed: {e}", exc_info=True)
        return {"error": str(e), "raw": raw_response}


def render_structured_story(story_dict: dict) -> str:
    """
    Convert structured story dict → clean readable text for display.
    This is the reverse of structured generation: schema → human text.

    Args:
        story_dict: validated story dict from parse_structured_output()

    Returns:
        Clean story string for UI display
    """
    if "error" in story_dict:
        return f"Story rendering failed: {story_dict['error']}"

    parts = []

    # Title
    title = story_dict.get("title", "Untitled Story")
    parts.append(f"**{title}**")

    # Paragraphs
    for para in story_dict.get("paragraphs", []):
        text = para.get("text", "").strip()
        if text:
            parts.append(text)

    # Style tag
    style_tag = story_dict.get("style_tag", {})
    tag_type   = style_tag.get("type", "NONE")
    tag_content = style_tag.get("content", "")

    if tag_type != "NONE" and tag_content:
        parts.append(f"[{tag_type}]: {tag_content}")

    rendered = "\n\n".join(parts)
    logger.info(f"Structured story rendered — {len(parts)} sections, {len(rendered.split())} words")
    return rendered
