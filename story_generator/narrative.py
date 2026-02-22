"""
Narrative Generation Layer - Layer 3
Professional-grade story generation with:
  5️⃣ Emotional arc / scene coherence
  6️⃣ Hallucination reduction (citation enforcement)
  7️⃣ Multimodal memory (session continuity)
  8️⃣ Structured JSON output mode
  4️⃣ Self-healing failure recovery
  🔴 Story Critic Loop (Phase 4)
"""

from dotenv import load_dotenv
import os
from google import genai
import json
import logging
from .character import extract_main_characters
from .monitoring import token_monitor
from .logging_config import log_execution_time
from .coherence import build_emotional_arc, build_arc_instructions, score_story_coherence
from .hallucination import build_citation_prompt, parse_cited_story, strip_citation_tags, generate_hallucination_report
from .memory import story_memory, build_memory_context
from .structured_output import build_structured_prompt, parse_structured_output, render_structured_story
from .critic import (
    build_critic_prompt, parse_critique_response,
    build_regeneration_prompt, format_critique_report,
    REGENERATION_THRESHOLD
)

logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.critical("GOOGLE_API_KEY not found in environment variables")
    raise ValueError("API key not found")

client = genai.Client(api_key=api_key)
logger.info("Narrative layer initialized successfully")

# ---------------------------------------------------
# Prompt Builder
# ---------------------------------------------------
def create_advanced_prompt(style, tone="Neutral", narrative_structure="Linear storytelling"):
    """Enhanced prompt builder with tone control and narrative structure"""
    base_prompt = f"""
    **Your Persona:** You are a friendly and engaging storyteller.
    **Main Goal:** Write a story in simple, clear, and modern English.
    **Task:** Create one single story that connects all the provided scenes in order.
    **Style Requirement:** The story must fit the '{style}' genre.
    """
    tone_instructions = {
        "Calm":      "Use peaceful, gentle language. Focus on harmony and tranquility. Avoid sudden conflicts.",
        "Dark":      "Use mysterious, intense language. Include shadows, uncertainty, and deeper emotions.",
        "Energetic": "Use dynamic, exciting language. Include action words, enthusiasm, and vibrant descriptions.",
        "Neutral":   "Use balanced, clear language suitable for all audiences."
    }
    tone_prompt = f"**Tone Requirement:** {tone_instructions.get(tone, tone_instructions['Neutral'])}"

    structure_instructions = {
        "Linear storytelling":   "Tell the story in chronological order from beginning to end.",
        "Hero's Journey":        "Follow the hero's journey: ordinary world → call to adventure → challenges → transformation → return with wisdom.",
        "Flashback-based":       "Start with the climax or ending, then reveal how we got there through flashbacks.",
        "Mystery reveal format": "Build suspense gradually, revealing clues throughout, with the major revelation at the end."
    }
    structure_prompt = f"**Narrative Structure:** {structure_instructions.get(narrative_structure, structure_instructions['Linear storytelling'])}"

    core_instructions = """
    **Core Instructions:**
    1. Tell ONE coherent story with beginning, middle, and end.
    2. Use key details from EVERY scene.
    3. Maintain character consistency across scenes.
    4. The scenes are in chronological order.
    5. Use only Bangladeshi names, characters, and places.

    **Output Format:**
    - Title at the top
    - 4 to 5 well-structured paragraphs
    """

    style_instruction = ""
    if style == "Morale":
        style_instruction = "\n    After the story, add:\n    [MORAL]: followed by a single-sentence moral."
    elif style == "Mystery":
        style_instruction = "\n    After the story, add:\n    [SOLUTION]: Reveal the culprit and key clue."
    elif style == "Thriller":
        style_instruction = "\n    After the story, add:\n    [TWIST]: Add a shocking final twist."

    logger.debug(f"Built prompt for style='{style}', tone='{tone}', structure='{narrative_structure}'")
    return base_prompt + "\n\n" + tone_prompt + "\n\n" + structure_prompt + "\n\n" + core_instructions + style_instruction


# ---------------------------------------------------
# Validation
# ---------------------------------------------------
def validate_scene_data(scene_data):
    """Validates scene data. Returns (is_valid: bool, message: str)"""
    if not scene_data:
        logger.warning("validate_scene_data: empty scene_data")
        return False, "No scene data provided"
    if isinstance(scene_data, dict) and "error" in scene_data:
        logger.error(f"validate_scene_data: error in scene data → {scene_data.get('error')}")
        return False, f"Scene extraction error: {scene_data.get('error', 'Unknown error')}"
    if not isinstance(scene_data, list):
        logger.error(f"validate_scene_data: expected list, got {type(scene_data)}")
        return False, "Scene data must be a list of scenes"
    if len(scene_data) == 0:
        return False, "No scenes found in the data"
    required_fields = ["setting", "characters", "key_action"]
    for i, scene in enumerate(scene_data):
        if not isinstance(scene, dict):
            return False, f"Scene {i+1} is not properly formatted"
        missing = [f for f in required_fields if not scene.get(f)]
        if len(missing) == len(required_fields):
            return False, f"Scene {i+1} is missing critical information"
    logger.info(f"validate_scene_data: {len(scene_data)} scene(s) passed")
    return True, "Valid scene data"


def validate_style_requirements(story_text, style):
    """Validates story meets genre requirements. Returns validation dict."""
    validation = {"valid": True, "missing_elements": []}
    style_requirements = {
        "Mystery":    ["[SOLUTION]", "mystery", "clue"],
        "Thriller":   ["[TWIST]", "suspense"],
        "Morale":     ["[MORAL]", "lesson"],
        "Comedy":     ["humor", "funny", "laugh"],
        "Sci-Fi":     ["technology", "future", "science"],
        "Adventure":  ["journey", "quest", "adventure"],
        "Fairy Tale": ["once upon", "magical", "fairy"]
    }
    if style in style_requirements:
        story_lower = story_text.lower()
        for element in style_requirements[style]:
            if element.startswith("[") and element.endswith("]"):
                if element not in story_text:
                    validation["missing_elements"].append(element)
            else:
                if element not in story_lower:
                    validation["missing_elements"].append(element)
    if validation["missing_elements"]:
        validation["valid"] = False
        logger.warning(f"Style validation failed for '{style}' — missing: {validation['missing_elements']}")
    else:
        logger.info(f"Style validation passed for '{style}'")
    return validation


# ---------------------------------------------------
# Prompt Assembly  (now injects arc + citation + memory)
# ---------------------------------------------------
def build_contextual_prompt(scene_data, character_map, style, tone,
                             narrative_structure, emotional_arc=None,
                             memory_entry=None, enable_citations=False):
    """
    Assembles the full generation prompt with optional:
      - 5️⃣ Emotional arc instructions
      - 6️⃣ Citation enforcement
      - 7️⃣ Memory context
    """
    structured_context = f"""
    SCENE ANALYSIS:
    The following scenes have been extracted from images in chronological order:
    {json.dumps(scene_data, indent=2)}
    Your task is to weave these scenes into one cohesive narrative.
    """

    character_instruction = ""
    if character_map:
        character_instruction = f"""
        CHARACTER CONSISTENCY:
        {json.dumps(character_map, indent=2)}
        STRICT RULE: Use ONLY these assigned names.
        """

    story_guidance = create_advanced_prompt(style, tone, narrative_structure)

    quality_instructions = """
    QUALITY REQUIREMENTS:
    ✓ Every uploaded scene must be referenced in the story
    ✓ Maintain logical flow between scenes
    ✓ Keep cultural context (Bangladeshi setting)
    ✓ Ensure age-appropriate content
    ✓ Create emotional engagement
    """

    # 5️⃣ Emotional arc injection
    arc_instructions = ""
    if emotional_arc:
        arc_instructions = build_arc_instructions(emotional_arc)

    # 6️⃣ Citation enforcement injection
    citation_instructions = ""
    if enable_citations:
        citation_instructions = build_citation_prompt(scene_data)

    # 7️⃣ Memory context injection
    memory_instructions = ""
    if memory_entry:
        memory_instructions = build_memory_context(memory_entry)

    logger.debug("Contextual prompt assembled with arc/citation/memory injections")
    return "\n\n".join(filter(bool, [
        memory_instructions,
        structured_context,
        character_instruction,
        arc_instructions,
        citation_instructions,
        story_guidance,
        quality_instructions
    ]))


# ---------------------------------------------------
# Self-Healing
# ---------------------------------------------------
def build_correction_prompt(original_story, missing_elements, style):
    """Builds targeted correction prompt for self-healing."""
    missing_str = ", ".join(missing_elements)
    logger.info(f"Building correction prompt — missing: {missing_str}")
    return f"""
    The following story is MISSING required elements for the '{style}' genre:
    --- ORIGINAL STORY ---
    {original_story}
    --- END ---
    MISSING ELEMENTS: {missing_str}
    Rewrite the story including all missing elements.
    { '[MORAL]: Add a clear moral lesson at the end.' if '[MORAL]' in missing_elements else '' }
    { '[SOLUTION]: Add a mystery solution revealing culprit and clue.' if '[SOLUTION]' in missing_elements else '' }
    { '[TWIST]: Add a shocking plot twist.' if '[TWIST]' in missing_elements else '' }
    Keep characters and setting the same. Return the complete improved story.
    """


def post_process_story(raw_story_text, style):
    """Post-processes generated story text."""
    try:
        cleaned_story = raw_story_text.strip()
        lines = cleaned_story.split('\n')
        processed_lines = [line.strip() for line in lines if line.strip()]
        final_story = '\n\n'.join(processed_lines)
        style_validation = validate_style_requirements(final_story, style)
        result = {
            "story":           final_story,
            "validation":      style_validation,
            "word_count":      len(final_story.split()),
            "paragraph_count": len([p for p in final_story.split('\n\n') if p.strip()])
        }
        logger.info(f"Post-processing: {result['word_count']} words, {result['paragraph_count']} paragraphs")
        return result
    except Exception as e:
        logger.error(f"post_process_story failed: {str(e)}")
        return {"story": raw_story_text, "validation": {"valid": False, "error": str(e)},
                "word_count": 0, "paragraph_count": 0}


def attempt_correction(original_story, missing_elements, style, max_attempts=2):
    """Self-healing: auto-regenerates story when validation fails."""
    logger.warning(f"Self-healing triggered — missing: {missing_elements}")
    for attempt in range(1, max_attempts + 1):
        try:
            correction_prompt = build_correction_prompt(original_story, missing_elements, style)
            response = client.models.generate_content(model="gemini-2.5-flash", contents=correction_prompt)
            token_monitor.add_usage(correction_prompt, response.text)
            corrected_result = post_process_story(response.text, style)
            if corrected_result["validation"]["valid"]:
                logger.info(f"✅ Self-healing succeeded on attempt {attempt}")
                return corrected_result["story"], True
            missing_elements = corrected_result["validation"]["missing_elements"]
            logger.warning(f"Attempt {attempt} still missing: {missing_elements}")
        except Exception as e:
            logger.error(f"Correction attempt {attempt} failed: {str(e)}")
    logger.error("Self-healing exhausted all attempts.")
    return original_story, False


# ---------------------------------------------------
# Main Generation Function
# ---------------------------------------------------
@log_execution_time
def generate_story_from_images(scene_data, style, tone="Neutral",
                                narrative_structure="Linear storytelling",
                                session_id=None, enable_citations=False,
                                use_structured_output=False):
    """
    Master story generation function.

    New parameters:
        session_id:            str — enables 7️⃣ memory/continuation
        enable_citations:      bool — enables 6️⃣ hallucination reduction
        use_structured_output: bool — enables 8️⃣ JSON schema output

    Returns:
        story string (always clean text, tags stripped internally)
    """
    logger.info(
        f"generate_story_from_images — style='{style}', tone='{tone}', "
        f"citations={enable_citations}, structured={use_structured_output}, "
        f"session='{session_id}'"
    )

    is_valid, validation_message = validate_scene_data(scene_data)
    if not is_valid:
        logger.error(f"Input validation failed: {validation_message}")
        return f"Story Generation Error: {validation_message}"

    try:
        # Layer 2: Gender-aware character mapping
        character_map = extract_main_characters(scene_data)
        logger.info(f"Character map: {character_map}")

        # 5️⃣ Build emotional arc
        emotional_arc = build_emotional_arc(scene_data)
        logger.info(f"Emotional arc: {emotional_arc.get('arc_summary')}")

        # 7️⃣ Load memory if session exists
        memory_entry = None
        if session_id:
            memory_entry = story_memory.load(session_id)
            if memory_entry:
                logger.info(f"Memory context loaded for session '{session_id}'")

        # 8️⃣ Structured output mode
        if use_structured_output:
            logger.info("Using structured JSON output mode (8️⃣)")
            prompt = build_structured_prompt(
                scene_data, character_map, style, tone,
                narrative_structure, emotional_arc
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            usage_stats = token_monitor.add_usage(prompt, response.text)
            logger.info(f"Structured gen tokens: {usage_stats['input_tokens']}→{usage_stats['output_tokens']}")

            story_dict   = parse_structured_output(response.text)
            final_story  = render_structured_story(story_dict)

            # Save to memory
            if session_id:
                story_memory.save(session_id, scene_data, character_map, final_story, emotional_arc)

            return final_story

        # Standard mode with optional citation + arc + memory
        final_prompt = build_contextual_prompt(
            scene_data, character_map, style, tone, narrative_structure,
            emotional_arc=emotional_arc,
            memory_entry=memory_entry,
            enable_citations=enable_citations
        )

        logger.info("Sending prompt to Gemini API...")
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=final_prompt
        )
        usage_stats = token_monitor.add_usage(final_prompt, response.text)
        logger.info(
            f"Generation complete — tokens: {usage_stats['input_tokens']}→"
            f"{usage_stats['output_tokens']}, cost: ${usage_stats['total_cost']:.5f}"
        )

        raw_story = response.text

        # 6️⃣ Parse and validate citations if enabled
        if enable_citations:
            parsed       = parse_cited_story(raw_story, len(scene_data))
            h_report     = generate_hallucination_report(parsed)
            logger.info(f"Hallucination report: {h_report['summary']}")
            raw_story    = strip_citation_tags(raw_story)

        # Post-process and validate
        processed_result = post_process_story(raw_story, style)

        # Self-healing if validation failed
        if not processed_result["validation"]["valid"]:
            missing = processed_result["validation"]["missing_elements"]
            logger.warning(f"Validation failed. Missing: {missing}. Triggering self-healing...")
            corrected_story, was_corrected = attempt_correction(
                processed_result["story"], missing, style
            )
            final_story = corrected_story
            if was_corrected:
                logger.info("Story corrected by self-healing layer")
            else:
                logger.warning("Self-healing could not fix story. Using best available.")
        else:
            final_story = processed_result["story"]

        # 7️⃣ Save to memory after successful generation
        if session_id:
            story_memory.save(session_id, scene_data, character_map, final_story, emotional_arc)
            logger.info(f"Story saved to memory for session '{session_id}'")

        logger.info("Story generation pipeline completed successfully")
        return final_story

    except Exception as e:
        logger.error(f"generate_story_from_images failed: {str(e)}", exc_info=True)
        return f"Story Generation Error: Failed to generate story - {str(e)}"


# ---------------------------------------------------
#  Phase 4 — Story Critic Loop
# ---------------------------------------------------
@log_execution_time
def generate_story_with_critic(scene_data: list, style: str, tone: str = "Neutral",
                                narrative_structure: str = "Linear storytelling",
                                session_id: str = None, enable_citations: bool = False) -> dict:
    """
    Full critic loop pipeline: Story → Critic → Improved Story

    Architecture:
        1. Generate initial story (standard pipeline)
        2. Send story to Gemini critic for scoring (coherence, emotional depth, cultural accuracy)
        3. If overall score < REGENERATION_THRESHOLD (70): regenerate using suggestions
        4. Return both the final story AND the critique report

    Interview talking point:
        "This is a self-refinement loop — the same pattern behind Constitutional AI
         and RLHF critique pipelines. Generation and evaluation are two separate
         model calls, which means you can swap in any evaluator without touching
         the generator. That separation of concerns is what makes it production-grade."

    Args:
        scene_data:          List of scene dicts from vision layer
        style:               Story genre
        tone:                Story tone
        narrative_structure: Narrative structure type
        session_id:          Optional session ID for memory
        enable_citations:    Whether to enforce scene citations

    Returns:
        {
          "final_story":     str  — best story (improved if critique triggered regen)
          "initial_story":   str  — original story before critique
          "critique":        dict — full critique from parse_critique_response()
          "critique_report": str  — formatted markdown report for UI
          "was_improved":    bool — True if regeneration was triggered
          "critic_score":    float — overall score out of 100
        }
    """
    logger.info(f"🔴 Critic loop started — style='{style}', tone='{tone}'")

    # ── Step 1: Generate initial story using the standard pipeline ──────────
    initial_story = generate_story_from_images(
        scene_data, style, tone, narrative_structure,
        session_id=session_id,
        enable_citations=enable_citations,
        use_structured_output=False      # Critic loop always works on plain text
    )

    if "Error" in str(initial_story):
        logger.error(f"Critic loop aborted — initial story generation failed: {initial_story}")
        return {
            "final_story":     initial_story,
            "initial_story":   initial_story,
            "critique":        {},
            "critique_report": "⚠️ Critic loop skipped — story generation failed.",
            "was_improved":    False,
            "critic_score":    0.0,
        }

    # ── Step 2: Critic call ─────────────────────────────────────────────────
    logger.info("Sending story to critic for evaluation...")
    critic_prompt = build_critic_prompt(initial_story, style, scene_data)

    try:
        critic_response = client.models.generate_content(
            model="gemini-2.5-flash", contents=critic_prompt
        )
        token_monitor.add_usage(critic_prompt, critic_response.text)
        critique = parse_critique_response(critic_response.text)
    except Exception as e:
        logger.error(f"Critic call failed: {e}", exc_info=True)
        return {
            "final_story":     initial_story,
            "initial_story":   initial_story,
            "critique":        {},
            "critique_report": f"⚠️ Critic call failed: {e}",
            "was_improved":    False,
            "critic_score":    0.0,
        }

    overall_score = critique.get("overall_score", 100.0)
    needs_improvement = critique.get("needs_improvement", False)
    critique_report = format_critique_report(critique)

    logger.info(
        f"Critic score: {overall_score:.1f}/100 — "
        f"needs_improvement: {needs_improvement} (threshold: {REGENERATION_THRESHOLD})"
    )

    # ── Step 3: Regenerate if score is below threshold ───────────────────────
    if needs_improvement:
        logger.info("🔄 Regeneration triggered — applying critic suggestions...")
        regen_prompt = build_regeneration_prompt(initial_story, critique, style)

        try:
            regen_response = client.models.generate_content(
                model="gemini-2.5-flash", contents=regen_prompt
            )
            token_monitor.add_usage(regen_prompt, regen_response.text)

            improved_result = post_process_story(regen_response.text, style)
            final_story = improved_result["story"]
            was_improved = True
            logger.info("✅ Critic-driven regeneration complete")
        except Exception as e:
            logger.error(f"Regeneration call failed: {e}", exc_info=True)
            final_story = initial_story
            was_improved = False
    else:
        logger.info("✅ Story passed critic threshold — no regeneration needed")
        final_story = initial_story
        was_improved = False

    # ── Step 4: Update memory with the final (possibly improved) story ───────
    if session_id:
        character_map = extract_main_characters(scene_data)
        emotional_arc = build_emotional_arc(scene_data)
        story_memory.save(session_id, scene_data, character_map, final_story, emotional_arc)
        logger.info(f"Critic-loop story saved to memory for session '{session_id}'")

    return {
        "final_story":     final_story,
        "initial_story":   initial_story,
        "critique":        critique,
        "critique_report": critique_report,
        "was_improved":    was_improved,
        "critic_score":    overall_score,
    }