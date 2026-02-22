"""
Story Critic Loop - Phase 4 Upgrade (🔴 Research Grade)
Implements a self-refinement pipeline: Story → Critic → Improved Story

The critic is a second, independent Gemini call that:
  1. Reads the generated story
  2. Scores it on 3 axes: coherence, emotional depth, cultural accuracy
  3. Returns concrete, actionable improvement suggestions
  4. The suggestions are fed back into a regeneration call

Why this matters for interviews:
  "This is the same pattern used in RLHF critique loops and Constitutional AI —
   instead of just generating and hoping, the system audits its own output and
   explicitly improves it. The two-call architecture separates generation from
   evaluation, which is a core design principle in production AI pipelines."

Architecture:
  generate_story_from_images()
      │
      ▼
  critique_story()          ← NEW (this module)
      │
      ▼
  regenerate_with_critique()  ← NEW (this module)
      │
      ▼
  final improved story
"""

import logging
from .logging_config import log_execution_time

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Critique Dimensions
# ---------------------------------------------------
# Each dimension has a weight — used to compute a weighted overall score.
CRITIQUE_DIMENSIONS = {
    "coherence": {
        "weight": 0.40,
        "description": "Does the story flow logically? Are scenes connected smoothly?",
    },
    "emotional_depth": {
        "weight": 0.35,
        "description": "Does the story evoke genuine emotion? Are character feelings shown?",
    },
    "cultural_accuracy": {
        "weight": 0.25,
        "description": "Are the names, places, and customs authentically Bangladeshi?",
    },
}

# Minimum weighted score below which regeneration is triggered
REGENERATION_THRESHOLD = 70


# ---------------------------------------------------
# Critic Prompt Builder
# ---------------------------------------------------
def build_critic_prompt(story_text: str, style: str, scene_data: list) -> str:
    """
    Build the prompt sent to Gemini in the critic call.

    The critic is instructed to:
    - Score each dimension 1–10
    - Give a concrete one-sentence reason per score
    - List specific, actionable suggestions

    Returns:
        prompt string
    """
    scene_count = len(scene_data) if scene_data else "unknown"
    scene_summary = ", ".join(
        [f"Scene {s.get('image_number', i+1)}: {s.get('key_action', 'unknown action')}"
         for i, s in enumerate(scene_data[:5])]
    ) if scene_data else "no scene data"

    prompt = f"""
You are a professional literary critic and AI story evaluator.
You will evaluate the following {style} story that was generated from {scene_count} image scenes.

Source scenes summary: {scene_summary}

--- STORY TO EVALUATE ---
{story_text}
--- END STORY ---

Evaluate this story on exactly 3 dimensions. For each, give:
  - A score from 1 to 10
  - One sentence explaining the score
  - One concrete, specific improvement suggestion

Return your evaluation as a JSON object with this EXACT structure (no extra text):
{{
  "scores": {{
    "coherence": {{
      "score": <integer 1-10>,
      "reason": "<one sentence>",
      "suggestion": "<one concrete action to improve>"
    }},
    "emotional_depth": {{
      "score": <integer 1-10>,
      "reason": "<one sentence>",
      "suggestion": "<one concrete action to improve>"
    }},
    "cultural_accuracy": {{
      "score": <integer 1-10>,
      "reason": "<one sentence>",
      "suggestion": "<one concrete action to improve>"
    }}
  }},
  "overall_score": <weighted float, coherence×0.40 + emotional_depth×0.35 + cultural_accuracy×0.25, scaled to 100>,
  "needs_improvement": <true if overall_score < 70, else false>,
  "top_priority": "<name of the lowest-scoring dimension>",
  "improvement_summary": "<2-3 sentences summarising the most important changes to make>"
}}
"""
    logger.debug(f"Critic prompt built for style='{style}', scenes={scene_count}")
    return prompt


# ---------------------------------------------------
# Regeneration Prompt Builder
# ---------------------------------------------------
def build_regeneration_prompt(original_story: str, critique: dict, style: str) -> str:
    """
    Build the regeneration prompt that feeds critique back to Gemini.

    Args:
        original_story: the story that was critiqued
        critique:       dict from parse_critique_response()
        style:          story genre

    Returns:
        prompt string for the improved story
    """
    scores  = critique.get("scores", {})
    summary = critique.get("improvement_summary", "Improve the overall quality.")

    # Collect only the dimensions that scored below 8/10
    weak_dimensions = [
        (dim, data)
        for dim, data in scores.items()
        if data.get("score", 10) < 8
    ]

    improvement_lines = "\n".join(
        [f"  • {dim.upper()} (scored {data['score']}/10): {data['suggestion']}"
         for dim, data in weak_dimensions]
    ) if weak_dimensions else "  • Maintain current quality — minor polish only."

    prompt = f"""
You are rewriting a {style} story based on expert critique feedback.

--- ORIGINAL STORY ---
{original_story}
--- END ORIGINAL ---

CRITIC'S SUMMARY:
{summary}

SPECIFIC IMPROVEMENTS REQUIRED:
{improvement_lines}

RULES FOR REWRITE:
1. Keep the SAME characters, names, setting, and overall plot.
2. Keep the SAME genre ({style}) and all required style elements (e.g. [MORAL], [TWIST]).
3. Apply ONLY the improvements listed above — do not change what was already good.
4. The rewritten story must be the same length or slightly longer.
5. Output ONLY the rewritten story — no commentary, no preamble.
"""
    logger.debug(f"Regeneration prompt built — {len(weak_dimensions)} weak dimension(s) targeted")
    return prompt


# ---------------------------------------------------
# Critique Response Parser
# ---------------------------------------------------
@log_execution_time
def parse_critique_response(raw_response: str) -> dict:
    """
    Parse the Gemini critic's JSON response into a structured critique dict.

    Args:
        raw_response: raw string from the critic Gemini call

    Returns:
        Parsed critique dict, or error dict on failure
    """
    import json

    try:
        cleaned = raw_response.strip()
        # Strip markdown code fences if the model added them
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        critique = json.loads(cleaned)

        # Validate expected top-level keys
        required_keys = ["scores", "overall_score", "needs_improvement", "improvement_summary"]
        missing = [k for k in required_keys if k not in critique]
        if missing:
            logger.warning(f"Critique response missing keys: {missing}")

        # Ensure overall_score is capped at 100
        if "overall_score" in critique:
            critique["overall_score"] = min(100.0, float(critique["overall_score"]))

        logger.info(
            f"Critique parsed — overall: {critique.get('overall_score', 'N/A')}/100, "
            f"needs_improvement: {critique.get('needs_improvement', 'N/A')}, "
            f"top priority: {critique.get('top_priority', 'N/A')}"
        )
        return critique

    except Exception as e:
        logger.error(f"parse_critique_response failed: {e} | snippet: {raw_response[:200]}")
        return {
            "error": str(e),
            "raw":   raw_response,
            "scores": {},
            "overall_score": 0,
            "needs_improvement": False,
            "improvement_summary": "Critique parsing failed."
        }


# ---------------------------------------------------
# Public: Format Critique for UI
# ---------------------------------------------------
def format_critique_report(critique: dict) -> str:
    """
    Convert a critique dict into a readable markdown report for the UI.

    Args:
        critique: output from parse_critique_response()

    Returns:
        Markdown string
    """
    if "error" in critique:
        return f"⚠️ Critique failed: {critique['error']}"

    scores   = critique.get("scores", {})
    overall  = critique.get("overall_score", 0)
    priority = critique.get("top_priority", "N/A")
    summary  = critique.get("improvement_summary", "")
    improved = critique.get("needs_improvement", False)

    emoji_for_score = lambda s: "🟢" if s >= 8 else ("🟡" if s >= 5 else "🔴")

    score_lines = []
    for dim, data in scores.items():
        s     = data.get("score", 0)
        emoji = emoji_for_score(s)
        score_lines.append(
            f"| {emoji} **{dim.replace('_', ' ').title()}** "
            f"| {s}/10 "
            f"| {data.get('reason', '')} |"
        )

    status_badge = "🔴 Improvement triggered" if improved else "✅ Passed quality threshold"

    report = f"""
### 🔍 Story Critic Report

| Dimension | Score | Reason |
|-----------|-------|--------|
{chr(10).join(score_lines)}

**Overall Score:** `{overall:.1f}/100` — {status_badge}
**Top Priority:** `{priority}`

**Critic's Summary:**
> {summary}
"""
    return report.strip()
