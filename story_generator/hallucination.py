"""
Hallucination Reduction Layer - Level 2 Upgrade (6️⃣)
Forces the AI to cite which scene each paragraph is based on.
This creates semi-structured storytelling with source attribution.

Interview talking point:
  "I implemented a hallucination reduction layer that forces the AI to cite
   which image/scene each paragraph references. If a paragraph can't be traced
   back to a scene, it's flagged as a potential hallucination. That's
   semi-structured storytelling with source attribution."
"""

import json
import logging
from .logging_config import log_execution_time

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# 6️⃣ Citation-Enforced Prompt Builder
# ---------------------------------------------------
def build_citation_prompt(scene_data: list) -> str:
    """
    Builds an instruction block that forces the AI to cite scenes per paragraph.
    Injected into the main generation prompt.

    Args:
        scene_data: list of scene dicts

    Returns:
        instruction string
    """
    scene_ids = [f"Scene {s.get('image_number', i+1)}" for i, s in enumerate(scene_data)]
    scene_list_str = ", ".join(scene_ids)

    instruction = f"""
    HALLUCINATION REDUCTION — SCENE CITATION REQUIRED (6️⃣):
    Available scenes: {scene_list_str}

    STRICT RULE: Every paragraph you write MUST start with a scene citation tag.

    Format each paragraph EXACTLY like this:
    [SCENE:1] First paragraph text here...
    [SCENE:2] Second paragraph text here...
    [SCENE:1,2] Paragraph combining scenes 1 and 2...

    IMPORTANT:
    - You may ONLY reference scenes that exist in the list above.
    - Do NOT invent details that cannot be traced to any scene.
    - The title line does NOT need a citation tag.
    - The [MORAL], [SOLUTION], or [TWIST] tag also does NOT need a citation.
    """

    logger.debug(f"Citation prompt built for {len(scene_data)} scenes: {scene_ids}")
    return instruction


# ---------------------------------------------------
# Citation Parser
# ---------------------------------------------------
def parse_cited_story(raw_story: str, scene_count: int) -> dict:
    """
    Parse a story that contains [SCENE:N] citation tags.

    Args:
        raw_story:    The AI-generated story with citation tags
        scene_count:  Total number of scenes (for validation)

    Returns:
        {
          "title":       "Story Title",
          "paragraphs":  [
              {"text": "...", "citations": [1, 2], "valid": True},
              ...
          ],
          "uncited_paragraphs":  [...],   # potential hallucinations
          "citation_coverage":   95,      # % of paragraphs properly cited
          "hallucination_flags": [...]    # paragraphs with no valid scene reference
        }
    """
    import re

    lines  = [l.strip() for l in raw_story.strip().split('\n') if l.strip()]
    result = {
        "title":               "",
        "paragraphs":          [],
        "uncited_paragraphs":  [],
        "citation_coverage":   0,
        "hallucination_flags": []
    }

    # First non-tag line = title
    for line in lines:
        if not line.startswith('[SCENE:') and not line.startswith('[MORAL') \
                and not line.startswith('[SOLUTION') and not line.startswith('[TWIST'):
            result["title"] = line
            break

    valid_scenes   = set(range(1, scene_count + 1))
    cited_count    = 0
    total_paras    = 0

    for line in lines:
        # Skip title and style tags
        if line == result["title"]:
            continue
        if line.startswith('[MORAL') or line.startswith('[SOLUTION') or line.startswith('[TWIST'):
            continue

        # Try to extract citation tag
        citation_match = re.match(r'^\[SCENE:([\d,\s]+)\]\s*(.*)', line)

        if citation_match:
            raw_citations = citation_match.group(1)
            paragraph_text = citation_match.group(2).strip()

            # Parse cited scene numbers
            cited_scenes = []
            for part in raw_citations.split(','):
                part = part.strip()
                if part.isdigit():
                    cited_scenes.append(int(part))

            # Validate: are cited scenes real?
            invalid_citations = [s for s in cited_scenes if s not in valid_scenes]
            is_valid = len(cited_scenes) > 0 and len(invalid_citations) == 0

            para_entry = {
                "text":              paragraph_text,
                "citations":         cited_scenes,
                "invalid_citations": invalid_citations,
                "valid":             is_valid
            }

            result["paragraphs"].append(para_entry)
            total_paras += 1

            if is_valid:
                cited_count += 1
            else:
                result["hallucination_flags"].append({
                    "paragraph": paragraph_text[:80] + "...",
                    "reason":    f"References non-existent scene(s): {invalid_citations}"
                })
                logger.warning(f"Invalid citation detected: {invalid_citations} — text: '{paragraph_text[:60]}...'")

        else:
            # No citation tag at all = potential hallucination
            if len(line) > 20:   # ignore short lines
                result["uncited_paragraphs"].append(line)
                result["hallucination_flags"].append({
                    "paragraph": line[:80] + "...",
                    "reason":    "No scene citation tag found"
                })
                total_paras += 1
                logger.warning(f"Uncited paragraph detected: '{line[:60]}...'")

    result["citation_coverage"] = int((cited_count / total_paras) * 100) if total_paras > 0 else 0
    logger.info(
        f"Citation parsing complete — coverage: {result['citation_coverage']}%, "
        f"flags: {len(result['hallucination_flags'])}"
    )
    return result


def strip_citation_tags(story_text: str) -> str:
    """
    Remove [SCENE:N] tags from story for clean display to users.
    Tags are only for internal analysis; users see clean text.
    """
    import re
    cleaned = re.sub(r'\[SCENE:[\d,\s]+\]\s*', '', story_text)
    logger.debug("Citation tags stripped from story for user display")
    return cleaned.strip()


# ---------------------------------------------------
# Hallucination Report
# ---------------------------------------------------
def generate_hallucination_report(parsed_story: dict) -> dict:
    """
    Generate a summary report of hallucination analysis.

    Returns:
        {
          "risk_level":        "low" | "medium" | "high",
          "citation_coverage": 95,
          "total_flags":       1,
          "summary":           "..."
        }
    """
    coverage = parsed_story.get("citation_coverage", 0)
    flags    = len(parsed_story.get("hallucination_flags", []))

    if coverage >= 90 and flags == 0:
        risk_level = "low"
    elif coverage >= 70 or flags <= 1:
        risk_level = "medium"
    else:
        risk_level = "high"

    summary = (
        f"Citation coverage: {coverage}% | "
        f"Flags: {flags} | "
        f"Risk: {risk_level.upper()}"
    )

    report = {
        "risk_level":        risk_level,
        "citation_coverage": coverage,
        "total_flags":       flags,
        "flagged_items":     parsed_story.get("hallucination_flags", []),
        "summary":           summary
    }

    logger.info(f"Hallucination report: {summary}")
    return report
