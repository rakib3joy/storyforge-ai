"""
Scene Coherence & Emotional Arc Tracking - Level 2 Upgrade
Tracks emotional transitions across scenes to enforce narrative coherence modeling.

Interview talking point:
  "I implemented scene-level emotional arc tracking. The system maps emotion transitions
   (happy → scared → hopeful) and instructs the AI to reflect those transformations
   in the narrative — that's narrative coherence modeling."
"""

import logging
from .logging_config import log_execution_time

logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Emotion Taxonomy
# ---------------------------------------------------
# Grouped by valence so we can detect meaningful shifts
EMOTION_GROUPS = {
    "positive":  ["happy", "joyful", "excited", "peaceful", "hopeful", "content", "playful", "proud", "loving"],
    "negative":  ["scared", "sad", "angry", "fearful", "anxious", "desperate", "lonely", "betrayed", "horrified"],
    "tense":     ["suspenseful", "nervous", "uncertain", "tense", "worried", "confused", "surprised"],
    "neutral":   ["calm", "neutral", "observing", "curious", "focused", "determined"]
}

def classify_emotion(emotion_text: str) -> str:
    """
    Classify a raw emotion string into a valence group.

    Args:
        emotion_text: raw emotion from scene extraction (e.g. "happy and excited")

    Returns:
        valence group string: 'positive' | 'negative' | 'tense' | 'neutral'
    """
    if not emotion_text:
        return "neutral"

    emotion_lower = emotion_text.lower()
    for group, keywords in EMOTION_GROUPS.items():
        if any(keyword in emotion_lower for keyword in keywords):
            return group

    return "neutral"


# ---------------------------------------------------
# 5️⃣ Emotional Arc Builder
# ---------------------------------------------------
@log_execution_time
def build_emotional_arc(scene_data: list) -> dict:
    """
    Analyse the emotional journey across all scenes.

    For each scene, extracts the emotion, classifies it into a valence group,
    and detects meaningful emotional transitions (e.g. positive → negative).

    Args:
        scene_data: list of scene dicts from Layer 1

    Returns:
        emotional_arc dict:
        {
            "arc":         [{"scene": 1, "emotion": "happy", "valence": "positive"}, ...],
            "transitions": [{"from_scene": 1, "to_scene": 2, "shift": "positive→negative"}, ...],
            "dominant_emotion": "positive",
            "arc_summary": "Story moves from positive → negative → positive (full arc)"
        }
    """
    logger.info(f"Building emotional arc for {len(scene_data)} scene(s)")

    arc = []
    for i, scene in enumerate(scene_data):
        raw_emotion = scene.get("emotions", "neutral")
        valence     = classify_emotion(raw_emotion)
        arc.append({
            "scene":   i + 1,
            "emotion": raw_emotion,
            "valence": valence
        })
        logger.debug(f"Scene {i+1} emotion: '{raw_emotion}' → group: '{valence}'")

    # Detect transitions between consecutive scenes
    transitions = []
    for i in range(len(arc) - 1):
        current  = arc[i]
        next_s   = arc[i + 1]
        if current["valence"] != next_s["valence"]:
            shift = f"{current['valence']}→{next_s['valence']}"
            transitions.append({
                "from_scene": current["scene"],
                "to_scene":   next_s["scene"],
                "shift":      shift,
                "from_emotion": current["emotion"],
                "to_emotion":   next_s["emotion"]
            })
            logger.info(f"Emotional transition detected: Scene {current['scene']} → {next_s['scene']} ({shift})")

    # Dominant emotion = most frequent valence
    valence_counts = {}
    for entry in arc:
        v = entry["valence"]
        valence_counts[v] = valence_counts.get(v, 0) + 1
    dominant_emotion = max(valence_counts, key=valence_counts.get) if valence_counts else "neutral"

    # Build a human-readable arc summary
    valence_sequence = [entry["valence"] for entry in arc]
    unique_sequence  = []
    for v in valence_sequence:
        if not unique_sequence or unique_sequence[-1] != v:
            unique_sequence.append(v)
    arc_summary = " → ".join(unique_sequence)

    result = {
        "arc":              arc,
        "transitions":      transitions,
        "dominant_emotion": dominant_emotion,
        "arc_summary":      arc_summary,
        "has_transformation": len(transitions) > 0
    }

    logger.info(
        f"Emotional arc complete — dominant: '{dominant_emotion}', "
        f"transitions: {len(transitions)}, arc: '{arc_summary}'"
    )
    return result


def build_arc_instructions(emotional_arc: dict) -> str:
    """
    Convert emotional arc analysis into concrete prompt instructions.
    These are injected into the story generation prompt so the AI
    reflects the emotional journey from the images.

    Args:
        emotional_arc: output from build_emotional_arc()

    Returns:
        instruction string to be added to the generation prompt
    """
    if not emotional_arc or not emotional_arc.get("arc"):
        return ""

    arc_lines = []
    for entry in emotional_arc["arc"]:
        arc_lines.append(
            f"  Scene {entry['scene']}: emotion='{entry['emotion']}' (tone group: {entry['valence']})"
        )

    transition_lines = []
    for t in emotional_arc.get("transitions", []):
        transition_lines.append(
            f"  Scene {t['from_scene']} → Scene {t['to_scene']}: "
            f"'{t['from_emotion']}' transforms into '{t['to_emotion']}' ({t['shift']})"
        )

    instructions = f"""
    EMOTIONAL ARC REQUIREMENTS (5️⃣ Narrative Coherence Modeling):
    The images show the following emotional journey:
    {chr(10).join(arc_lines)}

    Overall arc: {emotional_arc['arc_summary']}
    """

    if transition_lines:
        instructions += f"""
    Emotional transformations that MUST be reflected in the story:
    {chr(10).join(transition_lines)}

    RULE: The story must capture these emotional shifts. A character or scene
    that starts {'joyful' if 'positive' in emotional_arc['arc_summary'] else 'tense'}
    and becomes {'fearful' if 'negative' in emotional_arc['arc_summary'] else 'resolved'}
    must show WHY and HOW that transformation happened.
    """
    else:
        instructions += f"""
    The story maintains a consistent '{emotional_arc['dominant_emotion']}' tone throughout.
    """

    logger.debug("Emotional arc instructions built successfully")
    return instructions


# ---------------------------------------------------
# Coherence Scoring
# ---------------------------------------------------
def score_story_coherence(story_text: str, emotional_arc: dict) -> dict:
    """
    Score how well the generated story reflects the emotional arc.
    Simple keyword-based scoring — good for demo, honest in interview.

    Returns:
        {
          "score": 0-100,
          "matched_emotions": [...],
          "missing_emotions": [...],
          "feedback": "..."
        }
    """
    if not story_text or not emotional_arc.get("arc"):
        return {"score": 0, "matched_emotions": [], "missing_emotions": [], "feedback": "No data"}

    story_lower     = story_text.lower()
    matched         = []
    missing         = []

    for entry in emotional_arc["arc"]:
        emotion_words = entry["emotion"].lower().split()
        # Check if any word from the scene emotion appears in the story
        if any(word in story_lower for word in emotion_words if len(word) > 3):
            matched.append(entry["emotion"])
        else:
            missing.append(entry["emotion"])

    total     = len(emotional_arc["arc"])
    score     = int((len(matched) / total) * 100) if total > 0 else 0
    feedback  = (
        f"Story reflects {len(matched)}/{total} scene emotions. "
        f"Arc coverage: {score}%. "
        + ("All emotional transitions captured." if score == 100
           else f"Missing emotional coverage: {missing}")
    )

    logger.info(f"Coherence score: {score}/100 — matched: {matched}, missing: {missing}")
    return {
        "score":            score,
        "matched_emotions": matched,
        "missing_emotions": missing,
        "feedback":         feedback
    }
