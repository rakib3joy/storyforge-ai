"""
Multimodal Memory Module - Level 2 Upgrade (7️⃣)
Remembers previous story context so new images build on existing narrative.

Interview talking point:
  "I built a multimodal memory system. When a user uploads new images,
   the system retrieves the previous story context — characters, setting,
   emotional arc — and passes it as memory to the new generation.
   That's stateful multimodal storytelling."
"""

import json
import logging
from datetime import datetime
from .logging_config import log_execution_time

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Memory Store (in-memory; production → Redis / DB)
# ---------------------------------------------------
class StoryMemory:
    """
    Session-scoped memory that stores previous story context.
    In production this would be persisted to Redis or a database.
    For demo/interview: stored in Python dict (resets on restart).
    """

    def __init__(self):
        self._store: dict = {}       # session_id → memory_entry
        logger.info("StoryMemory initialized (in-memory store)")

    # ------------------------------------------------
    def save(self, session_id: str, scene_data: list,
             character_map: dict, story_text: str,
             emotional_arc: dict) -> dict:
        """
        Persist a completed story generation into memory.

        Args:
            session_id:    Unique key for this user session
            scene_data:    Layer 1 output (list of scene dicts)
            character_map: Layer 2 output (entity → BD name)
            story_text:    Final generated story string
            emotional_arc: coherence.py arc output

        Returns:
            memory_entry dict
        """
        # Extract a compact story summary (first 300 chars of story)
        story_summary = story_text[:300] + "..." if len(story_text) > 300 else story_text

        # Freeze character map so names stay consistent in future sessions
        frozen_characters = dict(character_map)

        # Capture the final scene setting for continuity
        last_scene    = scene_data[-1] if scene_data else {}
        last_setting  = last_scene.get("setting", "unknown")
        last_emotion  = last_scene.get("emotions", "neutral")

        memory_entry = {
            "session_id":        session_id,
            "timestamp":         datetime.now().isoformat(),
            "story_summary":     story_summary,
            "character_map":     frozen_characters,    # frozen names
            "scene_count":       len(scene_data),
            "last_setting":      last_setting,
            "last_emotion":      last_emotion,
            "arc_summary":       emotional_arc.get("arc_summary", ""),
            "dominant_emotion":  emotional_arc.get("dominant_emotion", "neutral"),
            "full_story":        story_text            # full text for context window
        }

        self._store[session_id] = memory_entry
        logger.info(
            f"Memory saved — session: '{session_id}', "
            f"chars: {list(frozen_characters.values())}, "
            f"arc: '{memory_entry['arc_summary']}'"
        )
        return memory_entry

    # ------------------------------------------------
    def load(self, session_id: str) -> dict | None:
        """
        Load previous memory for a session.

        Returns:
            memory_entry dict, or None if no history exists
        """
        entry = self._store.get(session_id)
        if entry:
            logger.info(f"Memory loaded — session: '{session_id}', timestamp: {entry['timestamp']}")
        else:
            logger.info(f"No memory found for session: '{session_id}'")
        return entry

    # ------------------------------------------------
    def clear(self, session_id: str) -> None:
        """Clear memory for a session (user presses 'Start New Story')"""
        if session_id in self._store:
            del self._store[session_id]
            logger.info(f"Memory cleared for session: '{session_id}'")

    # ------------------------------------------------
    def has_memory(self, session_id: str) -> bool:
        return session_id in self._store

    # ------------------------------------------------
    def get_all_sessions(self) -> list:
        """Return list of all active session IDs (for debug dashboard)"""
        return list(self._store.keys())


# ---------------------------------------------------
# Memory-Aware Prompt Builder
# ---------------------------------------------------
@log_execution_time
def build_memory_context(memory_entry: dict) -> str:
    """
    Convert a stored memory entry into a prompt injection block.
    This is appended to the new story generation prompt so the AI
    knows what happened before.

    Args:
        memory_entry: output of StoryMemory.load()

    Returns:
        context string ready for prompt injection
    """
    if not memory_entry:
        logger.debug("No memory context to inject")
        return ""

    # Characters that MUST keep their names in continuation
    char_list = ", ".join(
        [f"'{entity}' is called '{name}'"
         for entity, name in memory_entry.get("character_map", {}).items()]
    )

    context = f"""
    STORY MEMORY — PREVIOUS SESSION CONTEXT (7️⃣ Multimodal Memory):
    This is a CONTINUATION of a previous story. Use the context below
    to maintain consistency with what came before.

    Previous Story Summary:
    "{memory_entry['story_summary']}"

    Established Characters (MUST keep exact same names):
    {char_list if char_list else 'No established characters yet.'}

    Last Known Setting: {memory_entry['last_setting']}
    Last Emotional State: {memory_entry['last_emotion']}
    Previous Story Arc: {memory_entry['arc_summary']}

    CONTINUATION RULES:
    ✓ Use the SAME character names as the previous story.
    ✓ The new story should feel like a natural sequel or next chapter.
    ✓ Reference the previous emotional state when beginning the new narrative.
    ✓ The new setting may differ, but character personalities must stay consistent.
    """

    logger.info(
        f"Memory context injected — previous arc: '{memory_entry['arc_summary']}', "
        f"characters: {list(memory_entry.get('character_map', {}).values())}"
    )
    return context


# ---------------------------------------------------
# Global memory instance (app-level singleton)
# ---------------------------------------------------
story_memory = StoryMemory()
