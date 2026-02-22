"""
Text-to-Speech Layer - Layer 4
Audio synthesis with professional logging
"""

from gtts import gTTS
from io import BytesIO
import logging
from .logging_config import log_execution_time

# Setup logging - NO print() anywhere
logger = logging.getLogger(__name__)

@log_execution_time
def narrate_story(story_text):
    """
    Converts story text to audio using Google TTS.

    Args:
        story_text: The generated story string

    Returns:
        BytesIO audio object, or None if TTS fails
    """
    logger.info(f"narrate_story called — story length: {len(story_text)} characters")

    try:
        # Clean special tags before narration (they are for readers, not listeners)
        cleaned_story = story_text \
            .replace("[MORAL]:", "") \
            .replace("[SOLUTION]:", "") \
            .replace("[TWIST]:", "")

        logger.debug("Special tags cleaned from story text")

        tts = gTTS(text=cleaned_story, lang="en", slow=False)

        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)

        logger.info("Audio narration generated successfully")
        return audio_fp

    except Exception as e:
        logger.error(f"narrate_story failed: {str(e)}", exc_info=True)
        return None