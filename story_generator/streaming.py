"""
True Token-Level Streaming - Level 3 Upgrade (9️⃣)
Uses Gemini's real streaming API instead of UI-level paragraph simulation.

Interview talking point:
  "There are two types of streaming. UI-level streaming splits the completed
   response into chunks and displays them with delays — it's fake streaming.
   Token-level streaming uses the API's stream=True feature, where tokens
   arrive progressively from the model as they are generated. My system
   supports both, and I can explain the tradeoff: token-level streaming has
   lower perceived latency but requires a streaming-capable client."
"""

import logging
from dotenv import load_dotenv
import os
from google import genai
from .monitoring import token_monitor
from .logging_config import log_execution_time

logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found")

client = genai.Client(api_key=api_key)

# ---------------------------------------------------
# Streaming Mode Enum
# ---------------------------------------------------
class StreamingMode:
    TOKEN_LEVEL = "token_level"   # Real API streaming — tokens arrive live
    UI_LEVEL    = "ui_level"      # Simulated — full response split by paragraph
    DISABLED    = "disabled"      # No streaming — return full text at once


# ---------------------------------------------------
# 9️⃣ True Token-Level Streaming Generator
# ---------------------------------------------------
def stream_story_token_level(prompt: str):
    """
    True token-level streaming using Gemini's streaming API.
    Returns a generator that yields text chunks as they arrive.

    Usage in Streamlit:
        placeholder = st.empty()
        full_text = ""
        for chunk in stream_story_token_level(prompt):
            full_text += chunk
            placeholder.markdown(full_text)

    Args:
        prompt: The full story generation prompt

    Yields:
        str: text chunk from the model (may be a word, token, or phrase)
    """
    logger.info("Starting token-level streaming generation")
    total_chunks = 0
    full_response = ""

    try:
        # Use generate_content_stream for real streaming
        response_stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt
        )

        for chunk in response_stream:
            if chunk.text:
                full_response += chunk.text
                total_chunks  += 1
                yield chunk.text

        # Track tokens after stream completes
        usage_stats = token_monitor.add_usage(prompt, full_response)
        logger.info(
            f"Token-level streaming complete — "
            f"chunks: {total_chunks}, "
            f"tokens: {usage_stats['input_tokens']}→{usage_stats['output_tokens']}, "
            f"cost: ${usage_stats['total_cost']:.5f}"
        )

    except AttributeError:
        # Fallback: generate_content_stream not available in this SDK version
        logger.warning("generate_content_stream not available — falling back to UI-level streaming")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        token_monitor.add_usage(prompt, response.text)
        # Simulate token-level by yielding word by word
        for word in response.text.split(" "):
            yield word + " "

    except Exception as e:
        logger.error(f"Token-level streaming failed: {str(e)}", exc_info=True)
        yield f"\n\n[Streaming error: {str(e)}]"


def stream_story_ui_level(story_text: str):
    """
    UI-level streaming: yields story paragraph by paragraph.
    This is a simulation — the full text is already generated.
    Used when token-level streaming is unavailable or disabled.

    Args:
        story_text: completed story string

    Yields:
        str: one paragraph at a time
    """
    logger.info("Starting UI-level streaming (paragraph simulation)")
    paragraphs = [p.strip() for p in story_text.strip().split('\n\n') if p.strip()]

    for i, paragraph in enumerate(paragraphs):
        logger.debug(f"Yielding paragraph {i+1}/{len(paragraphs)}")
        yield paragraph + "\n\n"

    logger.info(f"UI-level streaming complete — {len(paragraphs)} paragraphs delivered")


def get_streaming_explanation() -> dict:
    """
    Returns a structured explanation of streaming modes.
    Used in the UI 'How It Works' section for interview demo.
    """
    return {
        "token_level": {
            "name":        "Token-Level Streaming (Real)",
            "description": "Tokens arrive from the model as they are generated. "
                           "Uses Gemini's generate_content_stream() API. "
                           "Lowest perceived latency — user sees text appear word by word.",
            "latency":     "~50-200ms first token",
            "requires":    "Streaming-capable client + server-sent events",
            "use_case":    "Production apps, chat interfaces"
        },
        "ui_level": {
            "name":        "UI-Level Streaming (Simulated)",
            "description": "Full response is generated first, then displayed "
                           "paragraph by paragraph with a time.sleep() delay. "
                           "Gives the appearance of streaming but is not real.",
            "latency":     "Full generation time + display delay",
            "requires":    "Nothing special — works everywhere",
            "use_case":    "Demos, prototypes, environments without SSE support"
        }
    }
