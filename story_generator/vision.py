"""
Scene Extraction Layer - Layer 1
Vision understanding with retry logic, caching, and professional logging
"""

from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from io import BytesIO
import json
import hashlib
from functools import wraps
import time
import logging
from .monitoring import token_monitor
from .logging_config import log_execution_time

# Setup logging - NO print() anywhere
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Load API Key
# ---------------------------------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.critical("GOOGLE_API_KEY not found in environment variables")
    raise ValueError("API key not found")

client = genai.Client(api_key=api_key)
logger.info("Vision layer initialized successfully")

# ---------------------------------------------------
# Retry Decorator
# ---------------------------------------------------
def retry_api_call(max_retries=2, delay=1, logger_instance=None):
    """
    Retry decorator for API calls with exponential backoff.
    logger_instance: optional external callback (for UI bridge)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait = delay * (2 ** attempt)
                        message = f"API call failed (attempt {attempt+1}/{max_retries+1}). Retrying in {wait}s... Error: {str(e)}"
                        logger.warning(message)
                        if logger_instance:
                            logger_instance("warning", message)
                        time.sleep(wait)
                    else:
                        error_msg = f"All {max_retries+1} retry attempts failed. Final error: {str(e)}"
                        logger.error(error_msg)
                        if logger_instance:
                            logger_instance("error", error_msg)
            raise last_exception
        return wrapper
    return decorator

# ---------------------------------------------------
# Image Hashing
# ---------------------------------------------------
def generate_image_hash(pil_image):
    """Generate MD5 hash fingerprint for image caching"""
    img_bytes = BytesIO()
    pil_image.save(img_bytes, format="JPEG")
    hash_value = hashlib.md5(img_bytes.getvalue()).hexdigest()
    logger.debug(f"Generated image hash: {hash_value[:8]}...")
    return hash_value

# ---------------------------------------------------
# Main Function
# ---------------------------------------------------
@log_execution_time
def extract_scene_from_images(images, logger_callback=None, use_cache=True):
    """
    Layer 1: Extract structured scene data from images.

    Args:
        images:           List of PIL.Image objects
        logger_callback:  Optional UI bridge callback function (level, message)
        use_cache:        Whether to log cache attempt (production: use Redis)

    Returns:
        List of scene dicts, or error dict on failure
    """
    logger.info(f"extract_scene_from_images called with {len(images)} image(s)")

    @retry_api_call(max_retries=2, delay=1, logger_instance=logger_callback)
    def _call_api(images):
        image_parts = []
        for i, img in enumerate(images):
            img_bytes = BytesIO()
            img.save(img_bytes, format="JPEG")
            image_parts.append(
                types.Part.from_bytes(
                    data=img_bytes.getvalue(),
                    mime_type="image/jpeg"
                )
            )
            logger.debug(f"Image {i+1} converted to bytes successfully")

        scene_prompt = """
        Analyze each image carefully.

        For EACH image, return a JSON object in this format:

        {
            "image_number": 1,
            "objects": [],
            "setting": "",
            "emotions": "",
            "characters": "",
            "time_of_day": "",
            "key_action": ""
        }

        Return a LIST of JSON objects.
        Only return valid JSON.
        """

        logger.info("Sending images to Gemini Vision API...")
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[scene_prompt, *image_parts]
        )

        usage_stats = token_monitor.add_usage(scene_prompt, response.text)
        logger.info(
            f"Vision API response received — "
            f"tokens: {usage_stats['input_tokens']}→{usage_stats['output_tokens']}, "
            f"cost: ${usage_stats['total_cost']:.5f}, "
            f"method: {usage_stats['tokenization_method']}"
        )

        if logger_callback:
            logger_callback(
                "info",
                f"Scene Analysis — Tokens: {usage_stats['input_tokens']}→{usage_stats['output_tokens']} | Cost: ${usage_stats['total_cost']:.5f} | Method: {usage_stats['tokenization_method']}"
            )

        return response.text

    # Cache logging
    image_hashes = [generate_image_hash(img) for img in images]
    logger.info(f"Image hashes generated: {[h[:8] for h in image_hashes]}")

    if use_cache:
        logger.info("Cache check: In production, Redis would be queried here with these hashes")

    try:
        response_text = _call_api(images)

        # Clean markdown formatting if present
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
            logger.debug("Stripped markdown code block from API response")

        scene_list = json.loads(cleaned_text)
        logger.info(f"Successfully parsed {len(scene_list)} scene(s) from API response")
        return scene_list

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {str(e)} | Raw response snippet: {response_text[:200]}")
        return {"error": f"JSON parsing failed: {str(e)}", "raw_output": response_text}

    except Exception as e:
        logger.error(f"extract_scene_from_images failed: {str(e)}", exc_info=True)
        return {"error": f"Scene extraction failed: {str(e)}", "raw_output": ""}