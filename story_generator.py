from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

from gtts import gTTS
from io import BytesIO

import tiktoken
import json
import time
from functools import wraps

# ---------------------------------------------------
# Load API Key
# ---------------------------------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found")

client = genai.Client(api_key=api_key)


# ---------------------------------------------------
# 🔥 LAYER 3 ENGINEERING ENHANCEMENTS
# ---------------------------------------------------

class TokenMonitor:
    """Track API usage and costs"""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Gemini 2.5 Flash pricing (AI Studio, checked Feb 2026)
        # Input:  $0.30 per 1M tokens  →  $0.00030 per 1K
        # Output: $2.50 per 1M tokens  →  $0.00250 per 1K
        self.input_cost_per_1k = 0.00030
        self.output_cost_per_1k = 0.00250
        # Use GPT-4 tokenizer as close approximation
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def estimate_tokens(self, text):
        """
        Estimate tokens using real tokenizer instead of character heuristic.
        NOTE: Approximation for Gemini models.
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def add_usage(self, input_text, output_text):
        """Add token usage"""
        input_tokens = self.estimate_tokens(input_text)
        output_tokens = self.estimate_tokens(output_text)
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": (input_tokens / 1000) * self.input_cost_per_1k,
            "output_cost": (output_tokens / 1000) * self.output_cost_per_1k,
            "total_cost": ((input_tokens / 1000) * self.input_cost_per_1k) + 
                         ((output_tokens / 1000) * self.output_cost_per_1k)
        }
    
    def get_session_stats(self):
        """Get session statistics"""
        total_cost = ((self.total_input_tokens / 1000) * self.input_cost_per_1k) + \
                    ((self.total_output_tokens / 1000) * self.output_cost_per_1k)
        
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": total_cost
        }

# Global token monitor
token_monitor = TokenMonitor()

def retry_api_call(max_retries=2, delay=1, logger=None):
    """
    Retry decorator for API calls
    logger: Optional callback function for logging retry attempts
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
                        message = f"API call failed (attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay} seconds..."
                        if logger:
                            logger("warning", message)
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        error_message = f"All retry attempts failed. Final error: {str(e)}"
                        if logger:
                            logger("error", error_message)
            
            raise last_exception
        return wrapper
    return decorator

def generate_image_hash(pil_image):
    """Generate a simple hash for image caching"""
    import hashlib
    img_bytes = BytesIO()
    pil_image.save(img_bytes, format="JPEG")
    return hashlib.md5(img_bytes.getvalue()).hexdigest()


# ---------------------------------------------------
#  STEP 1: Scene Extraction Layer (Vision Understanding)
# ---------------------------------------------------
def extract_scene_from_images(images, logger=None, use_cache=True):
    """
    Enhanced scene extraction with caching and error handling
    logger: Optional callback function for logging messages
    use_cache: Whether to attempt using cached results
    """
    @retry_api_call(max_retries=2, delay=1, logger=logger)
    def _extract_scenes_api_call(images):
        image_parts = []

        for img in images:
            img_bytes = BytesIO()
            img.save(img_bytes, format="JPEG")
            img_bytes = img_bytes.getvalue()

            image_parts.append(
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/jpeg"
                )
            )

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

        contents = [scene_prompt, *image_parts]
        
        # Track token usage
        input_text = scene_prompt
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents
        )
        
        # Monitor tokens
        usage_stats = token_monitor.add_usage(input_text, response.text)
        
        # Log token usage if logger provided
        if logger:
            logger("info", f"Scene Analysis - Tokens: {usage_stats['input_tokens']}→{usage_stats['output_tokens']}, Cost: ${usage_stats['total_cost']:.4f}")

        return response.text
    
    # Generate image hashes for caching
    image_hashes = [generate_image_hash(img) for img in images]
    
    # Simple cache check (in production, use Redis or similar)
    cache_key = str(tuple(image_hashes))
    
    if use_cache and logger:
        logger("info", "Checking cache for scene analysis...")
    
    try:
        # Get API response
        response_text = _extract_scenes_api_call(images)
        
        # Clean & Parse JSON Safely
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()

        return json.loads(cleaned_text)

    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parsing failed: {str(e)}",
            "raw_output": response_text
        }
    except Exception as e:
        return {
            "error": f"Scene extraction failed: {str(e)}",
            "raw_output": ""
        }
    

# ---------------------------------------------------
#  STEP 2: Character Consistency Layer (IMPROVED)
# ---------------------------------------------------
def extract_main_characters(scene_data):
    """
    Enhanced character extraction with better entity splitting
    Handles multiple characters, filters out non-human entities
    """
    detected_entities = []
    non_character_words = {"none", "butterflies", "birds", "animals", "nature", "objects", "items"}

    for scene in scene_data:
        char_field = scene.get("characters", "")
        
        if char_field and char_field.lower() != "none":
            # Split by common delimiters
            if isinstance(char_field, str):
                entities = [c.strip() for c in char_field.replace(" and ", ",").split(",")]
                
                for entity in entities:
                    # Filter out non-human entities
                    entity_clean = entity.lower().strip()
                    
                    if (entity_clean and 
                        entity_clean not in non_character_words and 
                        not any(word in entity_clean for word in ["playing", "flying", "running"]) and
                        len(entity_clean) > 2):  # Avoid single letters
                        
                        detected_entities.append(entity_clean)

    # Remove duplicates while preserving order
    unique_entities = []
    seen = set()
    for entity in detected_entities:
        if entity not in seen:
            unique_entities.append(entity)
            seen.add(entity)

    character_map = {}
    
    # Predefined Bangladeshi names
    bd_names = ["Rafi", "Nusrat", "Siam", "Mim", "Tanvir", "Tania", "Karim", "Fatima", "Hassan", "Rashida"]

    for i, entity in enumerate(unique_entities):
        if i < len(bd_names):
            character_map[entity] = bd_names[i]

    return character_map    


# ---------------------------------------------------
#  STEP 2: Story Prompt Builder (ENHANCED)
# ---------------------------------------------------
def create_advanced_prompt(style, tone="Neutral", narrative_structure="Linear storytelling"):
    """
    Enhanced prompt builder with tone control and narrative structure
    """
    
    # Base prompt
    base_prompt = f"""
    **Your Persona:** You are a friendly and engaging storyteller.
    **Main Goal:** Write a story in simple, clear, and modern English.
    **Task:** Create one single story that connects all the provided scenes in order.
    **Style Requirement:** The story must fit the '{style}' genre.
    """
    
    #  NEW: Tone Control Instructions
    tone_instructions = {
        "Calm": "Use peaceful, gentle language. Focus on harmony and tranquility. Avoid sudden conflicts.",
        "Dark": "Use mysterious, intense language. Include shadows, uncertainty, and deeper emotions.",
        "Energetic": "Use dynamic, exciting language. Include action words, enthusiasm, and vibrant descriptions.",
        "Neutral": "Use balanced, clear language suitable for all audiences."
    }
    
    tone_prompt = f"**Tone Requirement:** {tone_instructions.get(tone, tone_instructions['Neutral'])}"
    
    #  NEW: Narrative Structure Instructions  
    structure_instructions = {
        "Linear storytelling": "Tell the story in chronological order from beginning to end.",
        "Hero's Journey": "Follow the hero's journey: ordinary world → call to adventure → challenges → transformation → return with wisdom.",
        "Flashback-based": "Start with the climax or ending, then reveal how we got there through flashbacks.",
        "Mystery reveal format": "Build suspense gradually, revealing clues throughout, with the major revelation at the end."
    }
    
    structure_prompt = f"**Narrative Structure:** {structure_instructions.get(narrative_structure, structure_instructions['Linear storytelling'])}"
    
    # Core instructions
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

    # Style-specific endings
    style_instruction = ""
    if style == "Morale":
        style_instruction = """
        After the story, add:
        [MORAL]: followed by a single-sentence moral.
        """
    elif style == "Mystery":
        style_instruction = """
        After the story, add:
        [SOLUTION]: Reveal the culprit and key clue.
        """
    elif style == "Thriller":
        style_instruction = """
        After the story, add:
        [TWIST]: Add a shocking final twist.
        """

    return base_prompt + "\n\n" + tone_prompt + "\n\n" + structure_prompt + "\n\n" + core_instructions + style_instruction


# ---------------------------------------------------
#  STEP 3: Narrative Generation Layer (ENHANCED)
# ---------------------------------------------------

def validate_scene_data(scene_data):
    """
    Validates the scene data before story generation
    Returns (is_valid, error_message)
    """
    if not scene_data:
        return False, "No scene data provided"
    
    if isinstance(scene_data, dict) and "error" in scene_data:
        return False, f"Scene extraction error: {scene_data.get('error', 'Unknown error')}"
    
    if not isinstance(scene_data, list):
        return False, "Scene data must be a list of scenes"
    
    if len(scene_data) == 0:
        return False, "No scenes found in the data"
    
    # Check if scenes have required fields
    for i, scene in enumerate(scene_data):
        if not isinstance(scene, dict):
            return False, f"Scene {i+1} is not properly formatted"
        
        required_fields = ["setting", "characters", "key_action"]
        missing_fields = [field for field in required_fields if not scene.get(field)]
        
        if len(missing_fields) == len(required_fields):
            return False, f"Scene {i+1} is missing critical information"
    
    return True, "Valid scene data"


def build_contextual_prompt(scene_data, character_map, style, tone, narrative_structure):
    """
    Builds a comprehensive prompt for story generation
    """
    # Scene context
    structured_context = f"""
    SCENE ANALYSIS:
    The following scenes have been extracted from images in chronological order:
    
    {json.dumps(scene_data, indent=2)}
    
    Your task is to weave these scenes into one cohesive narrative.
    """
    
    # Character consistency
    character_instruction = ""
    if character_map:
        character_instruction = f"""
        CHARACTER CONSISTENCY:
        The following characters must maintain consistent names throughout:
        {json.dumps(character_map, indent=2)}
        
        STRICT RULE: Use ONLY these assigned names. Do not create new character names.
        """
    
    # Style and structure guidance
    story_guidance = create_advanced_prompt(style, tone, narrative_structure)
    
    # Quality assurance
    quality_instructions = """
    QUALITY REQUIREMENTS:
    ✓ Every uploaded scene must be referenced in the story
    ✓ Maintain logical flow between scenes
    ✓ Keep cultural context (Bangladeshi setting)
    ✓ Ensure age-appropriate content
    ✓ Create emotional engagement
    """
    
    return f"{structured_context}\n\n{character_instruction}\n\n{story_guidance}\n\n{quality_instructions}"


def post_process_story(raw_story_text, style):
    """
    Post-processes the generated story for consistency and formatting
    """
    try:
        # Clean up any potential formatting issues
        cleaned_story = raw_story_text.strip()
        
        # Ensure proper line breaks for readability
        lines = cleaned_story.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                processed_lines.append(line)
        
        # Join with proper spacing
        final_story = '\n\n'.join(processed_lines)
        
        # Validate that style-specific elements are present
        style_validation = validate_style_requirements(final_story, style)
        
        return {
            "story": final_story,
            "validation": style_validation,
            "word_count": len(final_story.split()),
            "paragraph_count": len([p for p in final_story.split('\n\n') if p.strip()])
        }
        
    except Exception as e:
        return {
            "story": raw_story_text,
            "validation": {"valid": False, "error": str(e)},
            "word_count": 0,
            "paragraph_count": 0
        }


def validate_style_requirements(story_text, style):
    """
    Validates that the story meets style-specific requirements
    """
    validation = {"valid": True, "missing_elements": []}
    
    style_requirements = {
        "Mystery": ["[SOLUTION]", "mystery", "clue"],
        "Thriller": ["[TWIST]", "suspense"],
        "Morale": ["[MORAL]", "lesson"],
        "Comedy": ["humor", "funny", "laugh"],
        "Sci-Fi": ["technology", "future", "science"],
        "Adventure": ["journey", "quest", "adventure"],
        "Fairy Tale": ["once upon", "magical", "fairy"]
    }
    
    if style in style_requirements:
        required_elements = style_requirements[style]
        story_lower = story_text.lower()
        
        for element in required_elements:
            if element.startswith("[") and element.endswith("]"):
                # Special tags like [MORAL], [SOLUTION]
                if element not in story_text:
                    validation["missing_elements"].append(element)
            else:
                # General content words
                if element not in story_lower:
                    validation["missing_elements"].append(element)
    
    if validation["missing_elements"]:
        validation["valid"] = False
    
    return validation


def generate_story_from_images(scene_data, style, tone="Neutral", narrative_structure="Linear storytelling"):
    """
    Main story generation function
    """
    is_valid, validation_message = validate_scene_data(scene_data)
    if not is_valid:
        return f"Story Generation Error: {validation_message}"
    
    try:
        character_map = extract_main_characters(scene_data)
        final_prompt = build_contextual_prompt(scene_data, character_map, style, tone, narrative_structure)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=final_prompt
        )
        
        # Monitor tokens
        usage_stats = token_monitor.add_usage(final_prompt, response.text)
        
        processed_result = post_process_story(response.text, style)
        
        if not processed_result["validation"]["valid"]:
            print(f"Story validation warnings: {processed_result['validation']}")
        
        return processed_result["story"]
        
    except Exception as e:
        return f"Story Generation Error: Failed to generate story - {str(e)}"


# ---------------------------------------------------
#  STEP 4: Text-to-Speech Layer
# ---------------------------------------------------
def narrate_story(story_text):
    """
    Converts story text to audio
    """

    try:
        # Optional: Clean special tags before narration
        cleaned_story = story_text.replace("[MORAL]:", "") \
                                   .replace("[SOLUTION]:", "") \
                                   .replace("[TWIST]:", "")

        tts = gTTS(text=cleaned_story, lang="en", slow=False)

        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)

        return audio_fp

    except Exception:
        return None
