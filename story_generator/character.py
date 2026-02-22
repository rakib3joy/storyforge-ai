"""
Character Consistency Layer - Layer 2
Enhanced entity recognition with cultural mapping and gender-aware assignment
"""

from dotenv import load_dotenv
import os
from google import genai
import json
import logging

# Load API for gender detection
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    client = genai.Client(api_key=api_key)

# Setup logging
logger = logging.getLogger(__name__)

def detect_gender_and_context(scene_data):
    """
    Use LLM to detect gender and cultural context from scenes
    Returns gender mapping for detected characters
    """
    if not api_key:
        logger.warning("No API key available for gender detection, using fallback")
        return {}
    
    try:
        # Extract all character mentions for analysis
        character_mentions = []
        for scene in scene_data:
            char_field = scene.get("characters", "")
            if char_field and char_field.lower() != "none":
                character_mentions.append({
                    "characters": char_field,
                    "setting": scene.get("setting", ""),
                    "key_action": scene.get("key_action", ""),
                    "emotions": scene.get("emotions", "")
                })
        
        if not character_mentions:
            return {}
            
        gender_detection_prompt = f"""
        Analyze the following character mentions and determine gender and cultural context:
        
        {json.dumps(character_mentions, indent=2)}
        
        For each unique character, determine:
        1. Gender (male/female/neutral/unknown)
        2. Age category (child/adult/elderly)
        3. Cultural context hints
        
        Return JSON format:
        {{
            "character_analysis": [
                {{
                    "character": "character_description",
                    "gender": "male/female/neutral/unknown", 
                    "age_category": "child/adult/elderly",
                    "cultural_hints": "any cultural context"
                }}
            ]
        }}
        
        Only return valid JSON.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=gender_detection_prompt
        )
        
        # Parse response
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.replace("```json", "").replace("```", "").strip()
            
        gender_data = json.loads(cleaned_text)
        logger.info(f"Gender detection successful: {len(gender_data.get('character_analysis', []))} characters analyzed")
        
        return gender_data.get('character_analysis', [])
        
    except Exception as e:
        logger.error(f"Gender detection failed: {str(e)}")
        return {}

def generate_culturally_appropriate_names(gender_analysis, cultural_context="Bangladeshi"):
    """
    Generate culturally appropriate names based on gender and age
    """
    
    # Enhanced Bangladeshi name database
    name_database = {
        "male": {
            "child": ["Rafi", "Siam", "Tanvir", "Arif", "Miraz", "Shohan"],
            "adult": ["Karim", "Hassan", "Rahman", "Mahmud", "Farhan", "Nasir"],
            "elderly": ["Abdul", "Mohammad", "Ibrahim", "Rashid", "Aminul", "Fazlul"]
        },
        "female": {
            "child": ["Nusrat", "Mim", "Tania", "Riya", "Sadia", "Priya"],
            "adult": ["Fatima", "Rashida", "Nasreen", "Salma", "Rehana", "Shahnaz"],
            "elderly": ["Begum", "Khatun", "Roushan", "Monowara", "Rahima", "Kamrun"]
        },
        "neutral": ["Rafi", "Siam", "Mim", "Tania", "Arif", "Nusrat"]
    }
    
    assigned_names = {}
    used_names = set()
    
    for analysis in gender_analysis:
        character = analysis.get('character', '').lower()
        gender = analysis.get('gender', 'neutral')
        age_category = analysis.get('age_category', 'adult')
        
        # Get appropriate name pool
        if gender in name_database and age_category in name_database[gender]:
            name_pool = name_database[gender][age_category]
        else:
            name_pool = name_database["neutral"]
        
        # Assign unused name
        for name in name_pool:
            if name not in used_names:
                assigned_names[character] = name
                used_names.add(name)
                logger.info(f"Assigned {gender} {age_category} name '{name}' to character '{character}'")
                break
    
    return assigned_names

def extract_main_characters(scene_data):
    """
    Enhanced character extraction with gender-aware cultural mapping
    """
    detected_entities = []
    non_character_words = {"none", "butterflies", "birds", "animals", "nature", "objects", "items"}

    # Step 1: Extract basic entities (existing logic)
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

    if not unique_entities:
        logger.info("No characters detected for mapping")
        return {}

    # Step 2: Detect gender and context using AI
    logger.info(f"Analyzing {len(unique_entities)} characters for gender and context")
    gender_analysis = detect_gender_and_context(scene_data)
    
    # Step 3: Generate appropriate names
    if gender_analysis:
        character_map = generate_culturally_appropriate_names(gender_analysis)
        logger.info(f"Generated gender-aware character mapping: {character_map}")
    else:
        # Fallback to basic assignment if gender detection fails
        logger.warning("Using fallback character mapping")
        bd_names = ["Rafi", "Nusrat", "Siam", "Mim", "Tanvir", "Tania", "Karim", "Fatima", "Hassan", "Rashida"]
        character_map = {}
        for i, entity in enumerate(unique_entities):
            if i < len(bd_names):
                character_map[entity] = bd_names[i]

    return character_map