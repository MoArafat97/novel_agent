import json
import os
from typing import List, Dict, Any

# Data directory path
DATA_DIR = 'data'

# File paths
NOVELS_FILE = os.path.join(DATA_DIR, 'novels.json')
CHARACTERS_FILE = os.path.join(DATA_DIR, 'characters.json')
LOCATIONS_FILE = os.path.join(DATA_DIR, 'locations.json')
LORE_FILE = os.path.join(DATA_DIR, 'lore.json')

def ensure_data_dir():
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_json_file(file_path: str, default: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Load data from a JSON file, return default if file doesn't exist."""
    if default is None:
        default = []
    
    ensure_data_dir()
    
    if not os.path.exists(file_path):
        # Create the file with default data
        save_json_file(file_path, default)
        return default
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupted or doesn't exist, return default
        return default

def save_json_file(file_path: str, data: List[Dict[str, Any]]):
    """Save data to a JSON file."""
    ensure_data_dir()
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Novel operations
def load_novels() -> List[Dict[str, Any]]:
    """Load all novels from the JSON file."""
    return load_json_file(NOVELS_FILE)

def save_novels(novels: List[Dict[str, Any]]):
    """Save novels to the JSON file."""
    save_json_file(NOVELS_FILE, novels)

def get_novel_by_id(novel_id: str) -> Dict[str, Any]:
    """Get a specific novel by ID."""
    novels = load_novels()
    return next((novel for novel in novels if novel['id'] == novel_id), None)

# Character operations
def load_characters() -> List[Dict[str, Any]]:
    """Load all characters from the JSON file."""
    return load_json_file(CHARACTERS_FILE)

def save_characters(characters: List[Dict[str, Any]]):
    """Save characters to the JSON file."""
    save_json_file(CHARACTERS_FILE, characters)

def get_character_by_id(character_id: str) -> Dict[str, Any]:
    """Get a specific character by ID."""
    characters = load_characters()
    return next((character for character in characters if character['id'] == character_id), None)

def get_characters_by_novel(novel_id: str) -> List[Dict[str, Any]]:
    """Get all characters for a specific novel."""
    characters = load_characters()
    return [character for character in characters if character['novel_id'] == novel_id]

# Location operations
def load_locations() -> List[Dict[str, Any]]:
    """Load all locations from the JSON file."""
    return load_json_file(LOCATIONS_FILE)

def save_locations(locations: List[Dict[str, Any]]):
    """Save locations to the JSON file."""
    save_json_file(LOCATIONS_FILE, locations)

def get_location_by_id(location_id: str) -> Dict[str, Any]:
    """Get a specific location by ID."""
    locations = load_locations()
    return next((location for location in locations if location['id'] == location_id), None)

def get_locations_by_novel(novel_id: str) -> List[Dict[str, Any]]:
    """Get all locations for a specific novel."""
    locations = load_locations()
    return [location for location in locations if location['novel_id'] == novel_id]

# Lore operations
def load_lore() -> List[Dict[str, Any]]:
    """Load all lore entries from the JSON file."""
    return load_json_file(LORE_FILE)

def save_lore(lore: List[Dict[str, Any]]):
    """Save lore entries to the JSON file."""
    save_json_file(LORE_FILE, lore)

def get_lore_by_id(lore_id: str) -> Dict[str, Any]:
    """Get a specific lore entry by ID."""
    lore_entries = load_lore()
    return next((lore for lore in lore_entries if lore['id'] == lore_id), None)

def get_lore_by_novel(novel_id: str) -> List[Dict[str, Any]]:
    """Get all lore entries for a specific novel."""
    lore_entries = load_lore()
    return [lore for lore in lore_entries if lore['novel_id'] == novel_id]

# Utility functions
def delete_novel_and_related_data(novel_id: str):
    """Delete a novel and all its related data (characters, locations, lore)."""
    # Delete novel
    novels = load_novels()
    novels = [novel for novel in novels if novel['id'] != novel_id]
    save_novels(novels)
    
    # Delete related characters
    characters = load_characters()
    characters = [character for character in characters if character['novel_id'] != novel_id]
    save_characters(characters)
    
    # Delete related locations
    locations = load_locations()
    locations = [location for location in locations if location['novel_id'] != novel_id]
    save_locations(locations)
    
    # Delete related lore
    lore_entries = load_lore()
    lore_entries = [lore for lore in lore_entries if lore['novel_id'] != novel_id]
    save_lore(lore_entries)

def get_novel_statistics(novel_id: str) -> Dict[str, int]:
    """Get statistics for a novel (count of characters, locations, lore)."""
    characters_count = len(get_characters_by_novel(novel_id))
    locations_count = len(get_locations_by_novel(novel_id))
    lore_count = len(get_lore_by_novel(novel_id))
    
    return {
        'characters': characters_count,
        'locations': locations_count,
        'lore': lore_count
    }

def initialize_data_files():
    """Initialize all data files with empty arrays if they don't exist."""
    load_novels()
    load_characters()
    load_locations()
    load_lore()

# Initialize data files when module is imported
initialize_data_files()
