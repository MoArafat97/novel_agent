# Lazywriter

A Flask-based web application for worldbuilding and novel planning. This is the foundation for a larger AI-assisted novel-writing platform, focusing on worldbuilding elements like characters, locations, and lore.

## Features

- **Novel Management**: Create, edit, and delete novels with genre classification
- **Character Development**: Build detailed character profiles with personality, backstory, and relationships
- **Location Building**: Design locations with climate, culture, and historical information
- **Lore Creation**: Establish world rules, magic systems, mythology, and cultural elements
- **Cross-Reference Analysis**: AI-powered system to discover relationships between entities and suggest updates
- **Semantic Search**: Find related content using advanced embedding-based search
- **AI-Assisted Creation**: Use DeepSeek LLM to generate detailed entities from simple prompts
- **Dual Database System**: TinyDB for structured data, ChromaDB for semantic memory
- **Responsive Design**: Clean, modern interface using Bootstrap 5

## Project Structure

```
novel_agent/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── utils/
│   └── json_ops.py       # JSON file operations
├── templates/            # HTML templates
│   ├── layout.html       # Base template
│   ├── index.html        # Homepage
│   ├── novel_*.html      # Novel-related templates
│   ├── character_*.html  # Character-related templates
│   ├── location_*.html   # Location-related templates
│   └── lore_*.html       # Lore-related templates
├── static/
│   └── style.css         # Custom CSS styles
└── data/                 # JSON data storage
    ├── novels.json
    ├── characters.json
    ├── locations.json
    └── lore.json
```

## Installation

1. Clone or download this repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

### Getting Started

1. **Create a Novel**: Start by creating your first novel with a title, genre, and description
2. **Add Characters**: Develop your story's characters with detailed profiles
3. **Build Locations**: Create the places where your story takes place
4. **Establish Lore**: Define the rules, history, and mythology of your world

### Cross-Reference Analysis

The Cross-Reference System automatically analyzes your worldbuilding content to discover relationships and maintain consistency:

1. **Navigate** to any character, location, or lore detail page
2. **Click** the "Cross-Reference" button to start analysis
3. **Review** discovered relationships and suggested updates
4. **Apply** approved changes to keep your world consistent

The system uses AI to:
- Detect mentions of existing entities in new content
- Identify relationships between characters, locations, and lore
- Suggest specific updates to related entities
- Find potential new entities that should be added

See [Cross-Reference System Documentation](docs/cross-reference-system.md) for detailed information.

### Data Management

- Structured data stored in TinyDB (JSON files in `data/tinydb/`)
- Semantic embeddings stored in ChromaDB (DuckDB+Parquet in `data/chromadb/`)
- Each entity (novel, character, location, lore) has a unique UUID
- Data is automatically saved when you create or edit entries
- Deleting a novel will also delete all associated characters, locations, and lore

## Future Enhancements

This application is designed to be easily extensible. Planned features include:

- AI-powered content generation using CrewAI or similar frameworks
- Context-aware suggestions and auto-completion
- Chapter and scene planning tools
- Export functionality for various formats
- Collaboration features
- Advanced search and filtering
- Relationship mapping between entities

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML5, Bootstrap 5, Font Awesome icons, JavaScript
- **Storage**: TinyDB (structured data) + ChromaDB (semantic embeddings)
- **AI Integration**: OpenRouter API with DeepSeek models
- **Templating**: Jinja2
- **Styling**: Custom CSS with Bootstrap framework
- **Search**: Semantic search using embeddings and vector similarity

## Contributing

This is a foundational project designed for expansion. Key areas for contribution:

1. AI integration for content generation
2. Database migration (SQLite, PostgreSQL)
3. User authentication and multi-user support
4. Advanced worldbuilding features
5. Export and import functionality
6. Mobile app development

## License

This project is open source and available under the MIT License.
