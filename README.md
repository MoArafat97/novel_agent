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

## CI/CD Pipeline

This project includes a comprehensive GitHub Actions CI/CD pipeline that automatically tests the application on every push and pull request.

### Pipeline Features

- **Code Quality**: Linting with Ruff, formatting checks with Black, import sorting with isort
- **Testing**: Unit tests, integration tests, Flask application tests, and performance tests
- **Security**: Vulnerability scanning with Safety and Bandit
- **Coverage**: Code coverage reporting with Codecov integration
- **Build Testing**: Ensures the application builds and starts correctly

### Setting Up CI/CD

1. **Fork or clone this repository**

2. **Set up GitHub Secrets** (for full functionality):
   ```
   OPENROUTER_API_KEY: Your OpenRouter API key for AI features
   ```

3. **Enable GitHub Actions** in your repository settings

4. **Push to main or create a pull request** to trigger the pipeline

### Running Tests Locally

Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

Run the full test suite:
```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest -m unit                    # Unit tests only
python -m pytest -m integration             # Integration tests only
python -m pytest tests/test_flask_app.py    # Flask app tests only

# Run with coverage
python -m pytest --cov=agents --cov=database --cov=utils --cov=app
```

Run linting and formatting:
```bash
# Check code quality
ruff check .

# Format code
black .

# Sort imports
isort .
```

Run security scans:
```bash
# Check for vulnerabilities
safety check

# Run security linter
bandit -r .
```

### Environment Variables for Testing

The CI/CD pipeline uses these environment variables for testing:

- `TINYDB_PATH`: Path to TinyDB storage (set to test directory)
- `CHROMADB_PATH`: Path to ChromaDB storage (set to test directory)
- `USE_OPENROUTER_EMBEDDINGS`: Set to 'false' for testing without API
- `OPENROUTER_API_KEY`: Your API key (use GitHub Secrets)
- `FLASK_ENV`: Set to 'testing'
- `FLASK_DEBUG`: Set to 'false' for testing

### Pipeline Status

The pipeline runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

Pipeline jobs:
1. **Lint**: Code quality and formatting checks
2. **Test**: Comprehensive test suite (unit, integration, Flask app)
3. **Build Test**: Application startup and basic functionality
4. **Performance Test**: Performance benchmarks (main branch only)
5. **Security Scan**: Vulnerability and security checks

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

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests locally: `python -m pytest`
5. Run linting: `ruff check . && black . && isort .`
6. Commit your changes
7. Push to your fork and create a pull request

The CI/CD pipeline will automatically test your changes and provide feedback.

## License

This project is open source and available under the MIT License.
