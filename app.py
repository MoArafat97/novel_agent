from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, stream_template
import uuid
import json
import logging
from datetime import datetime
from database import WorldState
from database.semantic_search import SemanticSearchEngine
from agents import CharacterCreatorAgent, CharacterEditorAgent, LoreCreatorAgent, LoreEditorAgent, LocationCreatorAgent, LocationEditorAgent
from routes.simple_cross_reference import simple_cross_reference_bp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'worldbuilder-secret-key-change-in-production'

# Initialize WorldState, Semantic Search, and AI Agents
world_state = WorldState()
semantic_search = SemanticSearchEngine(world_state)
character_creator = CharacterCreatorAgent()
character_editor = CharacterEditorAgent()
lore_creator = LoreCreatorAgent()
lore_editor = LoreEditorAgent()
location_creator = LocationCreatorAgent()
location_editor = LocationEditorAgent()
# Note: Cross-reference functionality now uses SimpleCrossReferenceService via blueprint

# Register blueprints
app.register_blueprint(simple_cross_reference_bp)

# Test route for cross-reference functionality
@app.route('/test-cross-reference')
def test_cross_reference():
    """Test page for cross-reference functionality."""
    with open('test_cross_reference.html', 'r') as f:
        return f.read()

# Home route
@app.route('/')
def index():
    novels = world_state.get_all('novels')
    return render_template('index.html', novels=novels)

# Novel routes
@app.route('/create_novel', methods=['GET', 'POST'])
def create_novel():
    if request.method == 'POST':
        novel_id = str(uuid.uuid4())
        new_novel = {
            'title': request.form['title'],
            'description': request.form['description'],
            'genre': request.form['genre']
        }
        
        success = world_state.add_or_update('novels', novel_id, new_novel)
        if success:
            flash('Novel created successfully! Worldbuilding sections are now available.', 'success')
            return redirect(url_for('novel_worldbuilding', novel_id=novel_id))
        else:
            flash('Failed to create novel. Please try again.', 'error')
    
    return render_template('create_novel.html')

@app.route('/novel/<novel_id>')
def novel_detail(novel_id):
    novel = world_state.get('novels', novel_id)
    if not novel:
        flash('Novel not found', 'error')
        return redirect(url_for('index'))

    # Only pass basic novel information for simplified view
    return render_template('novel_detail.html', novel=novel)

@app.route('/edit_novel/<novel_id>', methods=['GET', 'POST'])
def edit_novel(novel_id):
    novel = world_state.get('novels', novel_id)
    if not novel:
        flash('Novel not found', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        updated_novel = {
            'title': request.form['title'],
            'description': request.form['description'],
            'genre': request.form['genre']
        }
        
        success = world_state.add_or_update('novels', novel_id, updated_novel)
        if success:
            flash('Novel updated successfully!', 'success')
            return redirect(url_for('novel_detail', novel_id=novel_id))
        else:
            flash('Failed to update novel. Please try again.', 'error')
    
    return render_template('edit_novel.html', novel=novel)

@app.route('/delete_novel/<novel_id>', methods=['POST'])
def delete_novel(novel_id):
    success = world_state.delete_novel_and_related(novel_id)
    if success:
        flash('Novel and all related data deleted successfully!', 'success')
    else:
        flash('Failed to delete novel. Please try again.', 'error')
    return redirect(url_for('index'))

# Novel-specific worldbuilding routes
@app.route('/novel/<novel_id>/worldbuilding')
def novel_worldbuilding(novel_id):
    """Worldbuilding hub for a specific novel."""
    novel = world_state.get('novels', novel_id)
    if not novel:
        flash('Novel not found', 'error')
        return redirect(url_for('index'))
    
    # Get all entities for this novel
    novel_entities = world_state.get_entities_by_novel(novel_id)
    
    return render_template('novel_worldbuilding.html', 
                         novel=novel,
                         characters=novel_entities.get('characters', []),
                         locations=novel_entities.get('locations', []),
                         lore=novel_entities.get('lore', []))

@app.route('/novel/<novel_id>/worldbuilding/characters')
def novel_worldbuilding_characters(novel_id):
    """Characters section within worldbuilding for a specific novel."""
    novel = world_state.get('novels', novel_id)
    if not novel:
        flash('Novel not found', 'error')
        return redirect(url_for('index'))
    
    novel_entities = world_state.get_entities_by_novel(novel_id)
    characters = novel_entities.get('characters', [])
    
    return render_template('novel_worldbuilding_characters.html', 
                         novel=novel,
                         characters=characters)

@app.route('/novel/<novel_id>/worldbuilding/characters/create', methods=['GET', 'POST'])
def novel_create_character(novel_id):
    """Create a character for a specific novel using AI assistance."""
    novel = world_state.get('novels', novel_id)
    if not novel:
        flash('Novel not found', 'error')
        return redirect(url_for('index'))

    if request.method == 'POST':
        character_name = request.form['name'].strip()
        user_prompt = request.form['prompt'].strip()

        if not character_name or not user_prompt:
            flash('Please provide both a character name and description.', 'error')
            return render_template('novel_create_character.html', novel=novel)

        try:
            # Use AI to create detailed character
            novel_context = {
                'title': novel.get('title'),
                'genre': novel.get('genre'),
                'description': novel.get('description')
            }

            character_data = character_creator.create_character(
                name=character_name,
                user_prompt=user_prompt,
                novel_context=novel_context
            )

            # Add novel_id to character data
            character_data['novel_id'] = novel_id

            # Save to database
            character_id = str(uuid.uuid4())
            success = world_state.add_or_update('characters', character_id, character_data)

            if success:
                if character_data.get('ai_generated'):
                    flash(f'Character "{character_name}" created successfully using AI!', 'success')
                else:
                    flash(f'Character "{character_name}" created (AI unavailable, basic version created).', 'warning')
                return redirect(url_for('novel_character_detail', novel_id=novel_id, character_id=character_id))
            else:
                flash('Failed to save character. Please try again.', 'error')

        except Exception as e:
            logger.error(f"Character creation error: {e}")
            flash('An error occurred while creating the character. Please try again.', 'error')

    return render_template('novel_create_character.html',
                         novel=novel,
                         ai_available=character_creator.is_available())

@app.route('/novel/<novel_id>/worldbuilding/characters/<character_id>')
def novel_character_detail(novel_id, character_id):
    """Character detail page within novel worldbuilding."""
    novel = world_state.get('novels', novel_id)
    character = world_state.get('characters', character_id)
    
    if not novel or not character or character.get('novel_id') != novel_id:
        flash('Character not found in this novel', 'error')
        return redirect(url_for('novel_worldbuilding_characters', novel_id=novel_id))
    
    return render_template('novel_character_detail.html', novel=novel, character=character)

@app.route('/novel/<novel_id>/worldbuilding/characters/<character_id>/edit', methods=['GET', 'POST'])
def novel_edit_character(novel_id, character_id):
    """Edit a character within novel worldbuilding."""
    novel = world_state.get('novels', novel_id)
    character = world_state.get('characters', character_id)
    
    if not novel or not character or character.get('novel_id') != novel_id:
        flash('Character not found in this novel', 'error')
        return redirect(url_for('novel_worldbuilding_characters', novel_id=novel_id))
    
    if request.method == 'POST':
        edit_request = request.form.get('edit_request', '').strip()
        character_tags = request.form.get('character_tags', '').strip()

        # Handle tags update (can be combined with AI editing)
        if character_tags:
            # Parse tags from comma-separated string
            tags = [tag.strip() for tag in character_tags.split(',') if tag.strip()]

            # Add occupation as a tag if it exists and isn't already in tags
            if character.get('occupation') and character['occupation'] not in tags:
                tags.append(character['occupation'])

            # Update character with new tags
            character['tags'] = tags
            character['updated_at'] = datetime.now().isoformat()

            # Save tags update
            world_state.add_or_update('characters', character_id, character)

        # Handle AI editing if requested
        if edit_request:
            try:
                # Use AI to edit the character
                novel_context = {
                    'title': novel.get('title'),
                    'genre': novel.get('genre'),
                    'description': novel.get('description')
                }

                updated_character = character_editor.edit_character(
                    current_character=character,
                    edit_request=edit_request,
                    novel_context=novel_context
                )

                # Preserve manually edited tags if they were updated
                if character_tags:
                    updated_character['tags'] = character['tags']

                # Add occupation as a tag if it's not already there
                if updated_character.get('occupation'):
                    if 'tags' not in updated_character:
                        updated_character['tags'] = []
                    if updated_character['occupation'] not in updated_character['tags']:
                        updated_character['tags'].append(updated_character['occupation'])

                # Save the updated character
                success = world_state.add_or_update('characters', character_id, updated_character)

                if success:
                    if updated_character.get('ai_edited'):
                        flash(f'Character "{character.get("name")}" updated successfully using AI!', 'success')
                    else:
                        flash(f'Character "{character.get("name")}" updated (AI unavailable, no changes made).', 'warning')
                    return redirect(url_for('novel_character_detail', novel_id=novel_id, character_id=character_id))
                else:
                    flash('Failed to save character changes. Please try again.', 'error')

            except Exception as e:
                logger.error(f"Character editing error: {e}")
                flash('An error occurred while editing the character. Please try again.', 'error')
        elif character_tags:
            # Tags-only update was successful
            flash('Character tags updated successfully!', 'success')
            return redirect(url_for('novel_character_detail', novel_id=novel_id, character_id=character_id))
        else:
            flash('Please provide either an edit request or update the tags.', 'error')

    # Don't load AI suggestions immediately for faster page load
    # They'll be loaded on-demand via AJAX if user clicks "Get Suggestions"

    return render_template('novel_edit_character.html',
                         novel=novel,
                         character=character,
                         ai_available=character_editor.is_available(),
                         suggestions=[])

@app.route('/novel/<novel_id>/worldbuilding/characters/<character_id>/edit/preview', methods=['POST'])
def novel_character_edit_preview(novel_id, character_id):
    """Get a preview of character changes without saving."""
    novel = world_state.get('novels', novel_id)
    character = world_state.get('characters', character_id)

    if not novel or not character or character.get('novel_id') != novel_id:
        return jsonify({'error': 'Character not found'}), 404

    edit_request = request.json.get('edit_request', '').strip()
    if not edit_request:
        return jsonify({'error': 'No edit request provided'}), 400

    try:
        # Use AI to generate preview (without saving)
        novel_context = {
            'title': novel.get('title'),
            'genre': novel.get('genre'),
            'description': novel.get('description')
        }

        updated_character = character_editor.edit_character(
            current_character=character,
            edit_request=edit_request,
            novel_context=novel_context
        )

        # Return only the updated fields for preview
        preview_data = {
            'name': updated_character.get('name'),
            'age': updated_character.get('age'),
            'occupation': updated_character.get('occupation'),
            'role': updated_character.get('role'),
            'description': updated_character.get('description'),
            'personality': updated_character.get('personality'),
            'backstory': updated_character.get('backstory'),
            'tags': updated_character.get('tags', []),
            'ai_edited': updated_character.get('ai_edited', False)
        }

        return jsonify(preview_data)

    except Exception as e:
        logger.error(f"Character preview error: {e}")
        return jsonify({'error': 'Failed to generate preview'}), 500

@app.route('/novel/<novel_id>/worldbuilding/characters/<character_id>/edit/fast', methods=['POST'])
def novel_character_edit_fast(novel_id, character_id):
    """Fast character editing without preview - direct save."""
    novel = world_state.get('novels', novel_id)
    character = world_state.get('characters', character_id)

    if not novel or not character or character.get('novel_id') != novel_id:
        return jsonify({'error': 'Character not found'}), 404

    edit_request = request.json.get('edit_request', '').strip()
    if not edit_request:
        return jsonify({'error': 'No edit request provided'}), 400

    try:
        # Use AI to edit directly (no preview step)
        novel_context = {
            'title': novel.get('title'),
            'genre': novel.get('genre'),
            'description': novel.get('description')
        }

        updated_character = character_editor.edit_character(
            current_character=character,
            edit_request=edit_request,
            novel_context=novel_context
        )

        # Save immediately
        success = world_state.add_or_update('characters', character_id, updated_character)

        if success:
            return jsonify({
                'success': True,
                'message': 'Character updated successfully!',
                'character': {
                    'name': updated_character.get('name'),
                    'age': updated_character.get('age'),
                    'occupation': updated_character.get('occupation'),
                    'role': updated_character.get('role'),
                    'description': updated_character.get('description'),
                    'personality': updated_character.get('personality'),
                    'backstory': updated_character.get('backstory'),
                    'tags': updated_character.get('tags', [])
                }
            })
        else:
            return jsonify({'error': 'Failed to save character'}), 500

    except Exception as e:
        logger.error(f"Fast character edit error: {e}")
        return jsonify({'error': 'Failed to edit character'}), 500

@app.route('/novel/<novel_id>/worldbuilding/characters/<character_id>/suggestions', methods=['GET'])
def get_character_suggestions(novel_id, character_id):
    """Get AI suggestions for character improvements (lazy-loaded)."""
    novel = world_state.get('novels', novel_id)
    character = world_state.get('characters', character_id)

    if not novel or not character or character.get('novel_id') != novel_id:
        return jsonify({'error': 'Character not found'}), 404

    if not character_editor.is_available():
        return jsonify({'suggestions': []})

    try:
        suggestions = character_editor.suggest_improvements(character)
        return jsonify({'suggestions': suggestions})
    except Exception as e:
        logger.error(f"Character suggestions error: {e}")
        return jsonify({'suggestions': []})

@app.route('/novel/<novel_id>/worldbuilding/characters/<character_id>/edit/tags', methods=['POST'])
def novel_character_edit_tags(novel_id, character_id):
    """Update character tags only."""
    novel = world_state.get('novels', novel_id)
    character = world_state.get('characters', character_id)

    if not novel or not character or character.get('novel_id') != novel_id:
        return jsonify({'error': 'Character not found'}), 404

    try:
        tags = request.json.get('tags', [])

        # Add occupation as a tag if it exists and isn't already in tags
        if character.get('occupation') and character['occupation'] not in tags:
            tags.append(character['occupation'])

        # Update character with new tags
        character['tags'] = tags
        character['updated_at'] = datetime.now().isoformat()

        # Save tags update
        success = world_state.add_or_update('characters', character_id, character)

        if success:
            return jsonify({'success': True, 'tags': tags})
        else:
            return jsonify({'error': 'Failed to save tags'}), 500

    except Exception as e:
        logger.error(f"Tags update error: {e}")
        return jsonify({'error': 'Failed to update tags'}), 500

@app.route('/novel/<novel_id>/worldbuilding/characters/<character_id>/delete', methods=['POST'])
def novel_delete_character(novel_id, character_id):
    """Delete a character within novel worldbuilding."""
    character = world_state.get('characters', character_id)
    
    if not character or character.get('novel_id') != novel_id:
        flash('Character not found in this novel', 'error')
        return redirect(url_for('novel_worldbuilding_characters', novel_id=novel_id))
    
    success = world_state.delete('characters', character_id)
    if success:
        flash('Character deleted successfully!', 'success')
    else:
        flash('Failed to delete character. Please try again.', 'error')
    
    return redirect(url_for('novel_worldbuilding_characters', novel_id=novel_id))

# Novel-specific locations routes
@app.route('/novel/<novel_id>/worldbuilding/locations')
def novel_worldbuilding_locations(novel_id):
    """Locations section within worldbuilding for a specific novel."""
    novel = world_state.get('novels', novel_id)
    if not novel:
        flash('Novel not found', 'error')
        return redirect(url_for('index'))

    novel_entities = world_state.get_entities_by_novel(novel_id)
    locations = novel_entities.get('locations', [])

    return render_template('novel_worldbuilding_locations.html',
                         novel=novel,
                         locations=locations)

@app.route('/novel/<novel_id>/worldbuilding/locations/create', methods=['GET', 'POST'])
def novel_create_location(novel_id):
    """Create a location for a specific novel using AI assistance."""
    novel = world_state.get('novels', novel_id)
    if not novel:
        flash('Novel not found', 'error')
        return redirect(url_for('index'))

    if request.method == 'POST':
        # Handle AI-generated location data
        location_data_json = request.form.get('location_data')
        if location_data_json:
            try:
                import json
                location_data = json.loads(location_data_json)

                # Add novel_id to location data
                location_data['novel_id'] = novel_id

                # Save to database
                location_id = str(uuid.uuid4())
                success = world_state.add_or_update('locations', location_id, location_data)

                if success:
                    if location_data.get('ai_generated'):
                        flash(f'Location "{location_data.get("name")}" created successfully using AI!', 'success')
                    else:
                        flash(f'Location "{location_data.get("name")}" created (AI unavailable, basic version created).', 'warning')
                    return redirect(url_for('novel_location_detail', novel_id=novel_id, location_id=location_id))
                else:
                    flash('Failed to save location. Please try again.', 'error')

            except Exception as e:
                logger.error(f"Location creation error: {e}")
                flash('An error occurred while creating the location. Please try again.', 'error')

    return render_template('novel_create_location.html',
                         novel=novel,
                         ai_available=location_creator.is_available())

@app.route('/novel/<novel_id>/worldbuilding/locations/<location_id>')
def novel_location_detail(novel_id, location_id):
    """Location detail page within novel worldbuilding."""
    novel = world_state.get('novels', novel_id)
    location = world_state.get('locations', location_id)

    if not novel or not location or location.get('novel_id') != novel_id:
        flash('Location not found in this novel', 'error')
        return redirect(url_for('novel_worldbuilding_locations', novel_id=novel_id))

    return render_template('novel_location_detail.html', novel=novel, location=location)

@app.route('/novel/<novel_id>/worldbuilding/locations/<location_id>/edit', methods=['GET', 'POST'])
def novel_edit_location(novel_id, location_id):
    """Edit a location within novel worldbuilding using AI assistance."""
    novel = world_state.get('novels', novel_id)
    location = world_state.get('locations', location_id)

    if not novel or not location or location.get('novel_id') != novel_id:
        flash('Location not found in this novel', 'error')
        return redirect(url_for('novel_worldbuilding_locations', novel_id=novel_id))

    if request.method == 'POST':
        edit_request = request.form.get('edit_request', '').strip()
        location_tags = request.form.get('location_tags', '').strip()

        # Handle tags update (can be combined with AI editing)
        if location_tags:
            # Parse tags from comma-separated string
            tags = [tag.strip() for tag in location_tags.split(',') if tag.strip()]

            # Update location with new tags
            location['tags'] = tags
            location['updated_at'] = datetime.now().isoformat()

            # Save tags update
            world_state.add_or_update('locations', location_id, location)

        # Handle AI editing if requested
        if edit_request:
            try:
                # Use AI to edit the location
                novel_context = {
                    'title': novel.get('title'),
                    'genre': novel.get('genre'),
                    'description': novel.get('description')
                }

                updated_location = location_editor.edit_location(
                    current_location=location,
                    edit_request=edit_request,
                    novel_context=novel_context
                )

                # Preserve manually edited tags if they were updated
                if location_tags:
                    updated_location['tags'] = location['tags']

                # Preserve important metadata
                updated_location['novel_id'] = novel_id
                updated_location['id'] = location_id
                if 'created_at' in location:
                    updated_location['created_at'] = location['created_at']
                updated_location['updated_at'] = datetime.now().isoformat()

                # Save the updated location
                success = world_state.add_or_update('locations', location_id, updated_location)

                if success:
                    if updated_location.get('ai_edited'):
                        flash(f'Location "{location.get("name")}" updated successfully using AI!', 'success')
                    else:
                        flash(f'Location "{location.get("name")}" updated (AI unavailable, no changes made).', 'warning')
                    return redirect(url_for('novel_location_detail', novel_id=novel_id, location_id=location_id))
                else:
                    flash('Failed to save location changes. Please try again.', 'error')

            except Exception as e:
                logger.error(f"Location editing error: {e}")
                flash('An error occurred while editing the location. Please try again.', 'error')
        elif location_tags:
            # Tags-only update was successful
            flash('Location tags updated successfully!', 'success')
            return redirect(url_for('novel_location_detail', novel_id=novel_id, location_id=location_id))
        else:
            flash('Please provide either an edit request or update the tags.', 'error')

    return render_template('novel_edit_location.html',
                         novel=novel,
                         location=location,
                         ai_available=location_editor.is_available())

@app.route('/novel/<novel_id>/worldbuilding/locations/<location_id>/delete', methods=['POST'])
def novel_delete_location(novel_id, location_id):
    """Delete a location within novel worldbuilding."""
    location = world_state.get('locations', location_id)

    if not location or location.get('novel_id') != novel_id:
        flash('Location not found in this novel', 'error')
        return redirect(url_for('novel_worldbuilding_locations', novel_id=novel_id))

    success = world_state.delete('locations', location_id)
    if success:
        flash('Location deleted successfully!', 'success')
    else:
        flash('Failed to delete location. Please try again.', 'error')

    return redirect(url_for('novel_worldbuilding_locations', novel_id=novel_id))

# Novel-specific lore routes
@app.route('/novel/<novel_id>/worldbuilding/lore')
def novel_worldbuilding_lore(novel_id):
    """Lore section within worldbuilding for a specific novel."""
    novel = world_state.get('novels', novel_id)
    if not novel:
        flash('Novel not found', 'error')
        return redirect(url_for('index'))

    novel_entities = world_state.get_entities_by_novel(novel_id)
    lore = novel_entities.get('lore', [])

    return render_template('novel_worldbuilding_lore.html',
                         novel=novel,
                         lore=lore)

@app.route('/novel/<novel_id>/worldbuilding/lore/create', methods=['GET', 'POST'])
def novel_create_lore(novel_id):
    """Create a lore entry for a specific novel using AI assistance."""
    novel = world_state.get('novels', novel_id)
    if not novel:
        flash('Novel not found', 'error')
        return redirect(url_for('index'))

    if request.method == 'POST':
        lore_title = request.form['title'].strip()
        user_prompt = request.form['prompt'].strip()

        if not lore_title or not user_prompt:
            flash('Please provide both a lore title and description.', 'error')
            return render_template('novel_create_lore.html', novel=novel)

        try:
            # Prepare novel context for AI
            novel_context = {
                'title': novel.get('title'),
                'genre': novel.get('genre'),
                'description': novel.get('description')
            }

            # Use AI to create detailed lore entry
            lore_data = lore_creator.create_lore(lore_title, user_prompt, novel_context)

            # Add novel_id to lore data
            lore_data['novel_id'] = novel_id

            # Save to database
            lore_id = str(uuid.uuid4())
            success = world_state.add_or_update('lore', lore_id, lore_data)

            if success:
                if lore_data.get('ai_generated'):
                    flash(f'Lore entry "{lore_title}" created successfully using AI!', 'success')
                else:
                    flash(f'Lore entry "{lore_title}" created (AI unavailable, basic version created).', 'warning')
                return redirect(url_for('novel_lore_detail', novel_id=novel_id, lore_id=lore_id))
            else:
                flash('Failed to save lore entry. Please try again.', 'error')

        except Exception as e:
            logger.error(f"Lore creation error: {e}")
            flash('An error occurred while creating the lore entry. Please try again.', 'error')

    return render_template('novel_create_lore.html',
                         novel=novel,
                         ai_available=lore_creator.is_available())

@app.route('/novel/<novel_id>/worldbuilding/lore/<lore_id>')
def novel_lore_detail(novel_id, lore_id):
    """Lore detail page within novel worldbuilding."""
    novel = world_state.get('novels', novel_id)
    lore_entry = world_state.get('lore', lore_id)

    if not novel or not lore_entry or lore_entry.get('novel_id') != novel_id:
        flash('Lore entry not found in this novel', 'error')
        return redirect(url_for('novel_worldbuilding_lore', novel_id=novel_id))

    return render_template('novel_lore_detail.html', novel=novel, lore=lore_entry)

@app.route('/novel/<novel_id>/worldbuilding/lore/<lore_id>/edit', methods=['GET', 'POST'])
def novel_edit_lore(novel_id, lore_id):
    """Edit a lore entry within novel worldbuilding using AI assistance."""
    novel = world_state.get('novels', novel_id)
    lore_entry = world_state.get('lore', lore_id)

    if not novel or not lore_entry or lore_entry.get('novel_id') != novel_id:
        flash('Lore entry not found in this novel', 'error')
        return redirect(url_for('novel_worldbuilding_lore', novel_id=novel_id))

    if request.method == 'POST':
        edit_request = request.form.get('edit_request', '').strip()
        lore_tags = request.form.get('lore_tags', '').strip()

        # Handle tags update (can be combined with AI editing)
        if lore_tags:
            # Parse tags from comma-separated string
            tags = [tag.strip() for tag in lore_tags.split(',') if tag.strip()]

            # Update lore entry with new tags
            lore_entry['tags'] = tags
            lore_entry['updated_at'] = datetime.now().isoformat()

            # Save tags update
            world_state.add_or_update('lore', lore_id, lore_entry)

        # Handle AI editing if requested
        if edit_request:
            try:
                # Use AI to edit the lore entry
                novel_context = {
                    'title': novel.get('title'),
                    'genre': novel.get('genre'),
                    'description': novel.get('description')
                }

                updated_lore = lore_editor.edit_lore(
                    current_lore=lore_entry,
                    edit_request=edit_request,
                    novel_context=novel_context
                )

                # Preserve manually edited tags if they were updated
                if lore_tags:
                    updated_lore['tags'] = lore_entry['tags']

                # Save the updated lore entry
                success = world_state.add_or_update('lore', lore_id, updated_lore)

                if success:
                    if updated_lore.get('ai_edited'):
                        flash(f'Lore entry "{lore_entry.get("title")}" updated successfully using AI!', 'success')
                    else:
                        flash(f'Lore entry "{lore_entry.get("title")}" updated (AI unavailable, no changes made).', 'warning')
                    return redirect(url_for('novel_lore_detail', novel_id=novel_id, lore_id=lore_id))
                else:
                    flash('Failed to save lore entry changes. Please try again.', 'error')

            except Exception as e:
                logger.error(f"Lore editing error: {e}")
                flash('An error occurred while editing the lore entry. Please try again.', 'error')
        elif lore_tags:
            # Tags-only update was successful
            flash('Lore entry tags updated successfully!', 'success')
            return redirect(url_for('novel_lore_detail', novel_id=novel_id, lore_id=lore_id))
        else:
            flash('Please provide either an edit request or update the tags.', 'error')

    # Don't load AI suggestions immediately for faster page load
    # They'll be loaded on-demand via AJAX if user clicks "Get Suggestions"

    return render_template('novel_edit_lore.html',
                         novel=novel,
                         lore=lore_entry,
                         ai_available=lore_editor.is_available(),
                         suggestions=[])

@app.route('/novel/<novel_id>/worldbuilding/lore/<lore_id>/edit/preview', methods=['POST'])
def novel_lore_edit_preview(novel_id, lore_id):
    """Get a preview of lore changes without saving."""
    novel = world_state.get('novels', novel_id)
    lore_entry = world_state.get('lore', lore_id)

    if not novel or not lore_entry or lore_entry.get('novel_id') != novel_id:
        return jsonify({'error': 'Lore entry not found'}), 404

    edit_request = request.json.get('edit_request', '').strip()
    if not edit_request:
        return jsonify({'error': 'No edit request provided'}), 400

    try:
        # Use AI to generate preview (without saving)
        novel_context = {
            'title': novel.get('title'),
            'genre': novel.get('genre'),
            'description': novel.get('description')
        }

        updated_lore = lore_editor.edit_lore(
            current_lore=lore_entry,
            edit_request=edit_request,
            novel_context=novel_context
        )

        # Return only the updated fields for preview
        preview_data = {
            'title': updated_lore.get('title'),
            'category': updated_lore.get('category'),
            'description': updated_lore.get('description'),
            'details': updated_lore.get('details'),
            'significance': updated_lore.get('significance'),
            'connections': updated_lore.get('connections'),
            'tags': updated_lore.get('tags', []),
            'ai_edited': updated_lore.get('ai_edited', False)
        }

        return jsonify(preview_data)

    except Exception as e:
        logger.error(f"Lore preview error: {e}")
        return jsonify({'error': 'Failed to generate preview'}), 500

@app.route('/novel/<novel_id>/worldbuilding/lore/<lore_id>/suggestions', methods=['GET'])
def get_lore_suggestions(novel_id, lore_id):
    """Get AI suggestions for lore improvements (lazy-loaded)."""
    novel = world_state.get('novels', novel_id)
    lore_entry = world_state.get('lore', lore_id)

    if not novel or not lore_entry or lore_entry.get('novel_id') != novel_id:
        return jsonify({'error': 'Lore entry not found'}), 404

    if not lore_editor.is_available():
        return jsonify({'suggestions': []})

    try:
        suggestions = lore_editor.suggest_improvements(lore_entry)
        return jsonify({'suggestions': suggestions})
    except Exception as e:
        logger.error(f"Lore suggestions error: {e}")
        return jsonify({'suggestions': []})

@app.route('/novel/<novel_id>/worldbuilding/lore/<lore_id>/edit/fast', methods=['POST'])
def novel_lore_edit_fast(novel_id, lore_id):
    """Fast lore editing without preview - direct save."""
    novel = world_state.get('novels', novel_id)
    lore_entry = world_state.get('lore', lore_id)

    if not novel or not lore_entry or lore_entry.get('novel_id') != novel_id:
        return jsonify({'error': 'Lore entry not found'}), 404

    edit_request = request.json.get('edit_request', '').strip()
    if not edit_request:
        return jsonify({'error': 'No edit request provided'}), 400

    try:
        # Use AI to edit directly (no preview step)
        novel_context = {
            'title': novel.get('title'),
            'genre': novel.get('genre'),
            'description': novel.get('description')
        }

        updated_lore = lore_editor.edit_lore(
            current_lore=lore_entry,
            edit_request=edit_request,
            novel_context=novel_context
        )

        # Save immediately
        success = world_state.add_or_update('lore', lore_id, updated_lore)

        if success:
            return jsonify({
                'success': True,
                'message': 'Lore updated successfully!',
                'lore': {
                    'title': updated_lore.get('title'),
                    'category': updated_lore.get('category'),
                    'description': updated_lore.get('description'),
                    'details': updated_lore.get('details'),
                    'significance': updated_lore.get('significance'),
                    'connections': updated_lore.get('connections'),
                    'tags': updated_lore.get('tags', [])
                }
            })
        else:
            return jsonify({'error': 'Failed to save lore'}), 500

    except Exception as e:
        logger.error(f"Fast lore edit error: {e}")
        return jsonify({'error': 'Failed to edit lore'}), 500

@app.route('/novel/<novel_id>/worldbuilding/lore/<lore_id>/edit/tags', methods=['POST'])
def novel_lore_edit_tags(novel_id, lore_id):
    """Update lore tags only."""
    novel = world_state.get('novels', novel_id)
    lore_entry = world_state.get('lore', lore_id)

    if not novel or not lore_entry or lore_entry.get('novel_id') != novel_id:
        return jsonify({'error': 'Lore entry not found'}), 404

    try:
        # Get tags from form data
        tags_input = request.form.get('lore_tags', '').strip()
        tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()] if tags_input else []

        # Update lore entry with new tags
        lore_entry['tags'] = tags
        lore_entry['updated_at'] = datetime.now().isoformat()

        # Save to database
        success = world_state.add_or_update('lore', lore_id, lore_entry)

        if success:
            return jsonify({
                'success': True,
                'message': 'Tags updated successfully!',
                'tags': tags
            })
        else:
            return jsonify({'error': 'Failed to update tags'}), 500

    except Exception as e:
        logger.error(f"Lore tags update error: {e}")
        return jsonify({'error': 'Failed to update tags'}), 500

@app.route('/novel/<novel_id>/worldbuilding/lore/<lore_id>/delete', methods=['POST'])
def novel_delete_lore(novel_id, lore_id):
    """Delete a lore entry within novel worldbuilding."""
    lore_entry = world_state.get('lore', lore_id)

    if not lore_entry or lore_entry.get('novel_id') != novel_id:
        flash('Lore entry not found in this novel', 'error')
        return redirect(url_for('novel_worldbuilding_lore', novel_id=novel_id))

    success = world_state.delete('lore', lore_id)
    if success:
        flash('Lore entry deleted successfully!', 'success')
    else:
        flash('Failed to delete lore entry. Please try again.', 'error')

    return redirect(url_for('novel_worldbuilding_lore', novel_id=novel_id))

# Location API routes
@app.route('/api/locations/create', methods=['POST'])
def api_create_location():
    """API endpoint for creating locations with AI."""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        prompt = data.get('prompt', '').strip()
        novel_id = data.get('novel_id', '').strip()
        novel_context = data.get('novel_context', {})

        if not name or not prompt or not novel_id:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        # Verify novel exists
        novel = world_state.get('novels', novel_id)
        if not novel:
            return jsonify({'success': False, 'error': 'Novel not found'}), 404

        # Use AI to create detailed location
        location_data = location_creator.create_location(name, prompt, novel_context)

        return jsonify({
            'success': True,
            'location': location_data
        })

    except Exception as e:
        logger.error(f"API location creation error: {e}")
        return jsonify({'success': False, 'error': 'Failed to create location'}), 500

@app.route('/api/locations/<location_id>/ai-edit', methods=['POST'])
def api_edit_location(location_id):
    """API endpoint for editing locations with AI."""
    try:
        data = request.get_json()
        edit_request = data.get('edit_request', '').strip()
        novel_context = data.get('novel_context', {})

        if not edit_request:
            return jsonify({'success': False, 'error': 'No edit request provided'}), 400

        # Get current location
        location = world_state.get('locations', location_id)
        if not location:
            return jsonify({'success': False, 'error': 'Location not found'}), 404

        # Use AI to edit the location
        updated_location = location_editor.edit_location(location, edit_request, novel_context)

        return jsonify({
            'success': True,
            'location': updated_location
        })

    except Exception as e:
        logger.error(f"API location edit error: {e}")
        return jsonify({'success': False, 'error': 'Failed to edit location'}), 500

@app.route('/api/locations/<location_id>/suggestions', methods=['GET'])
def api_get_location_suggestions(location_id):
    """API endpoint for getting location improvement suggestions."""
    try:
        location = world_state.get('locations', location_id)
        if not location:
            return jsonify({'success': False, 'error': 'Location not found'}), 404

        if not location_editor.is_available():
            return jsonify({'success': True, 'suggestions': []})

        suggestions = location_editor.suggest_improvements(location)
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })

    except Exception as e:
        logger.error(f"API location suggestions error: {e}")
        return jsonify({'success': True, 'suggestions': []})

@app.route('/novel/<novel_id>/worldbuilding/locations/<location_id>/edit/preview', methods=['POST'])
def novel_location_edit_preview(novel_id, location_id):
    """Get a preview of location changes without saving."""
    novel = world_state.get('novels', novel_id)
    location = world_state.get('locations', location_id)

    if not novel or not location or location.get('novel_id') != novel_id:
        return jsonify({'error': 'Location not found'}), 404

    edit_request = request.json.get('edit_request', '').strip()
    if not edit_request:
        return jsonify({'error': 'No edit request provided'}), 400

    try:
        # Use AI to edit the location
        novel_context = {
            'title': novel.get('title'),
            'genre': novel.get('genre'),
            'description': novel.get('description')
        }

        updated_location = location_editor.edit_location(location, edit_request, novel_context)

        # Return only the updated fields for preview
        preview_data = {
            'name': updated_location.get('name'),
            'type': updated_location.get('type'),
            'climate': updated_location.get('climate'),
            'description': updated_location.get('description'),
            'geography': updated_location.get('geography'),
            'culture': updated_location.get('culture'),
            'history': updated_location.get('history'),
            'economy': updated_location.get('economy'),
            'notable_features': updated_location.get('notable_features'),
            'tags': updated_location.get('tags', []),
            'ai_edited': updated_location.get('ai_edited', False)
        }

        return jsonify(preview_data)

    except Exception as e:
        logger.error(f"Location preview error: {e}")
        return jsonify({'error': 'Failed to generate preview'}), 500

@app.route('/novel/<novel_id>/worldbuilding/locations/<location_id>/edit/fast', methods=['POST'])
def novel_location_edit_fast(novel_id, location_id):
    """Fast location editing without preview - direct save."""
    novel = world_state.get('novels', novel_id)
    location = world_state.get('locations', location_id)

    if not novel or not location or location.get('novel_id') != novel_id:
        return jsonify({'error': 'Location not found'}), 404

    edit_request = request.json.get('edit_request', '').strip()
    if not edit_request:
        return jsonify({'error': 'No edit request provided'}), 400

    try:
        # Use AI to edit directly (no preview step)
        novel_context = {
            'title': novel.get('title'),
            'genre': novel.get('genre'),
            'description': novel.get('description')
        }

        updated_location = location_editor.edit_location(location, edit_request, novel_context)

        # Add metadata
        updated_location['updated_at'] = datetime.now().isoformat()
        updated_location['ai_edited'] = True

        # Save the updated location
        success = world_state.add_or_update('locations', location_id, updated_location)

        if success:
            return jsonify({
                'success': True,
                'message': 'Location updated successfully!',
                'location': {
                    'name': updated_location.get('name'),
                    'type': updated_location.get('type'),
                    'climate': updated_location.get('climate'),
                    'description': updated_location.get('description'),
                    'geography': updated_location.get('geography'),
                    'culture': updated_location.get('culture'),
                    'history': updated_location.get('history'),
                    'economy': updated_location.get('economy'),
                    'notable_features': updated_location.get('notable_features'),
                    'tags': updated_location.get('tags', [])
                }
            })
        else:
            return jsonify({'error': 'Failed to save location'}), 500

    except Exception as e:
        logger.error(f"Fast location edit error: {e}")
        return jsonify({'error': 'Failed to edit location'}), 500

@app.route('/novel/<novel_id>/worldbuilding/locations/<location_id>/edit/tags', methods=['POST'])
def novel_location_edit_tags(novel_id, location_id):
    """Update location tags only."""
    novel = world_state.get('novels', novel_id)
    location = world_state.get('locations', location_id)

    if not novel or not location or location.get('novel_id') != novel_id:
        return jsonify({'error': 'Location not found'}), 404

    try:
        # Get tags from form data
        tags_input = request.form.get('location_tags', '').strip()
        tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()] if tags_input else []

        # Update location with new tags
        location['tags'] = tags
        location['updated_at'] = datetime.now().isoformat()

        # Save to database
        success = world_state.add_or_update('locations', location_id, location)

        if success:
            return jsonify({
                'success': True,
                'message': 'Tags updated successfully!',
                'tags': tags
            })
        else:
            return jsonify({'error': 'Failed to update tags'}), 500

    except Exception as e:
        logger.error(f"Location tags update error: {e}")
        return jsonify({'error': 'Failed to update tags'}), 500

# Novel-specific search route
@app.route('/novel/<novel_id>/worldbuilding/search')
def novel_worldbuilding_search(novel_id):
    """Search section within worldbuilding for a specific novel."""
    novel = world_state.get('novels', novel_id)
    if not novel:
        flash('Novel not found', 'error')
        return redirect(url_for('index'))

    query = request.args.get('q', '')
    entity_types = request.args.getlist('types')

    results = []
    categorized_results = {}

    if query:
        try:
            if entity_types:
                results = semantic_search.search(
                    query=query,
                    entity_types=entity_types,
                    novel_id=novel_id,
                    n_results=20
                )
            else:
                categorized_results = semantic_search.search_by_category(
                    query=query,
                    novel_id=novel_id
                )
        except Exception as e:
            logger.error(f"Search error: {e}")
            flash('Search failed. Please try again.', 'error')

    return render_template('novel_worldbuilding_search.html',
                         novel=novel,
                         query=query,
                         results=results,
                         categorized_results=categorized_results,
                         selected_types=entity_types)



# API routes
@app.route('/api/stats')
def api_stats():
    """Get database statistics."""
    try:
        stats = {
            'novels_count': len(world_state.get_all('novels')),
            'characters_count': len(world_state.get_all('characters')),
            'locations_count': len(world_state.get_all('locations')),
            'lore_count': len(world_state.get_all('lore')),
            'chromadb': {
                'novels_embeddings': world_state.chromadb_manager.get_collection_count('novels'),
                'characters_embeddings': world_state.chromadb_manager.get_collection_count('characters'),
                'locations_embeddings': world_state.chromadb_manager.get_collection_count('locations'),
                'lore_embeddings': world_state.chromadb_manager.get_collection_count('lore')
            }
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({'error': 'Failed to get stats'}), 500

# Cross-reference API routes for entity lists
@app.route('/api/novel/<novel_id>/worldbuilding/characters')
def api_novel_characters(novel_id):
    """API endpoint to get characters for a novel as JSON."""
    try:
        # Verify novel exists
        novel = world_state.get('novels', novel_id)
        if not novel:
            return jsonify({'error': 'Novel not found'}), 404

        # Get characters for this novel
        all_characters = world_state.get_all('characters')
        novel_characters = [char for char in all_characters if char.get('novel_id') == novel_id]

        # Return simplified character data for the API
        characters_data = []
        for char in novel_characters:
            characters_data.append({
                'id': char['id'],
                'name': char.get('name', 'Unnamed Character'),
                'role': char.get('role', ''),
                'description': char.get('description', '')[:100] + '...' if len(char.get('description', '')) > 100 else char.get('description', '')
            })

        return jsonify(characters_data)
    except Exception as e:
        logger.error(f"API characters error: {e}")
        return jsonify({'error': 'Failed to get characters'}), 500

@app.route('/api/novel/<novel_id>/worldbuilding/locations')
def api_novel_locations(novel_id):
    """API endpoint to get locations for a novel as JSON."""
    try:
        # Verify novel exists
        novel = world_state.get('novels', novel_id)
        if not novel:
            return jsonify({'error': 'Novel not found'}), 404

        # Get locations for this novel
        all_locations = world_state.get_all('locations')
        novel_locations = [loc for loc in all_locations if loc.get('novel_id') == novel_id]

        # Return simplified location data for the API
        locations_data = []
        for loc in novel_locations:
            locations_data.append({
                'id': loc['id'],
                'name': loc.get('name', 'Unnamed Location'),
                'type': loc.get('type', ''),
                'description': loc.get('description', '')[:100] + '...' if len(loc.get('description', '')) > 100 else loc.get('description', '')
            })

        return jsonify(locations_data)
    except Exception as e:
        logger.error(f"API locations error: {e}")
        return jsonify({'error': 'Failed to get locations'}), 500

@app.route('/api/novel/<novel_id>/worldbuilding/lore')
def api_novel_lore(novel_id):
    """API endpoint to get lore for a novel as JSON."""
    try:
        # Verify novel exists
        novel = world_state.get('novels', novel_id)
        if not novel:
            return jsonify({'error': 'Novel not found'}), 404

        # Get lore for this novel
        all_lore = world_state.get_all('lore')
        novel_lore = [lore for lore in all_lore if lore.get('novel_id') == novel_id]

        # Return simplified lore data for the API
        lore_data = []
        for lore in novel_lore:
            lore_data.append({
                'id': lore['id'],
                'title': lore.get('title', 'Untitled Lore'),
                'category': lore.get('category', ''),
                'content': lore.get('content', '')[:100] + '...' if len(lore.get('content', '')) > 100 else lore.get('content', '')
            })

        return jsonify(lore_data)
    except Exception as e:
        logger.error(f"API lore error: {e}")
        return jsonify({'error': 'Failed to get lore'}), 500















# Change history and undo routes
@app.route('/novel/<novel_id>/change-history')
def novel_change_history(novel_id):
    """Get change history for a novel."""
    try:
        # Verify novel exists
        novel = world_state.get('novels', novel_id)
        if not novel:
            return jsonify({'success': False, 'error': 'Novel not found'}), 404

        # Get change history manager
        from utils.change_history import get_change_history_manager
        history_manager = get_change_history_manager()

        # Get history
        limit = request.args.get('limit', 50, type=int)
        history = history_manager.get_history(novel_id, limit)

        # Convert to JSON-serializable format
        history_data = [session.to_dict() for session in history]

        return jsonify({
            'success': True,
            'history': history_data,
            'total_sessions': len(history_data)
        })

    except Exception as e:
        logger.error(f"Change history error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get change history'}), 500

@app.route('/novel/<novel_id>/change-history/entity/<entity_type>/<entity_id>')
def entity_change_history(novel_id, entity_type, entity_id):
    """Get change history for a specific entity."""
    try:
        # Verify novel exists
        novel = world_state.get('novels', novel_id)
        if not novel:
            return jsonify({'success': False, 'error': 'Novel not found'}), 404

        # Verify entity exists
        entity = world_state.get(entity_type, entity_id)
        if not entity or entity.get('novel_id') != novel_id:
            return jsonify({'success': False, 'error': 'Entity not found'}), 404

        # Get change history manager
        from utils.change_history import get_change_history_manager
        history_manager = get_change_history_manager()

        # Get entity history
        limit = request.args.get('limit', 20, type=int)
        history = history_manager.get_entity_history(entity_type, entity_id, limit)

        # Convert to JSON-serializable format
        history_data = [change.to_dict() for change in history]

        return jsonify({
            'success': True,
            'history': history_data,
            'total_changes': len(history_data)
        })

    except Exception as e:
        logger.error(f"Entity change history error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get entity change history'}), 500

@app.route('/novel/<novel_id>/undo/change/<change_id>', methods=['POST'])
def undo_change(novel_id, change_id):
    """Undo a specific change."""
    try:
        # Verify novel exists
        novel = world_state.get('novels', novel_id)
        if not novel:
            return jsonify({'success': False, 'error': 'Novel not found'}), 404

        # Get change history manager
        from utils.change_history import get_change_history_manager
        history_manager = get_change_history_manager()

        # Undo the change
        success = history_manager.undo_change(change_id, world_state)

        if success:
            return jsonify({
                'success': True,
                'message': 'Change undone successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to undo change - conflicts detected or change not found'
            }), 400

    except Exception as e:
        logger.error(f"Undo change error: {e}")
        return jsonify({'success': False, 'error': 'Failed to undo change'}), 500

@app.route('/novel/<novel_id>/undo/session/<session_id>', methods=['POST'])
def undo_session(novel_id, session_id):
    """Undo an entire session of changes."""
    try:
        # Verify novel exists
        novel = world_state.get('novels', novel_id)
        if not novel:
            return jsonify({'success': False, 'error': 'Novel not found'}), 404

        # Get change history manager
        from utils.change_history import get_change_history_manager
        history_manager = get_change_history_manager()

        # Undo the session
        success = history_manager.undo_session(session_id, world_state)

        if success:
            return jsonify({
                'success': True,
                'message': 'Session undone successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to undo session - some changes could not be reverted'
            }), 400

    except Exception as e:
        logger.error(f"Undo session error: {e}")
        return jsonify({'success': False, 'error': 'Failed to undo session'}), 500

# AJAX Save Endpoints for Real-time Editing
@app.route('/api/characters/<character_id>/save', methods=['POST'])
def api_save_character(character_id):
    """API endpoint for saving character changes via AJAX."""
    try:
        # Get form data
        edit_request = request.form.get('edit_request', '').strip()
        character_tags = request.form.get('character_tags', '').strip()

        # Get current character
        character = world_state.get('characters', character_id)
        if not character:
            return jsonify({'success': False, 'error': 'Character not found'}), 404

        # Handle tags update
        if character_tags:
            tags = [tag.strip() for tag in character_tags.split(',') if tag.strip()]
            character['tags'] = tags
            character['updated_at'] = datetime.now().isoformat()
            world_state.add_or_update('characters', character_id, character)

        # Handle AI editing if requested
        if edit_request:
            # Get novel context
            novel = world_state.get('novels', character.get('novel_id'))
            novel_context = {
                'title': novel.get('title', '') if novel else '',
                'genre': novel.get('genre', '') if novel else '',
                'description': novel.get('description', '') if novel else ''
            }

            # Use AI to edit the character
            updated_character = character_editor.edit_character(
                current_character=character,
                edit_request=edit_request,
                novel_context=novel_context
            )

            # Preserve manually edited tags if they were updated
            if character_tags:
                updated_character['tags'] = character['tags']

            # Check if AI editing actually succeeded
            if updated_character.get('ai_edit_failed'):
                error_msg = updated_character.get('ai_edit_error', 'Unknown AI error')
                return jsonify({
                    'success': False,
                    'error': f'AI editing failed: {error_msg}',
                    'ai_unavailable': True
                }), 503

            # Validate updated character data
            if not updated_character or not isinstance(updated_character, dict):
                logger.error(f"Invalid character data returned from AI: {type(updated_character)}")
                return jsonify({
                    'success': False,
                    'error': 'AI returned invalid character data'
                }), 500

            # Add metadata for successful AI edit
            updated_character['updated_at'] = datetime.now().isoformat()
            updated_character['ai_edited'] = True

            # Save the updated character
            success = world_state.add_or_update('characters', character_id, updated_character)

            if success:
                return jsonify({
                    'success': True,
                    'message': f'Character "{character.get("name", "Unknown")}" updated successfully using AI!',
                    'character': updated_character
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to save character changes'}), 500

        elif character_tags:
            # Tags-only update was successful
            return jsonify({
                'success': True,
                'message': 'Character tags updated successfully!',
                'character': character
            })
        else:
            return jsonify({'success': False, 'error': 'No changes provided'}), 400

    except Exception as e:
        logger.error(f"Character save error: {e}")
        return jsonify({'success': False, 'error': 'Failed to save character'}), 500

@app.route('/api/locations/<location_id>/save', methods=['POST'])
def api_save_location(location_id):
    """API endpoint for saving location changes via AJAX."""
    try:
        # Get form data
        edit_request = request.form.get('edit_request', '').strip()
        location_tags = request.form.get('location_tags', '').strip()

        # Get current location
        location = world_state.get('locations', location_id)
        if not location:
            return jsonify({'success': False, 'error': 'Location not found'}), 404

        # Handle tags update
        if location_tags:
            tags = [tag.strip() for tag in location_tags.split(',') if tag.strip()]
            location['tags'] = tags
            location['updated_at'] = datetime.now().isoformat()
            world_state.add_or_update('locations', location_id, location)

        # Handle AI editing if requested
        if edit_request:
            # Get novel context
            novel = world_state.get('novels', location.get('novel_id'))
            novel_context = {
                'title': novel.get('title', '') if novel else '',
                'genre': novel.get('genre', '') if novel else '',
                'description': novel.get('description', '') if novel else ''
            }

            # Use AI to edit the location
            updated_location = location_editor.edit_location(
                current_location=location,
                edit_request=edit_request,
                novel_context=novel_context
            )

            # Preserve manually edited tags if they were updated
            if location_tags:
                updated_location['tags'] = location['tags']

            # Check if AI editing actually succeeded
            if updated_location.get('ai_edit_failed'):
                error_msg = updated_location.get('ai_edit_error', 'Unknown AI error')
                return jsonify({
                    'success': False,
                    'error': f'AI editing failed: {error_msg}',
                    'ai_unavailable': True
                }), 503

            # Preserve important metadata for successful AI edit
            updated_location['novel_id'] = location.get('novel_id')
            updated_location['id'] = location_id
            if 'created_at' in location:
                updated_location['created_at'] = location['created_at']
            updated_location['updated_at'] = datetime.now().isoformat()
            updated_location['ai_edited'] = True

            # Save the updated location
            success = world_state.add_or_update('locations', location_id, updated_location)

            if success:
                return jsonify({
                    'success': True,
                    'message': f'Location "{location.get("name")}" updated successfully using AI!',
                    'location': updated_location
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to save location changes'}), 500

        elif location_tags:
            # Tags-only update was successful
            return jsonify({
                'success': True,
                'message': 'Location tags updated successfully!',
                'location': location
            })
        else:
            return jsonify({'success': False, 'error': 'No changes provided'}), 400

    except Exception as e:
        logger.error(f"Location save error: {e}")
        return jsonify({'success': False, 'error': 'Failed to save location'}), 500

@app.route('/api/lore/<lore_id>/save', methods=['POST'])
def api_save_lore(lore_id):
    """API endpoint for saving lore changes via AJAX."""
    try:
        # Get form data
        edit_request = request.form.get('edit_request', '').strip()
        lore_tags = request.form.get('lore_tags', '').strip()

        # Get current lore entry
        lore_entry = world_state.get('lore', lore_id)
        if not lore_entry:
            return jsonify({'success': False, 'error': 'Lore entry not found'}), 404

        # Handle tags update
        if lore_tags:
            tags = [tag.strip() for tag in lore_tags.split(',') if tag.strip()]
            lore_entry['tags'] = tags
            lore_entry['updated_at'] = datetime.now().isoformat()
            world_state.add_or_update('lore', lore_id, lore_entry)

        # Handle AI editing if requested
        if edit_request:
            # Get novel context
            novel = world_state.get('novels', lore_entry.get('novel_id'))
            novel_context = {
                'title': novel.get('title', '') if novel else '',
                'genre': novel.get('genre', '') if novel else '',
                'description': novel.get('description', '') if novel else ''
            }

            # Use AI to edit the lore entry
            updated_lore = lore_editor.edit_lore(
                current_lore=lore_entry,
                edit_request=edit_request,
                novel_context=novel_context
            )

            # Preserve manually edited tags if they were updated
            if lore_tags:
                updated_lore['tags'] = lore_entry['tags']

            # Check if AI editing actually succeeded
            if updated_lore.get('ai_edit_failed'):
                error_msg = updated_lore.get('ai_edit_error', 'Unknown AI error')
                return jsonify({
                    'success': False,
                    'error': f'AI editing failed: {error_msg}',
                    'ai_unavailable': True
                }), 503

            # Add metadata for successful AI edit
            updated_lore['updated_at'] = datetime.now().isoformat()
            updated_lore['ai_edited'] = True

            # Save the updated lore entry
            success = world_state.add_or_update('lore', lore_id, updated_lore)

            if success:
                return jsonify({
                    'success': True,
                    'message': f'Lore entry "{lore_entry.get("title")}" updated successfully using AI!',
                    'lore': updated_lore
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to save lore changes'}), 500

        elif lore_tags:
            # Tags-only update was successful
            return jsonify({
                'success': True,
                'message': 'Lore entry tags updated successfully!',
                'lore': lore_entry
            })
        else:
            return jsonify({'success': False, 'error': 'No changes provided'}), 400

    except Exception as e:
        logger.error(f"Lore save error: {e}")
        return jsonify({'success': False, 'error': 'Failed to save lore entry'}), 500

# Fast Mode Endpoints for Speed Optimization
@app.route('/novel/<novel_id>/worldbuilding/characters/<character_id>/edit/fast', methods=['POST'])
def fast_edit_character(novel_id, character_id):
    """Fast character editing endpoint with optimized settings."""
    try:
        data = request.get_json()
        edit_request = data.get('edit_request', '').strip()

        if not edit_request:
            return jsonify({'success': False, 'error': 'Edit request is required'}), 400

        # Get current character
        character = world_state.get('characters', character_id)
        if not character:
            return jsonify({'success': False, 'error': 'Character not found'}), 404

        # Get novel context
        novel = world_state.get('novels', novel_id)
        novel_context = {
            'title': novel.get('title', '') if novel else '',
            'genre': novel.get('genre', '') if novel else ''
        }

        # Use AI to edit with fast settings
        updated_character = character_editor.edit_character(
            current_character=character,
            edit_request=edit_request,
            novel_context=novel_context
        )

        # Check for AI failure
        if updated_character.get('ai_edit_failed'):
            error_msg = updated_character.get('ai_edit_error', 'Unknown AI error')
            return jsonify({
                'success': False,
                'error': f'AI editing failed: {error_msg}',
                'ai_unavailable': True
            }), 503

        # Add metadata
        updated_character['updated_at'] = datetime.now().isoformat()
        updated_character['ai_edited'] = True

        # Save the updated character
        success = world_state.add_or_update('characters', character_id, updated_character)

        if success:
            return jsonify({
                'success': True,
                'message': f'Character "{character.get("name")}" updated successfully using Fast Mode!',
                'character': updated_character
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save character changes'}), 500

    except Exception as e:
        logger.error(f"Fast character edit error: {e}")
        return jsonify({'success': False, 'error': 'Failed to edit character'}), 500



if __name__ == '__main__':
    app.run(debug=True)
