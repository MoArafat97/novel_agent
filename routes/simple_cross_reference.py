"""
Simplified Cross-Reference Routes

Handles new entity detection and creation workflow.
"""

from flask import Blueprint, request, jsonify, render_template
import logging
from database.world_state import WorldState
from utils.simple_cross_reference_service import SimpleCrossReferenceService

logger = logging.getLogger(__name__)

# Create blueprint
simple_cross_reference_bp = Blueprint('simple_cross_reference', __name__)

# Initialize services
world_state = WorldState()
cross_reference_service = SimpleCrossReferenceService(world_state)

@simple_cross_reference_bp.route('/novel/<novel_id>/cross-reference/analyze', methods=['POST'])
def analyze_for_new_entities(novel_id):
    """Analyze entity content for new entities that should be added to the database."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['entity_type', 'entity_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        entity_type = data['entity_type']
        entity_id = data['entity_id']
        
        # Get the entity data from the database
        entity_data = world_state.get(entity_type, entity_id)
        if not entity_data:
            return jsonify({'success': False, 'error': 'Entity not found'}), 404
        
        # Analyze for new entities
        result = cross_reference_service.analyze_content_for_new_entities(
            entity_type, entity_id, entity_data, novel_id
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Cross-reference analysis failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@simple_cross_reference_bp.route('/novel/<novel_id>/cross-reference/create-entities', methods=['POST'])
def create_selected_entities(novel_id):
    """Create the entities that the user selected from detection results."""
    try:
        data = request.get_json()
        
        if 'selected_entities' not in data:
            return jsonify({'success': False, 'error': 'Missing selected_entities'}), 400
        
        selected_entities = data['selected_entities']
        
        if not isinstance(selected_entities, list):
            return jsonify({'success': False, 'error': 'selected_entities must be a list'}), 400
        
        # Create the selected entities
        result = cross_reference_service.create_selected_entities(selected_entities, novel_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Entity creation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@simple_cross_reference_bp.route('/novel/<novel_id>/cross-reference/detection-ui')
def show_detection_ui(novel_id):
    """Show the new entity detection UI."""
    try:
        # Get detection results from query parameters (if any)
        entity_type = request.args.get('entity_type')
        entity_id = request.args.get('entity_id')
        
        context = {
            'novel_id': novel_id,
            'entity_type': entity_type,
            'entity_id': entity_id
        }
        
        return render_template('cross_reference/simple_detection.html', **context)
        
    except Exception as e:
        logger.error(f"Failed to show detection UI: {e}")
        return f"Error: {e}", 500

@simple_cross_reference_bp.route('/novel/<novel_id>/cross-reference/test')
def test_detection(novel_id):
    """Test endpoint for debugging the detection system."""
    try:
        # Get a test character for demonstration
        novel_entities = world_state.get_entities_by_novel(novel_id)
        characters = novel_entities.get('characters', [])

        if not characters:
            return jsonify({'error': 'No characters found for testing'}), 404

        test_character = characters[0]

        # Analyze the test character
        result = cross_reference_service.analyze_content_for_new_entities(
            'characters', test_character['id'], test_character, novel_id
        )

        return jsonify({
            'test_character': {
                'name': test_character.get('name', 'Unknown'),
                'id': test_character['id']
            },
            'detection_result': result
        })

    except Exception as e:
        logger.error(f"Test detection failed: {e}")
        return jsonify({'error': str(e)}), 500

@simple_cross_reference_bp.route('/novel/<novel_id>/cross-reference/status')
def cross_reference_status(novel_id):
    """Get cross-reference agent status."""
    try:
        # Verify novel exists
        novel = world_state.get('novels', novel_id)
        if not novel:
            return jsonify({'success': False, 'error': 'Novel not found'}), 404

        # For the simple cross-reference system, we just check if the service is available
        return jsonify({
            'success': True,
            'available': True,  # Simple cross-reference is always available
            'agent_info': {
                'has_world_state': world_state is not None,
                'has_service': cross_reference_service is not None,
                'type': 'simple_cross_reference'
            }
        })

    except Exception as e:
        logger.error(f"Cross-reference status error: {e}")
        return jsonify({'success': False, 'error': 'Failed to get status'}), 500
