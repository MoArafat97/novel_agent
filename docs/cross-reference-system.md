# Cross-Reference System Documentation

## Overview

The Cross-Reference System is a powerful feature in Lazywriter that automatically analyzes newly created or edited content (characters, locations, lore) to identify relationships with existing entities. It helps maintain consistency and discover connections within your worldbuilding content.

## Features

### Core Functionality

- **Automatic Entity Detection**: Identifies mentions of existing entities in new content
- **Relationship Discovery**: Finds connections between entities using AI analysis
- **Update Suggestions**: Generates specific updates for related entities based on new information
- **New Entity Detection**: Identifies potential new entities that should be added to the database
- **User Approval Workflow**: Allows users to review and approve/reject suggested changes

### Technical Approach

The system uses a hybrid approach for optimal performance:

1. **Lightweight Text Analysis**: Initial detection using regex patterns and named entity recognition
2. **Semantic Search Integration**: Leverages existing ChromaDB infrastructure for finding related content
3. **LLM Verification**: Uses DeepSeek LLM for relationship verification and update generation
4. **User Control**: All changes require explicit user approval

## User Guide

### Accessing Cross-Reference Analysis

1. Navigate to any character, location, or lore detail page
2. Click the **Cross-Reference** button (blue button with network icon)
3. Wait for the analysis to complete (usually 10-30 seconds)
4. Review the results in the modal dialog

### Understanding Results

The analysis results are organized into several sections:

#### Discovered Relationships
- Shows connections between the current entity and existing entities
- Includes relationship type (knows, lives_in, works_at, etc.)
- Displays confidence level (high, medium, low)
- Provides evidence from the text

#### Suggested Updates
- Lists specific field updates for related entities
- Shows exactly what would be changed
- Allows selective approval of individual updates
- Includes confidence ratings

#### Potential New Entities
- Identifies characters, locations, or lore concepts mentioned but not in database
- Provides context and evidence for each detection
- Suggests entity type and description

### Applying Updates

1. Review the suggested updates carefully
2. Check the boxes next to updates you want to apply
3. Click **Apply Selected Updates**
4. The page will refresh to show the applied changes

## Technical Implementation

### Architecture

```
CrossReferenceAgent
├── EntityDetectionUtils (lightweight analysis)
├── SemanticSearchEngine (finding related content)
├── DeepSeek LLM (verification and generation)
└── WorldState (data persistence)
```

### Key Components

#### CrossReferenceAgent
- Main orchestrator class
- Coordinates all analysis steps
- Handles LLM interactions
- Manages update application

#### EntityDetectionUtils
- Lightweight text analysis
- Regex pattern matching
- Named entity recognition
- Context analysis

#### API Endpoints

##### POST `/novel/<novel_id>/cross-reference/analyze`
Triggers cross-reference analysis for an entity.

**Request Body:**
```json
{
  "entity_type": "characters|locations|lore",
  "entity_id": "entity-uuid"
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "entity_type": "characters",
    "entity_id": "uuid",
    "entity_name": "Entity Name",
    "detected_entities": [...],
    "semantic_matches": [...],
    "verified_relationships": [...],
    "suggested_updates": [...],
    "new_entities": [...],
    "analysis_timestamp": "2024-01-01T12:00:00"
  }
}
```

##### POST `/novel/<novel_id>/cross-reference/apply`
Applies approved updates to entities.

**Request Body:**
```json
{
  "updates": [
    {
      "target_entity_id": "uuid",
      "target_entity_type": "characters",
      "suggested_changes": {
        "field_name": "new_value"
      }
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "applied_updates": [...],
  "failed_updates": [...],
  "total_applied": 5,
  "total_failed": 0
}
```

##### GET `/novel/<novel_id>/cross-reference/status`
Checks if cross-reference analysis is available.

**Response:**
```json
{
  "success": true,
  "available": true,
  "agent_info": {
    "has_openrouter": true,
    "has_world_state": true,
    "has_semantic_search": true
  }
}
```

### Configuration

The system requires the following environment variables:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key
CHAT_MODEL=deepseek/deepseek-chat:free
EMBEDDING_MODEL=text-embedding-3-small
```

### Performance Considerations

- **Lightweight First**: Uses regex and pattern matching before LLM calls
- **Caching**: Leverages existing semantic search infrastructure
- **Timeouts**: Implements proper timeouts for LLM requests
- **Error Handling**: Graceful degradation when services are unavailable

## Error Handling

### Common Issues

1. **Agent Not Available**
   - Missing OpenRouter API key
   - WorldState not initialized
   - Semantic search not available

2. **Analysis Failures**
   - Network timeouts
   - LLM service errors
   - Invalid entity data

3. **Update Failures**
   - Entity not found
   - Database write errors
   - Invalid update data

### Troubleshooting

1. Check environment variables are set correctly
2. Verify OpenRouter API key is valid
3. Ensure database connections are working
4. Check browser console for JavaScript errors

## Best Practices

### For Users

1. **Review Carefully**: Always review suggested updates before applying
2. **Start Small**: Test with a few updates first
3. **Check Results**: Verify applied changes are correct
4. **Use Selectively**: Not all suggestions need to be applied

### For Developers

1. **Error Handling**: Always handle LLM failures gracefully
2. **User Feedback**: Provide clear status indicators
3. **Performance**: Optimize for common use cases
4. **Testing**: Test with various content types and sizes

## Future Enhancements

- **Batch Processing**: Analyze multiple entities at once
- **Relationship Visualization**: Visual network of entity connections
- **Smart Suggestions**: Learn from user approval patterns
- **Integration**: Connect with writing tools and plot management
- **Advanced NLP**: Enhanced entity recognition and classification

## Support

For technical issues or questions:

1. Check the troubleshooting section above
2. Review the application logs
3. Test with the demo script: `python test_cross_reference_demo.py`
4. Verify all dependencies are installed and configured
