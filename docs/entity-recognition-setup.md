# OptimizedEntityRecognizer Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Install required packages
pip install stanza scikit-learn numpy pandas

# Download Stanza English model
python -c "import stanza; stanza.download('en')"
```

### 2. Verify Installation

```python
# Test basic functionality
from utils.stanza_entity_recognizer import OptimizedEntityRecognizer

recognizer = OptimizedEntityRecognizer()
print("✅ OptimizedEntityRecognizer installed successfully!")
```

### 3. Run Performance Test

```bash
# Run comprehensive benchmark
python test_stanza_entity_recognizer.py

# Expected output:
# ✅ All tests completed successfully!
```

## Configuration Options

### Basic Configuration

```python
recognizer = OptimizedEntityRecognizer(
    world_state=world_state,    # Required for gazetteer functionality
    cache_size=1000            # Number of cached results (default: 1000)
)
```

### Advanced Configuration

```python
# Custom confidence weights
recognizer.confidence_weights = {
    'exact': 1.0,              # Perfect matches
    'gazetteer': 0.9,          # Known entity matches
    'stanza_ner': 0.7,         # Stanza NER detections
    'contextual': 0.6,         # Context-based classification
    'partial': 0.8             # Partial name matches
}

# Custom context indicators
recognizer.context_indicators['characters'].update({
    'protagonist', 'villain', 'hero', 'champion'
})

# Custom entity type mappings
recognizer.stanza_entity_mappings['WORK_OF_ART'] = 'lore'
```

## Integration with CrossReferenceAgent

The integration is automatic when both systems are available:

```python
# In agents/cross_reference_agent.py
class CrossReferenceAgent:
    def __init__(self, world_state=None, semantic_search=None):
        # ... existing code ...
        
        # OptimizedEntityRecognizer is automatically initialized
        try:
            self.optimized_recognizer = OptimizedEntityRecognizer(
                world_state=world_state,
                cache_size=1000
            )
            logger.info("OptimizedEntityRecognizer initialized")
        except Exception as e:
            logger.warning(f"Falling back to basic detection: {e}")
            self.optimized_recognizer = None
```

## Performance Tuning

### For High-Volume Applications

```python
# Optimize for throughput
recognizer = OptimizedEntityRecognizer(
    cache_size=2000,           # Larger cache
    world_state=world_state
)

# Use higher confidence threshold for speed
matches = recognizer.recognize_entities(
    content=content,
    novel_id=novel_id,
    confidence_threshold=0.7   # Higher threshold = faster
)
```

### For High-Accuracy Applications

```python
# Optimize for accuracy
recognizer = OptimizedEntityRecognizer(
    cache_size=1000,
    world_state=world_state
)

# Use lower confidence threshold for completeness
matches = recognizer.recognize_entities(
    content=content,
    novel_id=novel_id,
    confidence_threshold=0.3   # Lower threshold = more results
)
```

### Memory Optimization

```python
# For memory-constrained environments
recognizer = OptimizedEntityRecognizer(
    cache_size=500,            # Smaller cache
    world_state=world_state
)

# Periodic cleanup
if len(recognizer.cache) > 400:
    recognizer.clear_cache()
```

## Monitoring and Maintenance

### Performance Monitoring

```python
# Check cache effectiveness
cache_stats = recognizer.get_cache_stats()
print(f"Cache hit rate: {cache_stats['valid_entries'] / cache_stats['total_entries']:.1%}")

# Monitor relationship learning
rel_stats = recognizer.get_relationship_cache_stats()
print(f"Relationship patterns learned: {rel_stats['total_patterns']}")

# Analyze confidence distribution
matches = recognizer.recognize_entities(content, novel_id)
confidence_dist = recognizer.get_confidence_distribution(matches)
print(f"High confidence entities: {confidence_dist['very_high']}")
```

### Regular Maintenance

```python
# Weekly optimization
def weekly_maintenance():
    recognizer.optimize_for_performance()
    cache_stats = recognizer.get_cache_stats()
    
    if cache_stats['cache_utilization'] > 0.9:
        # Increase cache size if needed
        recognizer.cache_size = min(2000, recognizer.cache_size * 1.2)
    
    logger.info(f"Maintenance complete. Cache utilization: {cache_stats['cache_utilization']:.1%}")

# Monthly cleanup
def monthly_cleanup():
    recognizer.clear_cache()
    recognizer.clear_relationship_cache()
    logger.info("Monthly cache cleanup completed")
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Stanza model not found"
```bash
# Solution: Download the English model
python -c "import stanza; stanza.download('en')"
```

#### Issue: High memory usage
```python
# Solution: Reduce cache size and clean periodically
recognizer = OptimizedEntityRecognizer(cache_size=500)
recognizer.clear_cache()  # Clean when needed
```

#### Issue: Slow performance
```python
# Solution: Check cache utilization and optimize
cache_stats = recognizer.get_cache_stats()
if cache_stats['cache_utilization'] < 0.5:
    # Increase cache size
    recognizer.cache_size = 1500

recognizer.optimize_for_performance()
```

#### Issue: Low accuracy on custom entities
```python
# Solution: Update gazetteers and context indicators
recognizer.update_gazetteer(novel_id, 'characters', your_characters)
recognizer.context_indicators['characters'].update(your_context_words)
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed processing information
matches = recognizer.recognize_entities(content, novel_id)
```

## Testing

### Unit Tests

```bash
# Run basic functionality tests
python test_stanza_entity_recognizer.py
```

### Integration Tests

```bash
# Run integration tests with CrossReferenceAgent
python test_cross_reference_integration.py
```

### Performance Benchmarks

```bash
# Run comprehensive performance benchmark
python performance_benchmark.py

# Expected results:
# - Accuracy: 75%+ F1 score
# - Speed: <100ms response time
# - Cache: >1000x speedup for cached content
```

## Production Deployment

### Recommended Settings

```python
# Production configuration
recognizer = OptimizedEntityRecognizer(
    world_state=world_state,
    cache_size=1500            # Balanced cache size
)

# Production thresholds
CONFIDENCE_THRESHOLD = 0.5     # Balanced accuracy/precision
LLM_THRESHOLD = 0.7           # Conservative LLM verification
```

### Health Checks

```python
def health_check():
    """Health check for production monitoring."""
    try:
        # Test basic functionality
        test_content = "Test entity recognition with Aragorn."
        matches = recognizer.recognize_entities(test_content, "test-novel")
        
        # Check cache performance
        cache_stats = recognizer.get_cache_stats()
        
        return {
            'status': 'healthy',
            'entities_detected': len(matches),
            'cache_utilization': cache_stats['cache_utilization'],
            'cache_entries': cache_stats['total_entries']
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }
```

### Monitoring Alerts

Set up alerts for:
- Cache utilization > 95%
- Response time > 200ms
- Error rate > 1%
- Memory usage > 80%

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Run the diagnostic tests
3. Review the performance benchmark results
4. Check system logs for detailed error information

The OptimizedEntityRecognizer is designed to be robust and self-monitoring, with comprehensive logging and performance metrics to help identify and resolve issues quickly.
