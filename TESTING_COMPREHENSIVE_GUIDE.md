# Comprehensive Testing Guide for Lazywriter

## 🎯 Overview

This guide covers the complete testing infrastructure for the Lazywriter worldbuilding system, including unit tests, integration tests, performance benchmarks, and testing best practices.

## 📁 Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and test utilities
├── test_agents.py              # AI agent unit tests
├── test_database.py            # Database operation tests
├── test_utils.py               # Utility module tests
├── test_integration.py         # End-to-end integration tests
├── test_performance.py         # Performance and benchmarking tests
└── test_runner.py              # Comprehensive test runner
```

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
pip install pytest pytest-mock pytest-asyncio pytest-cov psutil
```

### 2. Run Tests

```bash
# Quick smoke test
python tests/test_runner.py smoke

# Run all unit tests
python tests/test_runner.py unit

# Run with coverage
python tests/test_runner.py all --coverage

# Run specific test
python tests/test_runner.py --test test_agents.py::TestCharacterCreatorAgent
```

## 🧪 Test Categories

### Unit Tests (`test_agents.py`, `test_database.py`, `test_utils.py`)

**Coverage:**
- ✅ All AI agents (Character, Lore, Location creators/editors)
- ✅ Cross-reference agent and multi-agent coordinator
- ✅ Database operations (WorldState, TinyDB, ChromaDB)
- ✅ Semantic search functionality
- ✅ Entity detection and recognition
- ✅ Utility functions (caching, streaming, rate limiting)

**Key Features:**
- Mocked external dependencies (OpenRouter API, Stanza NLP)
- Comprehensive error handling tests
- Data validation and schema tests
- Performance-aware unit tests

### Integration Tests (`test_integration.py`)

**Coverage:**
- ✅ Complete worldbuilding workflows
- ✅ Character creation → editing → cross-reference pipeline
- ✅ Multi-entity relationship analysis
- ✅ Database synchronization (TinyDB ↔ ChromaDB)
- ✅ End-to-end cross-reference analysis

**Key Features:**
- Real workflow simulation
- Multi-component interaction testing
- Data consistency verification
- Concurrent operation testing

### Performance Tests (`test_performance.py`)

**Coverage:**
- ✅ Database operation speed (bulk insert/query)
- ✅ AI agent response times and caching
- ✅ Entity recognition processing speed
- ✅ Memory usage optimization
- ✅ Concurrent access performance
- ✅ Error recovery speed
- ✅ Stress testing under load

**Key Features:**
- Throughput measurements
- Memory usage monitoring
- Concurrent stress testing
- Performance regression detection

## 🔧 Test Configuration

### Pytest Configuration (`pytest.ini`)

```ini
[tool:pytest]
testpaths = tests
addopts = 
    --verbose
    --cov=agents
    --cov=database
    --cov=utils
    --cov-report=html:htmlcov
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow running tests
    requires_api: Tests that require API keys
    requires_models: Tests that require ML models
```

### Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.requires_api` - Tests needing API keys
- `@pytest.mark.requires_models` - Tests needing ML models

## 🛠️ Test Utilities and Fixtures

### Shared Fixtures (`conftest.py`)

```python
# Database fixtures
@pytest.fixture
def mock_world_state(temp_data_dir):
    """Mock WorldState for testing"""

@pytest.fixture  
def mock_semantic_search(mock_world_state):
    """Mock SemanticSearchEngine for testing"""

# Agent fixtures
@pytest.fixture
def character_creator_agent(mock_openrouter_client):
    """CharacterCreatorAgent with mocked API"""

# Data fixtures
@pytest.fixture
def sample_character_data():
    """Sample character data for testing"""
```

### Test Utilities

```python
class TestUtils:
    @staticmethod
    def assert_valid_entity_structure(entity, entity_type):
        """Validate entity structure"""
    
    @staticmethod
    def assert_valid_uuid(uuid_string):
        """Validate UUID format"""
    
    @staticmethod
    def mock_ai_response(response_data):
        """Generate mock AI response"""
```

## 📊 Running Tests

### Test Runner Modes

```bash
# Environment check
python tests/test_runner.py check

# Quick tests (no slow/API tests)
python tests/test_runner.py quick

# Unit tests only
python tests/test_runner.py unit --verbose

# Integration tests only
python tests/test_runner.py integration

# Performance benchmarks
python tests/test_runner.py performance

# All tests with coverage
python tests/test_runner.py all --coverage

# Generate comprehensive report
python tests/test_runner.py report

# Smoke tests (basic functionality)
python tests/test_runner.py smoke
```

### Direct Pytest Usage

```bash
# Run specific test file
pytest tests/test_agents.py -v

# Run specific test class
pytest tests/test_agents.py::TestCharacterCreatorAgent -v

# Run specific test method
pytest tests/test_agents.py::TestCharacterCreatorAgent::test_create_character_success -v

# Run tests with markers
pytest -m "unit and not slow" -v

# Run with coverage
pytest --cov=agents --cov=database --cov=utils --cov-report=html
```

## 🎯 Test Coverage Goals

### Current Coverage Targets

- **Agents**: 90%+ coverage
- **Database**: 85%+ coverage  
- **Utils**: 80%+ coverage
- **Overall**: 80%+ coverage

### Coverage Reports

```bash
# Generate HTML coverage report
python tests/test_runner.py all --coverage

# View coverage report
open htmlcov/index.html
```

## 🚨 Error Handling Tests

### Network Error Simulation

```python
def test_network_timeout_handling(self, character_creator_agent):
    """Test handling of network timeouts."""
    character_creator_agent.openrouter_client.chat.completions.create.side_effect = requests.Timeout("Timeout")
    
    result = character_creator_agent.create_character("Test", "Test prompt")
    assert "fallback" in result["description"].lower()
```

### API Rate Limit Testing

```python
def test_rate_limit_handling(self, character_creator_agent):
    """Test handling of API rate limits."""
    from openai import RateLimitError
    character_creator_agent.openrouter_client.chat.completions.create.side_effect = RateLimitError(
        "Rate limit exceeded", response=Mock(), body=None
    )
    
    result = character_creator_agent.create_character("Test", "Test prompt")
    assert result["name"] == "Test"
```

## ⚡ Performance Benchmarking

### Database Performance

```python
def test_bulk_insert_performance(self, mock_world_state):
    """Test performance of bulk entity insertions."""
    entity_count = 100
    start_time = time.time()
    
    for i in range(entity_count):
        # Insert entity
        pass
    
    total_time = time.time() - start_time
    throughput = entity_count / total_time
    
    assert throughput > 10  # At least 10 entities per second
```

### Memory Usage Testing

```python
def test_memory_efficiency(self):
    """Test memory usage under load."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operations
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
```

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-mock
    - name: Run tests
      run: python tests/test_runner.py all --coverage
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 🐛 Debugging Tests

### Verbose Output

```bash
# Run with maximum verbosity
pytest tests/test_agents.py -vvv -s

# Show local variables on failure
pytest tests/test_agents.py --tb=long -vvv

# Drop into debugger on failure
pytest tests/test_agents.py --pdb
```

### Test Isolation

```bash
# Run single test in isolation
pytest tests/test_agents.py::TestCharacterCreatorAgent::test_create_character_success -v

# Run with fresh imports
pytest tests/test_agents.py --forked
```

## 📈 Performance Monitoring

### Benchmark Results Format

```
Database Performance:
- Bulk insert: 45.2 entities/second
- Individual queries: 123.4 queries/second
- Memory usage: 12.3 MB for 1000 entities

Agent Performance:
- Character creation: 2.1 characters/second
- Edit with cache hit: 15.2x speedup
- Error recovery: 0.23 seconds average

Cache Performance:
- Set operations: 5,432 ops/second
- Get operations: 12,845 ops/second
- Memory efficiency: 98.2% within limits
```

## ✅ Best Practices

### Writing Tests

1. **Use descriptive test names**: `test_character_creation_with_invalid_json`
2. **Test one thing per test**: Focus on single functionality
3. **Use fixtures for setup**: Avoid repetitive setup code
4. **Mock external dependencies**: Don't rely on external APIs
5. **Test error conditions**: Include failure scenarios
6. **Verify side effects**: Check database changes, cache updates

### Test Organization

1. **Group related tests**: Use test classes for organization
2. **Use appropriate markers**: Mark slow, integration, API tests
3. **Maintain test data**: Keep sample data consistent
4. **Document complex tests**: Add docstrings for complex scenarios

### Performance Testing

1. **Set realistic thresholds**: Based on actual requirements
2. **Test under load**: Include concurrent access tests
3. **Monitor memory usage**: Prevent memory leaks
4. **Test error recovery**: Measure resilience performance

## 🔍 Troubleshooting

### Common Issues

1. **Import errors**: Check PYTHONPATH and project structure
2. **Fixture not found**: Verify fixture scope and location
3. **Mock not working**: Check mock target path
4. **Slow tests**: Use markers to exclude from quick runs
5. **Memory issues**: Check for proper cleanup in fixtures

### Debug Commands

```bash
# Check test discovery
pytest --collect-only

# Run with profiling
pytest --profile

# Check fixture usage
pytest --fixtures

# Validate test markers
pytest --markers
```

---

## 📞 Support

For testing issues or questions:
1. Check this guide first
2. Review test output and error messages
3. Use verbose mode for detailed information
4. Check fixture and mock configurations
5. Verify test environment setup

**Happy Testing! 🧪✨**
