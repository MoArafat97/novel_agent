#!/bin/bash

# Cross-Reference Feature Setup and Testing Script
echo "=== Setting up Cross-Reference Feature Testing Environment ==="

# Update system packages
sudo apt-get update -y

# Install Python 3 and pip if not available
sudo apt-get install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Add virtual environment activation to profile
echo "source /mnt/persist/workspace/venv/bin/activate" >> $HOME/.profile

# Install Python dependencies
pip install --upgrade pip

# Install requirements from requirements.txt
pip install -r requirements.txt

# Install additional dependencies for Stanza (mentioned in docs)
pip install stanza scikit-learn numpy pandas

# Download Stanza English model
python3 -c "import stanza; stanza.download('en')"

# Create necessary directories
mkdir -p data/tinydb
mkdir -p data/chromadb
mkdir -p logs

# Set up environment variables for testing
cat > .env << 'EOF'
# Test environment configuration
OPENROUTER_API_KEY=test_key_placeholder
CHAT_MODEL=deepseek/deepseek-chat:free
EMBEDDING_MODEL=text-embedding-3-small
FLASK_ENV=development
FLASK_DEBUG=True
EOF

# Create a comprehensive test script for cross-reference functionality
cat > test_cross_reference_comprehensive.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive Cross-Reference Feature Test Suite

This script tests all components of the cross-reference system to identify
and report any issues that need to be fixed.
"""

import sys
import os
import traceback
import json
import uuid
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all cross-reference components can be imported."""
    print("\n=== Testing Imports ===")
    
    try:
        # Core components
        from agents.cross_reference_agent import CrossReferenceAgent
        print("✅ CrossReferenceAgent imported successfully")
        
        from utils.entity_detection import EntityDetectionUtils
        print("✅ EntityDetectionUtils imported successfully")
        
        from utils.stanza_entity_recognizer import OptimizedEntityRecognizer
        print("✅ OptimizedEntityRecognizer imported successfully")
        
        from agents.relationship_detection_agent import RelationshipDetectionAgent
        print("✅ RelationshipDetectionAgent imported successfully")
        
        from utils.cross_reference_cache import CrossReferenceCacheManager
        print("✅ CrossReferenceCacheManager imported successfully")
        
        # Database components
        from database import WorldState
        from database.semantic_search import SemanticSearchEngine
        print("✅ Database components imported successfully")
        
        # Flask app
        from app import app
        print("✅ Flask app imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_database_initialization():
    """Test database initialization."""
    print("\n=== Testing Database Initialization ===")
    
    try:
        from database import WorldState
        from database.semantic_search import SemanticSearchEngine
        
        # Initialize WorldState
        world_state = WorldState()
        print("✅ WorldState initialized successfully")
        
        # Initialize SemanticSearchEngine
        semantic_search = SemanticSearchEngine()
        print("✅ SemanticSearchEngine initialized successfully")
        
        return world_state, semantic_search
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        traceback.print_exc()
        return None, None

def test_cross_reference_agent_initialization():
    """Test CrossReferenceAgent initialization."""
    print("\n=== Testing CrossReferenceAgent Initialization ===")
    
    try:
        from agents.cross_reference_agent import CrossReferenceAgent
        from database import WorldState
        from database.semantic_search import SemanticSearchEngine
        
        world_state = WorldState()
        semantic_search = SemanticSearchEngine()
        
        # Initialize CrossReferenceAgent
        agent = CrossReferenceAgent(world_state=world_state, semantic_search=semantic_search)
        print("✅ CrossReferenceAgent initialized successfully")
        
        # Test availability check
        is_available = agent.is_available()
        print(f"✅ Agent availability check: {is_available}")
        
        return agent
        
    except Exception as e:
        print(f"❌ CrossReferenceAgent initialization failed: {e}")
        traceback.print_exc()
        return None

def test_entity_detection():
    """Test entity detection functionality."""
    print("\n=== Testing Entity Detection ===")
    
    try:
        from utils.entity_detection import EntityDetectionUtils
        
        detector = EntityDetectionUtils()
        
        # Test text
        test_text = "Aragorn walked through Rivendell with Gandalf the Grey. They discussed the Ring of Power."
        
        # Test existing entities (mock data)
        existing_entities = {
            'characters': [
                {'id': 'char1', 'name': 'Aragorn'},
                {'id': 'char2', 'name': 'Gandalf'}
            ],
            'locations': [
                {'id': 'loc1', 'name': 'Rivendell'}
            ],
            'lore': [
                {'id': 'lore1', 'name': 'Ring of Power'}
            ]
        }
        
        # Detect entities
        detected = detector.detect_potential_entities(test_text, existing_entities)
        print(f"✅ Entity detection completed. Found: {len(detected.get('characters', []))} characters, {len(detected.get('locations', []))} locations, {len(detected.get('lore', []))} lore")
        
        return True
        
    except Exception as e:
        print(f"❌ Entity detection failed: {e}")
        traceback.print_exc()
        return False

def test_optimized_entity_recognizer():
    """Test OptimizedEntityRecognizer functionality."""
    print("\n=== Testing OptimizedEntityRecognizer ===")
    
    try:
        from utils.stanza_entity_recognizer import OptimizedEntityRecognizer
        from database import WorldState
        
        world_state = WorldState()
        recognizer = OptimizedEntityRecognizer(world_state=world_state)
        print("✅ OptimizedEntityRecognizer initialized successfully")
        
        # Test entity recognition
        test_content = "Aragorn walked through Rivendell with Gandalf."
        novel_id = "test-novel"
        
        matches = recognizer.recognize_entities(
            content=test_content,
            novel_id=novel_id,
            confidence_threshold=0.3
        )
        
        print(f"✅ Entity recognition completed. Found {len(matches)} matches")
        for match in matches[:3]:  # Show first 3 matches
            print(f"   - {match.entity_name} ({match.entity_type}) - confidence: {match.confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ OptimizedEntityRecognizer failed: {e}")
        traceback.print_exc()
        return False

def test_flask_routes():
    """Test Flask routes for cross-reference functionality."""
    print("\n=== Testing Flask Routes ===")
    
    try:
        from app import app
        
        with app.test_client() as client:
            # Test cross-reference status endpoint
            response = client.get('/novel/test-novel/cross-reference/status')
            print(f"✅ Status endpoint responded with status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.get_json()
                print(f"   Available: {data.get('available', 'unknown')}")
            
            return True
            
    except Exception as e:
        print(f"❌ Flask routes test failed: {e}")
        traceback.print_exc()
        return False

def test_sample_analysis():
    """Test a complete cross-reference analysis workflow."""
    print("\n=== Testing Sample Analysis Workflow ===")
    
    try:
        from agents.cross_reference_agent import CrossReferenceAgent
        from database import WorldState
        from database.semantic_search import SemanticSearchEngine
        
        # Initialize components
        world_state = WorldState()
        semantic_search = SemanticSearchEngine()
        agent = CrossReferenceAgent(world_state=world_state, semantic_search=semantic_search)
        
        # Create test novel and entities
        novel_id = str(uuid.uuid4())
        character_id = str(uuid.uuid4())
        
        # Create test novel
        novel_data = {
            'id': novel_id,
            'title': 'Test Novel',
            'genre': 'Fantasy',
            'description': 'A test novel for cross-reference testing'
        }
        world_state.create('novels', novel_data)
        
        # Create test character
        character_data = {
            'id': character_id,
            'novel_id': novel_id,
            'name': 'Aragorn',
            'description': 'A ranger from the North',
            'backstory': 'Aragorn is the heir to the throne of Gondor. He met Gandalf in Rivendell.'
        }
        world_state.create('characters', character_data)
        
        print("✅ Test data created successfully")
        
        # Test analysis (basic version without LLM)
        if agent.is_available():
            print("⚠️  Agent is available but requires API key for full testing")
        else:
            print("ℹ️  Agent not fully available (expected without API keys)")
        
        # Test basic entity detection
        content_text = character_data['backstory']
        detected_entities = agent._detect_entity_mentions(content_text, novel_id)
        print(f"✅ Basic entity detection completed. Found {len(detected_entities)} entities")
        
        # Clean up test data
        world_state.delete('characters', character_id)
        world_state.delete('novels', novel_id)
        print("✅ Test data cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample analysis failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive cross-reference tests."""
    print("🔍 Cross-Reference Feature Comprehensive Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Imports", test_imports()))
    test_results.append(("Database Initialization", test_database_initialization()[0] is not None))
    test_results.append(("CrossReferenceAgent Initialization", test_cross_reference_agent_initialization() is not None))
    test_results.append(("Entity Detection", test_entity_detection()))
    test_results.append(("OptimizedEntityRecognizer", test_optimized_entity_recognizer()))
    test_results.append(("Flask Routes", test_flask_routes()))
    test_results.append(("Sample Analysis", test_sample_analysis()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Cross-reference feature appears to be working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Issues need to be addressed.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
EOF

chmod +x test_cross_reference_comprehensive.py

echo "✅ Cross-reference testing environment setup complete!"
echo "📁 Created comprehensive test script: test_cross_reference_comprehensive.py"
echo "🔧 Virtual environment activated and dependencies installed"
echo "📦 Stanza English model downloaded"
echo "⚙️  Environment variables configured"