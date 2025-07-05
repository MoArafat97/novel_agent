"""
Performance and benchmarking tests for the Lazywriter system.

This module tests performance characteristics, memory usage, speed optimization,
and error handling under various conditions.
"""

import pytest
import time
import threading
import gc
import psutil
import os
from unittest.mock import Mock, patch
from database.world_state import WorldState
from agents.character_creator import CharacterCreatorAgent
from agents.cross_reference_agent import CrossReferenceAgent
from utils.stanza_entity_recognizer import OptimizedEntityRecognizer
from utils.cross_reference_cache import CrossReferenceCacheManager


@pytest.mark.performance
class TestDatabasePerformance:
    """Test database operation performance."""
    
    def test_bulk_insert_performance(self, mock_world_state):
        """Test performance of bulk entity insertions."""
        novel_id = "perf_test_novel"
        entity_count = 100
        
        start_time = time.time()
        
        # Insert entities
        for i in range(entity_count):
            entity_data = {
                'id': f'char_{i:04d}',
                'novel_id': novel_id,
                'name': f'Character {i}',
                'description': f'Performance test character {i}',
                'tags': [f'tag{i}', 'performance', 'test']
            }
            
            result = mock_world_state.add_or_update(
                'characters', entity_data['id'], entity_data, skip_embeddings=True
            )
            assert result is True
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert total_time < 10.0  # Should complete within 10 seconds
        
        # Calculate throughput
        throughput = entity_count / total_time
        assert throughput > 10  # At least 10 entities per second
        
        print(f"Bulk insert performance: {entity_count} entities in {total_time:.2f}s ({throughput:.1f} entities/s)")
    
    def test_bulk_query_performance(self, mock_world_state):
        """Test performance of bulk queries."""
        novel_id = "query_perf_test"
        entity_count = 50
        
        # First, create test data
        entity_ids = []
        for i in range(entity_count):
            entity_data = {
                'id': f'query_char_{i:04d}',
                'novel_id': novel_id,
                'name': f'Query Character {i}',
                'description': f'Query performance test character {i}'
            }
            
            mock_world_state.add_or_update(
                'characters', entity_data['id'], entity_data, skip_embeddings=True
            )
            entity_ids.append(entity_data['id'])
        
        # Test individual queries
        start_time = time.time()
        
        for entity_id in entity_ids:
            result = mock_world_state.get('characters', entity_id)
            assert result is not None
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Performance assertions
        assert query_time < 5.0  # Should complete within 5 seconds
        
        query_throughput = entity_count / query_time
        assert query_throughput > 20  # At least 20 queries per second
        
        print(f"Individual query performance: {entity_count} queries in {query_time:.2f}s ({query_throughput:.1f} queries/s)")
        
        # Test bulk query
        start_time = time.time()
        
        novel_entities = mock_world_state.get_entities_by_novel(novel_id)
        
        end_time = time.time()
        bulk_query_time = end_time - start_time
        
        assert bulk_query_time < 2.0  # Bulk query should be faster
        assert len(novel_entities['characters']) == entity_count
        
        print(f"Bulk query performance: {entity_count} entities in {bulk_query_time:.2f}s")
    
    def test_concurrent_database_access(self, mock_world_state):
        """Test database performance under concurrent access."""
        novel_id = "concurrent_test"
        thread_count = 5
        entities_per_thread = 10
        
        results = []
        errors = []
        
        def worker_thread(thread_id):
            try:
                thread_results = []
                for i in range(entities_per_thread):
                    entity_data = {
                        'id': f'concurrent_{thread_id}_{i:03d}',
                        'novel_id': novel_id,
                        'name': f'Concurrent Character T{thread_id}-{i}',
                        'description': f'Concurrent test character from thread {thread_id}'
                    }
                    
                    result = mock_world_state.add_or_update(
                        'characters', entity_data['id'], entity_data, skip_embeddings=True
                    )
                    thread_results.append(result)
                
                results.extend(thread_results)
            except Exception as e:
                errors.append(e)
        
        # Start concurrent operations
        start_time = time.time()
        
        threads = []
        for thread_id in range(thread_count):
            thread = threading.Thread(target=worker_thread, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        concurrent_time = end_time - start_time
        
        # Verify results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == thread_count * entities_per_thread
        assert all(result is True for result in results)
        
        # Performance assertion
        total_operations = thread_count * entities_per_thread
        concurrent_throughput = total_operations / concurrent_time
        
        print(f"Concurrent access performance: {total_operations} operations in {concurrent_time:.2f}s ({concurrent_throughput:.1f} ops/s)")
        
        # Verify data integrity
        novel_entities = mock_world_state.get_entities_by_novel(novel_id)
        assert len(novel_entities['characters']) == total_operations


@pytest.mark.performance
class TestAgentPerformance:
    """Test AI agent performance."""
    
    def test_character_creation_speed(self, character_creator_agent, test_utils):
        """Test character creation speed."""
        # Mock fast AI response
        ai_response = test_utils.mock_ai_response({
            "description": "A quick test character",
            "age": "25",
            "occupation": "Tester",
            "personality": "Fast and efficient",
            "backstory": "Created for speed testing",
            "tags": ["test", "speed"],
            "role": "supporting character"
        })
        
        character_creator_agent.openrouter_client.chat.completions.create.return_value.choices[0].message.content = ai_response
        
        # Test multiple character creations
        creation_count = 10
        start_time = time.time()
        
        for i in range(creation_count):
            result = character_creator_agent.create_character(
                name=f"Speed Test Character {i}",
                user_prompt="A character for speed testing"
            )
            assert result['name'] == f"Speed Test Character {i}"
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        # Performance assertions
        creation_throughput = creation_count / creation_time
        
        print(f"Character creation performance: {creation_count} characters in {creation_time:.2f}s ({creation_throughput:.1f} chars/s)")
        
        # Should be reasonably fast (accounting for mocked API calls)
        assert creation_time < 5.0
        assert creation_throughput > 2.0
    
    def test_character_editing_cache_performance(self, character_editor_agent, sample_character_data, test_utils):
        """Test character editing performance with caching."""
        ai_response = test_utils.mock_ai_response(sample_character_data)
        character_editor_agent.openrouter_client.chat.completions.create.return_value.choices[0].message.content = ai_response
        
        edit_request = "Make the character more experienced"
        
        # First edit (cache miss)
        start_time = time.time()
        result1 = character_editor_agent.edit_character(
            current_character=sample_character_data,
            edit_request=edit_request
        )
        first_edit_time = time.time() - start_time
        
        # Second edit (cache hit)
        start_time = time.time()
        result2 = character_editor_agent.edit_character(
            current_character=sample_character_data,
            edit_request=edit_request
        )
        second_edit_time = time.time() - start_time
        
        # Verify caching worked
        assert result1 == result2
        assert second_edit_time < first_edit_time  # Cache should be faster
        
        # API should only be called once due to caching
        assert character_editor_agent.openrouter_client.chat.completions.create.call_count == 1
        
        print(f"Edit performance - First: {first_edit_time:.3f}s, Cached: {second_edit_time:.3f}s (speedup: {first_edit_time/second_edit_time:.1f}x)")


@pytest.mark.performance
@pytest.mark.requires_models
class TestEntityRecognitionPerformance:
    """Test entity recognition performance."""
    
    def test_entity_recognition_speed(self, mock_world_state):
        """Test entity recognition processing speed."""
        with patch('utils.stanza_entity_recognizer.stanza') as mock_stanza:
            # Mock Stanza pipeline
            mock_doc = Mock()
            mock_doc.text = "Test text with entities"
            mock_doc.ents = []
            
            mock_pipeline = Mock()
            mock_pipeline.return_value = mock_doc
            mock_stanza.Pipeline.return_value = mock_pipeline
            
            recognizer = OptimizedEntityRecognizer(world_state=mock_world_state)
            
            # Test processing multiple texts
            test_texts = [
                "Aragorn walked through Rivendell with Gandalf the Grey.",
                "The Ring of Power was hidden in the Shire by Frodo Baggins.",
                "Legolas and Gimli fought at Helm's Deep during the great battle.",
                "Sauron's forces attacked Minas Tirith in the final war.",
                "Galadriel protected Lothlórien with her magical powers."
            ]
            
            start_time = time.time()
            
            for text in test_texts:
                results = recognizer.recognize_entities(
                    content=text,
                    novel_id="performance_test",
                    use_cache=False  # Disable cache for pure processing speed
                )
                assert isinstance(results, list)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Performance assertions
            text_count = len(test_texts)
            processing_throughput = text_count / processing_time
            
            print(f"Entity recognition performance: {text_count} texts in {processing_time:.2f}s ({processing_throughput:.1f} texts/s)")
            
            # Should process reasonably fast
            assert processing_time < 10.0
            assert processing_throughput > 0.5
    
    def test_entity_recognition_caching_performance(self, mock_world_state):
        """Test entity recognition caching effectiveness."""
        with patch('utils.stanza_entity_recognizer.stanza') as mock_stanza:
            mock_doc = Mock()
            mock_doc.text = "Cached test text"
            mock_doc.ents = []
            
            mock_pipeline = Mock()
            mock_pipeline.return_value = mock_doc
            mock_stanza.Pipeline.return_value = mock_pipeline
            
            recognizer = OptimizedEntityRecognizer(world_state=mock_world_state)
            
            test_text = "Aragorn is a ranger from Gondor who became king."
            novel_id = "cache_test"
            
            # First processing (cache miss)
            start_time = time.time()
            results1 = recognizer.recognize_entities(
                content=test_text,
                novel_id=novel_id,
                use_cache=True
            )
            first_time = time.time() - start_time
            
            # Second processing (cache hit)
            start_time = time.time()
            results2 = recognizer.recognize_entities(
                content=test_text,
                novel_id=novel_id,
                use_cache=True
            )
            second_time = time.time() - start_time
            
            # Verify caching effectiveness
            assert len(results1) == len(results2)
            assert second_time < first_time  # Cache should be significantly faster
            
            speedup = first_time / second_time if second_time > 0 else float('inf')
            print(f"Entity recognition caching - First: {first_time:.3f}s, Cached: {second_time:.3f}s (speedup: {speedup:.1f}x)")
            
            # Cache should provide significant speedup
            assert speedup > 2.0 or second_time < 0.001  # Either 2x speedup or very fast


@pytest.mark.performance
class TestCachePerformance:
    """Test caching system performance."""
    
    def test_cache_operations_speed(self):
        """Test cache operation speed."""
        cache = CrossReferenceCacheManager(max_size=1000, ttl=3600)
        
        # Test bulk cache operations
        operation_count = 1000
        
        # Test bulk set operations
        start_time = time.time()
        
        for i in range(operation_count):
            cache.set(f"key_{i:04d}", {"data": f"value_{i}", "index": i})
        
        set_time = time.time() - start_time
        
        # Test bulk get operations
        start_time = time.time()
        
        for i in range(operation_count):
            result = cache.get(f"key_{i:04d}")
            assert result is not None
            assert result["index"] == i
        
        get_time = time.time() - start_time
        
        # Performance assertions
        set_throughput = operation_count / set_time
        get_throughput = operation_count / get_time
        
        print(f"Cache performance - Set: {set_throughput:.0f} ops/s, Get: {get_throughput:.0f} ops/s")
        
        # Cache operations should be very fast
        assert set_throughput > 1000  # At least 1000 sets per second
        assert get_throughput > 5000  # At least 5000 gets per second
    
    def test_cache_memory_efficiency(self):
        """Test cache memory usage."""
        cache = CrossReferenceCacheManager(max_size=100, ttl=3600)
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Add many items to cache
        for i in range(200):  # More than max_size to test eviction
            large_data = {
                "id": f"item_{i:04d}",
                "data": "x" * 1000,  # 1KB of data per item
                "metadata": {"index": i, "created": time.time()}
            }
            cache.set(f"large_key_{i:04d}", large_data)
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Verify cache size limit is respected
        assert len(cache.cache) <= cache.max_size
        
        # Memory increase should be reasonable (less than 1MB for this test)
        assert memory_increase < 1024 * 1024  # Less than 1MB
        
        print(f"Cache memory usage: {memory_increase / 1024:.1f} KB for {cache.max_size} items")


@pytest.mark.performance
class TestErrorHandlingPerformance:
    """Test error handling performance and resilience."""
    
    def test_error_recovery_speed(self, character_creator_agent):
        """Test speed of error recovery."""
        # Test multiple error scenarios
        error_scenarios = [
            Exception("Network error"),
            TimeoutError("Request timeout"),
            ValueError("Invalid response"),
            KeyError("Missing key"),
            ConnectionError("Connection failed")
        ]
        
        recovery_times = []
        
        for error in error_scenarios:
            # Mock API to raise error
            character_creator_agent.openrouter_client.chat.completions.create.side_effect = error
            
            start_time = time.time()
            
            # Should recover gracefully with fallback
            result = character_creator_agent.create_character(
                name="Error Test Character",
                user_prompt="Test error handling"
            )
            
            recovery_time = time.time() - start_time
            recovery_times.append(recovery_time)
            
            # Verify fallback worked
            assert result['name'] == "Error Test Character"
            assert "fallback" in result['description'].lower()
        
        # Performance assertions
        avg_recovery_time = sum(recovery_times) / len(recovery_times)
        max_recovery_time = max(recovery_times)
        
        print(f"Error recovery performance - Avg: {avg_recovery_time:.3f}s, Max: {max_recovery_time:.3f}s")
        
        # Error recovery should be fast
        assert avg_recovery_time < 1.0  # Average under 1 second
        assert max_recovery_time < 2.0  # Maximum under 2 seconds
    
    def test_high_error_rate_resilience(self, character_creator_agent):
        """Test system resilience under high error rates."""
        # Simulate 80% error rate
        call_count = 0
        
        def mock_api_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count % 5 != 0:  # 80% failure rate
                raise ConnectionError("Simulated network error")
            else:
                # Successful response
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = '{"description": "Success", "age": "25", "occupation": "Test", "personality": "Resilient", "backstory": "Survived errors", "tags": ["test"], "role": "protagonist"}'
                return mock_response
        
        character_creator_agent.openrouter_client.chat.completions.create.side_effect = mock_api_call
        
        # Test multiple operations under high error rate
        success_count = 0
        total_operations = 20
        
        start_time = time.time()
        
        for i in range(total_operations):
            result = character_creator_agent.create_character(
                name=f"Resilience Test {i}",
                user_prompt="Test high error rate resilience"
            )
            
            # Should always return a result (either success or fallback)
            assert result is not None
            assert result['name'] == f"Resilience Test {i}"
            
            if "fallback" not in result['description'].lower():
                success_count += 1
        
        total_time = time.time() - start_time
        
        # Performance and resilience assertions
        success_rate = success_count / total_operations
        operations_per_second = total_operations / total_time
        
        print(f"High error rate resilience - Success rate: {success_rate:.1%}, Speed: {operations_per_second:.1f} ops/s")
        
        # Should maintain reasonable performance even with high error rate
        assert operations_per_second > 5.0  # At least 5 operations per second
        assert success_rate > 0.15  # At least 15% success rate (accounting for 80% error rate)


@pytest.mark.performance
@pytest.mark.slow
class TestStressTests:
    """Stress tests for system limits."""
    
    def test_memory_stress(self, mock_world_state):
        """Test system behavior under memory stress."""
        novel_id = "memory_stress_test"
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create many entities with large data
        entity_count = 500
        
        for i in range(entity_count):
            large_entity = {
                'id': f'stress_{i:05d}',
                'novel_id': novel_id,
                'name': f'Stress Test Entity {i}',
                'description': 'x' * 2000,  # 2KB description
                'backstory': 'y' * 3000,    # 3KB backstory
                'tags': [f'tag{j}' for j in range(20)]  # Many tags
            }
            
            result = mock_world_state.add_or_update(
                'characters', large_entity['id'], large_entity, skip_embeddings=True
            )
            assert result is True
        
        # Force garbage collection
        gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Verify entities were created
        novel_entities = mock_world_state.get_entities_by_novel(novel_id)
        assert len(novel_entities['characters']) == entity_count
        
        # Memory usage should be reasonable (less than 100MB for this test)
        memory_mb = memory_increase / (1024 * 1024)
        print(f"Memory stress test: {entity_count} large entities used {memory_mb:.1f} MB")
        
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
    
    def test_concurrent_stress(self, mock_world_state):
        """Test system under concurrent stress."""
        novel_id = "concurrent_stress_test"
        thread_count = 20
        operations_per_thread = 25
        
        results = []
        errors = []
        
        def stress_worker(thread_id):
            try:
                thread_results = []
                for i in range(operations_per_thread):
                    entity_data = {
                        'id': f'stress_t{thread_id:02d}_e{i:03d}',
                        'novel_id': novel_id,
                        'name': f'Stress Entity T{thread_id}-{i}',
                        'description': f'Concurrent stress test entity {i} from thread {thread_id}'
                    }
                    
                    result = mock_world_state.add_or_update(
                        'characters', entity_data['id'], entity_data, skip_embeddings=True
                    )
                    thread_results.append(result)
                
                results.extend(thread_results)
            except Exception as e:
                errors.append(e)
        
        # Start stress test
        start_time = time.time()
        
        threads = []
        for thread_id in range(thread_count):
            thread = threading.Thread(target=stress_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        stress_time = end_time - start_time
        
        # Verify results
        total_operations = thread_count * operations_per_thread
        
        assert len(errors) == 0, f"Stress test errors: {errors}"
        assert len(results) == total_operations
        assert all(result is True for result in results)
        
        # Performance metrics
        stress_throughput = total_operations / stress_time
        
        print(f"Concurrent stress test: {total_operations} operations in {stress_time:.2f}s ({stress_throughput:.1f} ops/s)")
        
        # Should maintain reasonable performance under stress
        assert stress_throughput > 50  # At least 50 operations per second
        
        # Verify data integrity
        novel_entities = mock_world_state.get_entities_by_novel(novel_id)
        assert len(novel_entities['characters']) == total_operations
