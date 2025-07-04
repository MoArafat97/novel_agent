#!/usr/bin/env python3
"""
Performance Benchmark for OptimizedEntityRecognizer

This script provides comprehensive performance benchmarks comparing the new
OptimizedEntityRecognizer system with the old regex-based approach, measuring
accuracy, speed, and resource usage.
"""

import sys
import os
import time
import statistics
import logging
from typing import Dict, List, Any, Tuple
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.cross_reference_agent import CrossReferenceAgent
from database import WorldState
from utils.stanza_entity_recognizer import OptimizedEntityRecognizer

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for benchmarking
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        """Initialize benchmark with test data."""
        self.world_state = self._create_comprehensive_world_state()
        self.test_cases = self._create_test_cases()
        self.results = {}
    
    def _create_comprehensive_world_state(self) -> WorldState:
        """Create a comprehensive world state with realistic data."""
        world_state = WorldState()
        novel_id = "benchmark-novel"
        
        # Create novel
        novel_data = {
            'id': novel_id,
            'title': 'Benchmark Fantasy Epic',
            'description': 'A comprehensive fantasy world for benchmarking'
        }
        world_state.add_or_update('novels', novel_id, novel_data)
        
        # Create 50 characters with varied names
        characters = [
            {'id': f'char-{i:03d}', 'novel_id': novel_id, 'name': name, 'description': desc, 'tags': tags}
            for i, (name, desc, tags) in enumerate([
                ('Aragorn Elessar', 'King of Gondor', ['king', 'ranger', 'warrior']),
                ('Gandalf the Grey', 'Wise wizard', ['wizard', 'mage', 'guide']),
                ('Legolas Greenleaf', 'Elven archer', ['elf', 'archer', 'prince']),
                ('Gimli son of Gloin', 'Dwarf warrior', ['dwarf', 'warrior', 'axe']),
                ('Frodo Baggins', 'Hobbit ring-bearer', ['hobbit', 'ring-bearer', 'brave']),
                ('Samwise Gamgee', 'Loyal gardener', ['hobbit', 'gardener', 'loyal']),
                ('Boromir of Gondor', 'Captain of the Guard', ['human', 'captain', 'noble']),
                ('Faramir the Wise', 'Steward of Gondor', ['human', 'steward', 'wise']),
                ('Éowyn Dernhelm', 'Shieldmaiden of Rohan', ['human', 'warrior', 'noble']),
                ('Théoden King', 'Lord of the Mark', ['human', 'king', 'horseman']),
                ('Arwen Undómiel', 'Elven princess', ['elf', 'princess', 'healer']),
                ('Elrond Half-elven', 'Lord of Rivendell', ['elf', 'lord', 'wise']),
                ('Galadriel the Fair', 'Lady of Lothlórien', ['elf', 'lady', 'powerful']),
                ('Saruman the White', 'Fallen wizard', ['wizard', 'traitor', 'corrupt']),
                ('Sauron the Dark Lord', 'Enemy of all', ['dark-lord', 'evil', 'powerful']),
                ('Tom Bombadil', 'Master of the Old Forest', ['enigma', 'ancient', 'powerful']),
                ('Treebeard Fangorn', 'Eldest of Ents', ['ent', 'ancient', 'guardian']),
                ('Denethor the Mad', 'Steward of Gondor', ['human', 'steward', 'mad']),
                ('Bilbo Baggins', 'The Burglar', ['hobbit', 'burglar', 'adventurer']),
                ('Merry Brandybuck', 'Knight of Rohan', ['hobbit', 'knight', 'brave']),
                ('Pippin Took', 'Guard of Gondor', ['hobbit', 'guard', 'young']),
                ('Éomer Marshal', 'Rider of Rohan', ['human', 'marshal', 'warrior']),
                ('Haldir of Lothlórien', 'Elven guardian', ['elf', 'guardian', 'archer']),
                ('Celeborn the Wise', 'Lord of Lothlórien', ['elf', 'lord', 'ancient']),
                ('Glorfindel the Bold', 'Elven lord', ['elf', 'lord', 'warrior']),
                ('Radagast the Brown', 'Wizard of nature', ['wizard', 'nature', 'animals']),
                ('Beorn the Skin-changer', 'Guardian of the wild', ['shape-shifter', 'guardian', 'wild']),
                ('Bard the Bowman', 'Dragon-slayer', ['human', 'archer', 'hero']),
                ('Thranduil Elvenking', 'King of the Woodland Realm', ['elf', 'king', 'proud']),
                ('Azog the Defiler', 'Orc chieftain', ['orc', 'chieftain', 'evil']),
                ('Bolg of the North', 'Orc commander', ['orc', 'commander', 'brutal']),
                ('Gothmog Lieutenant', 'Orc general', ['orc', 'general', 'fierce']),
                ('Mouth of Sauron', 'Dark messenger', ['human', 'messenger', 'corrupt']),
                ('Witch-king of Angmar', 'Lord of the Nazgûl', ['nazgul', 'lord', 'undead']),
                ('Shelob the Great', 'Giant spider', ['spider', 'ancient', 'evil']),
                ('Sméagol Gollum', 'Wretched creature', ['creature', 'corrupted', 'obsessed']),
                ('Déagol the Fisher', 'Hobbit-like being', ['hobbit-like', 'fisher', 'victim']),
                ('Old Man Willow', 'Malicious tree', ['tree', 'malicious', 'ancient']),
                ('Goldberry River-daughter', 'Water spirit', ['spirit', 'water', 'beautiful']),
                ('Quickbeam Bregalad', 'Hasty Ent', ['ent', 'hasty', 'young']),
                ('Shadowfax Lord of Horses', 'Mearas stallion', ['horse', 'mearas', 'swift']),
                ('Roheryn Aragorn\'s Horse', 'Faithful steed', ['horse', 'faithful', 'strong']),
                ('Asfaloth Glorfindel\'s Horse', 'White horse', ['horse', 'white', 'elven']),
                ('Brego Éomer\'s Horse', 'War horse', ['horse', 'war', 'brave']),
                ('Hasufel Borrowed Horse', 'Rohan horse', ['horse', 'rohan', 'swift']),
                ('Arod Legolas\'s Horse', 'Grey horse', ['horse', 'grey', 'sure-footed']),
                ('Snowmane Théoden\'s Horse', 'Royal steed', ['horse', 'royal', 'white']),
                ('Windfola Éowyn\'s Horse', 'Swift mare', ['horse', 'mare', 'swift']),
                ('Firefoot Éomer\'s Warhorse', 'Battle steed', ['horse', 'battle', 'fierce']),
                ('Stybba Merry\'s Pony', 'Small pony', ['pony', 'small', 'sturdy'])
            ])
        ]
        
        for char in characters:
            world_state.add_or_update('characters', char['id'], char)
        
        # Create 30 locations
        locations = [
            {'id': f'loc-{i:03d}', 'novel_id': novel_id, 'name': name, 'description': desc, 'tags': tags}
            for i, (name, desc, tags) in enumerate([
                ('Rivendell', 'Elven refuge', ['elven', 'refuge', 'safe']),
                ('Gondor', 'Kingdom of men', ['kingdom', 'men', 'south']),
                ('Rohan', 'Land of horses', ['kingdom', 'horses', 'plains']),
                ('Lothlórien', 'Golden wood', ['elven', 'forest', 'golden']),
                ('Moria', 'Dwarven mines', ['dwarven', 'mines', 'dangerous']),
                ('Isengard', 'Saruman\'s tower', ['tower', 'fortress', 'corrupt']),
                ('Mordor', 'Land of shadow', ['evil', 'shadow', 'dark']),
                ('Shire', 'Hobbit homeland', ['hobbit', 'peaceful', 'green']),
                ('Minas Tirith', 'City of kings', ['city', 'white', 'fortress']),
                ('Helm\'s Deep', 'Ancient fortress', ['fortress', 'ancient', 'strong']),
                ('Edoras', 'Capital of Rohan', ['city', 'rohan', 'golden']),
                ('Osgiliath', 'Ruined city', ['city', 'ruined', 'ancient']),
                ('Weathertop', 'Ancient watchtower', ['tower', 'ancient', 'ruins']),
                ('Bree', 'Town of men', ['town', 'men', 'crossroads']),
                ('Lake-town', 'Town on the lake', ['town', 'lake', 'trade']),
                ('Erebor', 'Lonely Mountain', ['mountain', 'dwarven', 'kingdom']),
                ('Mirkwood', 'Dark forest', ['forest', 'dark', 'dangerous']),
                ('Fangorn Forest', 'Home of Ents', ['forest', 'ents', 'ancient']),
                ('Dead Marshes', 'Haunted swamps', ['marshes', 'haunted', 'dead']),
                ('Pelennor Fields', 'Great battlefield', ['fields', 'battle', 'vast']),
                ('Cair Andros', 'River fortress', ['fortress', 'river', 'strategic']),
                ('Dol Guldur', 'Hill of sorcery', ['fortress', 'evil', 'sorcery']),
                ('Barad-dûr', 'Dark tower', ['tower', 'dark', 'evil']),
                ('Mount Doom', 'Fiery mountain', ['mountain', 'fire', 'doom']),
                ('Grey Havens', 'Elven port', ['port', 'elven', 'departure']),
                ('Bag End', 'Bilbo\'s home', ['home', 'hobbit', 'comfortable']),
                ('Prancing Pony', 'Inn at Bree', ['inn', 'bree', 'meeting']),
                ('Green Dragon', 'Hobbiton inn', ['inn', 'hobbiton', 'cozy']),
                ('Bucklebury Ferry', 'River crossing', ['ferry', 'river', 'crossing']),
                ('Old Forest', 'Ancient woodland', ['forest', 'ancient', 'mysterious'])
            ])
        ]
        
        for loc in locations:
            world_state.add_or_update('locations', loc['id'], loc)
        
        # Create 20 lore items
        lore_items = [
            {'id': f'lore-{i:03d}', 'novel_id': novel_id, 'name': name, 'description': desc, 'tags': tags}
            for i, (name, desc, tags) in enumerate([
                ('One Ring', 'Master ring of power', ['ring', 'power', 'evil']),
                ('Sting', 'Bilbo\'s elven sword', ['sword', 'elven', 'glowing']),
                ('Glamdring', 'Gandalf\'s sword', ['sword', 'ancient', 'powerful']),
                ('Orcrist', 'Thorin\'s sword', ['sword', 'goblin-cleaver', 'ancient']),
                ('Andúril', 'Flame of the West', ['sword', 'reforged', 'royal']),
                ('Palantír', 'Seeing stones', ['stone', 'seeing', 'dangerous']),
                ('Silmarils', 'Holy jewels', ['jewel', 'holy', 'light']),
                ('Arkenstone', 'Heart of the Mountain', ['jewel', 'dwarven', 'precious']),
                ('Mithril', 'True silver', ['metal', 'precious', 'light']),
                ('Lembas', 'Elven waybread', ['food', 'elven', 'sustaining']),
                ('Miruvor', 'Cordial of Rivendell', ['drink', 'healing', 'elven']),
                ('Athelas', 'Kingsfoil herb', ['herb', 'healing', 'royal']),
                ('Mallorn Trees', 'Golden trees', ['tree', 'golden', 'elven']),
                ('Ent-draught', 'Water of the Ents', ['drink', 'ent', 'strengthening']),
                ('Old Toby', 'Finest pipe-weed', ['tobacco', 'hobbit', 'finest']),
                ('Longbottom Leaf', 'Premium pipe-weed', ['tobacco', 'hobbit', 'premium']),
                ('Barrow-blades', 'Ancient daggers', ['dagger', 'ancient', 'blessed']),
                ('Phial of Galadriel', 'Light of Eärendil', ['phial', 'light', 'elven']),
                ('Horn of Gondor', 'Boromir\'s horn', ['horn', 'gondor', 'noble']),
                ('Vilya', 'Ring of Air', ['ring', 'elven', 'air'])
            ])
        ]
        
        for lore in lore_items:
            world_state.add_or_update('lore', lore['id'], lore)
        
        return world_state
    
    def _create_test_cases(self) -> List[Dict[str, Any]]:
        """Create comprehensive test cases with varying complexity."""
        return [
            {
                'name': 'Simple Mentions',
                'content': 'Aragorn walked with Gandalf to Rivendell.',
                'expected_entities': ['Aragorn', 'Gandalf', 'Rivendell'],
                'complexity': 'low'
            },
            {
                'name': 'Complex Narrative',
                'content': '''
                In the halls of Minas Tirith, Aragorn was crowned King of Gondor by Gandalf the Grey.
                Legolas and Gimli stood witness, while Arwen watched from the balcony.
                The people of Rohan, led by Éomer, cheered as Andúril was raised high.
                Even the Ents from Fangorn Forest had come to see this historic moment.
                ''',
                'expected_entities': ['Minas Tirith', 'Aragorn', 'Gondor', 'Gandalf', 'Legolas', 'Gimli', 'Arwen', 'Rohan', 'Éomer', 'Andúril', 'Ents', 'Fangorn Forest'],
                'complexity': 'high'
            },
            {
                'name': 'Ambiguous References',
                'content': '''
                The ring was found by someone near the river. A wizard came looking for it,
                but the finder had already left for the mountains. The king\'s men searched
                everywhere, but the precious was nowhere to be found.
                ''',
                'expected_entities': ['ring', 'wizard', 'king'],
                'complexity': 'medium'
            },
            {
                'name': 'Dense Entity Text',
                'content': '''
                At the Council of Elrond in Rivendell, Frodo presented the One Ring.
                Boromir of Gondor argued with Aragorn, while Legolas and Gimli debated.
                Gandalf spoke of Saruman\'s betrayal at Isengard, and how the palantír
                had corrupted him. The Fellowship was formed with members from all races:
                Frodo and Sam from the Shire, Merry and Pippin as well, Boromir from
                Minas Tirith, Legolas from Mirkwood, Gimli from Erebor, and Aragorn
                the Ranger who would become King of Gondor.
                ''',
                'expected_entities': ['Elrond', 'Rivendell', 'Frodo', 'One Ring', 'Boromir', 'Gondor', 'Aragorn', 'Legolas', 'Gimli', 'Gandalf', 'Saruman', 'Isengard', 'palantír', 'Fellowship', 'Sam', 'Shire', 'Merry', 'Pippin', 'Minas Tirith', 'Mirkwood', 'Erebor'],
                'complexity': 'very_high'
            },
            {
                'name': 'Partial Matches',
                'content': '''
                The Grey Wizard met with the Elven Lord at the White City.
                The Ranger spoke with the Dwarf Lord about the Dark Tower.
                The Ring-bearer carried the Precious through the Dead Marshes.
                ''',
                'expected_entities': ['Grey', 'Wizard', 'Elven', 'Lord', 'White', 'City', 'Ranger', 'Dwarf', 'Dark', 'Tower', 'Ring-bearer', 'Precious', 'Dead Marshes'],
                'complexity': 'medium'
            }
        ]

    def benchmark_accuracy(self) -> Dict[str, Any]:
        """Benchmark accuracy of entity recognition."""
        print("\n=== Accuracy Benchmark ===")

        # Initialize both systems
        cross_ref_agent = CrossReferenceAgent(world_state=self.world_state)
        novel_id = "benchmark-novel"

        accuracy_results = {
            'test_cases': [],
            'overall_stats': {
                'optimized': {'precision': 0, 'recall': 0, 'f1': 0},
                'legacy': {'precision': 0, 'recall': 0, 'f1': 0}
            }
        }

        total_optimized_precision = []
        total_optimized_recall = []
        total_legacy_precision = []
        total_legacy_recall = []

        for test_case in self.test_cases:
            print(f"\nTesting: {test_case['name']} ({test_case['complexity']} complexity)")

            # Test optimized system
            optimized_results = cross_ref_agent._detect_entity_mentions(
                test_case['content'], novel_id
            )
            optimized_entities = set(r['entity_name'].lower() for r in optimized_results)

            # Test legacy system (disable optimized recognizer temporarily)
            original_recognizer = cross_ref_agent.optimized_recognizer
            cross_ref_agent.optimized_recognizer = None

            legacy_results = cross_ref_agent._detect_entity_mentions(
                test_case['content'], novel_id
            )
            legacy_entities = set(r['entity_name'].lower() for r in legacy_results)

            # Restore optimized recognizer
            cross_ref_agent.optimized_recognizer = original_recognizer

            # Calculate metrics
            expected = set(e.lower() for e in test_case['expected_entities'])

            # Optimized metrics
            opt_tp = len(optimized_entities & expected)
            opt_fp = len(optimized_entities - expected)
            opt_fn = len(expected - optimized_entities)

            opt_precision = opt_tp / (opt_tp + opt_fp) if (opt_tp + opt_fp) > 0 else 0
            opt_recall = opt_tp / (opt_tp + opt_fn) if (opt_tp + opt_fn) > 0 else 0
            opt_f1 = 2 * (opt_precision * opt_recall) / (opt_precision + opt_recall) if (opt_precision + opt_recall) > 0 else 0

            # Legacy metrics
            leg_tp = len(legacy_entities & expected)
            leg_fp = len(legacy_entities - expected)
            leg_fn = len(expected - legacy_entities)

            leg_precision = leg_tp / (leg_tp + leg_fp) if (leg_tp + leg_fp) > 0 else 0
            leg_recall = leg_tp / (leg_tp + leg_fn) if (leg_tp + leg_fn) > 0 else 0
            leg_f1 = 2 * (leg_precision * leg_recall) / (leg_precision + leg_recall) if (leg_precision + leg_recall) > 0 else 0

            # Store results
            test_result = {
                'name': test_case['name'],
                'complexity': test_case['complexity'],
                'expected_count': len(expected),
                'optimized': {
                    'found': len(optimized_entities),
                    'precision': opt_precision,
                    'recall': opt_recall,
                    'f1': opt_f1,
                    'entities': list(optimized_entities)
                },
                'legacy': {
                    'found': len(legacy_entities),
                    'precision': leg_precision,
                    'recall': leg_recall,
                    'f1': leg_f1,
                    'entities': list(legacy_entities)
                }
            }

            accuracy_results['test_cases'].append(test_result)

            # Accumulate for overall stats
            total_optimized_precision.append(opt_precision)
            total_optimized_recall.append(opt_recall)
            total_legacy_precision.append(leg_precision)
            total_legacy_recall.append(leg_recall)

            print(f"  Expected: {len(expected)} entities")
            print(f"  Optimized: {len(optimized_entities)} found, P={opt_precision:.2f}, R={opt_recall:.2f}, F1={opt_f1:.2f}")
            print(f"  Legacy: {len(legacy_entities)} found, P={leg_precision:.2f}, R={leg_recall:.2f}, F1={leg_f1:.2f}")

        # Calculate overall statistics
        accuracy_results['overall_stats']['optimized'] = {
            'precision': statistics.mean(total_optimized_precision),
            'recall': statistics.mean(total_optimized_recall),
            'f1': statistics.mean([tc['optimized']['f1'] for tc in accuracy_results['test_cases']])
        }

        accuracy_results['overall_stats']['legacy'] = {
            'precision': statistics.mean(total_legacy_precision),
            'recall': statistics.mean(total_legacy_recall),
            'f1': statistics.mean([tc['legacy']['f1'] for tc in accuracy_results['test_cases']])
        }

        return accuracy_results

    def benchmark_speed(self) -> Dict[str, Any]:
        """Benchmark processing speed."""
        print("\n=== Speed Benchmark ===")

        cross_ref_agent = CrossReferenceAgent(world_state=self.world_state)
        novel_id = "benchmark-novel"

        speed_results = {
            'test_cases': [],
            'overall_stats': {
                'optimized': {'avg_time': 0, 'min_time': 0, 'max_time': 0},
                'legacy': {'avg_time': 0, 'min_time': 0, 'max_time': 0}
            }
        }

        all_optimized_times = []
        all_legacy_times = []

        for test_case in self.test_cases:
            print(f"\nTesting speed: {test_case['name']}")

            # Benchmark optimized system (multiple runs)
            optimized_times = []
            for _ in range(5):  # 5 runs for average
                start_time = time.time()
                cross_ref_agent._detect_entity_mentions(test_case['content'], novel_id)
                end_time = time.time()
                optimized_times.append(end_time - start_time)

            # Benchmark legacy system
            original_recognizer = cross_ref_agent.optimized_recognizer
            cross_ref_agent.optimized_recognizer = None

            legacy_times = []
            for _ in range(5):  # 5 runs for average
                start_time = time.time()
                cross_ref_agent._detect_entity_mentions(test_case['content'], novel_id)
                end_time = time.time()
                legacy_times.append(end_time - start_time)

            # Restore optimized recognizer
            cross_ref_agent.optimized_recognizer = original_recognizer

            # Calculate statistics
            opt_avg = statistics.mean(optimized_times)
            opt_min = min(optimized_times)
            opt_max = max(optimized_times)

            leg_avg = statistics.mean(legacy_times)
            leg_min = min(legacy_times)
            leg_max = max(legacy_times)

            speedup = leg_avg / opt_avg if opt_avg > 0 else float('inf')

            test_result = {
                'name': test_case['name'],
                'complexity': test_case['complexity'],
                'optimized': {
                    'avg_time': opt_avg,
                    'min_time': opt_min,
                    'max_time': opt_max
                },
                'legacy': {
                    'avg_time': leg_avg,
                    'min_time': leg_min,
                    'max_time': leg_max
                },
                'speedup': speedup
            }

            speed_results['test_cases'].append(test_result)
            all_optimized_times.extend(optimized_times)
            all_legacy_times.extend(legacy_times)

            print(f"  Optimized: {opt_avg:.4f}s avg ({opt_min:.4f}-{opt_max:.4f}s)")
            print(f"  Legacy: {leg_avg:.4f}s avg ({leg_min:.4f}-{leg_max:.4f}s)")
            print(f"  Speedup: {speedup:.1f}x")

        # Overall statistics
        speed_results['overall_stats']['optimized'] = {
            'avg_time': statistics.mean(all_optimized_times),
            'min_time': min(all_optimized_times),
            'max_time': max(all_optimized_times)
        }

        speed_results['overall_stats']['legacy'] = {
            'avg_time': statistics.mean(all_legacy_times),
            'min_time': min(all_legacy_times),
            'max_time': max(all_legacy_times)
        }

        return speed_results

    def benchmark_caching_effectiveness(self) -> Dict[str, Any]:
        """Benchmark caching system effectiveness."""
        print("\n=== Caching Effectiveness Benchmark ===")

        cross_ref_agent = CrossReferenceAgent(world_state=self.world_state)
        novel_id = "benchmark-novel"

        if not hasattr(cross_ref_agent, 'optimized_recognizer') or not cross_ref_agent.optimized_recognizer:
            print("OptimizedEntityRecognizer not available - skipping caching benchmark")
            return {}

        recognizer = cross_ref_agent.optimized_recognizer
        test_content = self.test_cases[1]['content']  # Use complex narrative

        # Clear cache
        recognizer.clear_cache()

        # First run (no cache)
        start_time = time.time()
        recognizer.recognize_entities(test_content, novel_id)
        first_run_time = time.time() - start_time

        # Second run (with cache)
        start_time = time.time()
        recognizer.recognize_entities(test_content, novel_id)
        cached_run_time = time.time() - start_time

        # Multiple cached runs
        cached_times = []
        for _ in range(10):
            start_time = time.time()
            recognizer.recognize_entities(test_content, novel_id)
            cached_times.append(time.time() - start_time)

        cache_stats = recognizer.get_cache_stats()
        rel_stats = recognizer.get_relationship_cache_stats()

        caching_results = {
            'first_run_time': first_run_time,
            'cached_run_time': cached_run_time,
            'avg_cached_time': statistics.mean(cached_times),
            'cache_speedup': first_run_time / statistics.mean(cached_times) if statistics.mean(cached_times) > 0 else float('inf'),
            'cache_stats': cache_stats,
            'relationship_stats': rel_stats
        }

        print(f"  First run (no cache): {first_run_time:.4f}s")
        print(f"  Cached runs average: {statistics.mean(cached_times):.4f}s")
        print(f"  Cache speedup: {caching_results['cache_speedup']:.1f}x")
        print(f"  Cache utilization: {cache_stats['cache_utilization']:.1%}")
        print(f"  Relationship patterns learned: {rel_stats['total_patterns']}")

        return caching_results

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and compile results."""
        print("Starting Comprehensive Performance Benchmark")
        print("=" * 60)

        results = {
            'timestamp': time.time(),
            'system_info': {
                'test_cases': len(self.test_cases),
                'entities_in_world': {
                    'characters': len([c for c in self.world_state.get_all('characters') if c.get('novel_id') == 'benchmark-novel']),
                    'locations': len([l for l in self.world_state.get_all('locations') if l.get('novel_id') == 'benchmark-novel']),
                    'lore': len([l for l in self.world_state.get_all('lore') if l.get('novel_id') == 'benchmark-novel'])
                }
            }
        }

        try:
            # Run accuracy benchmark
            results['accuracy'] = self.benchmark_accuracy()

            # Run speed benchmark
            results['speed'] = self.benchmark_speed()

            # Run caching benchmark
            results['caching'] = self.benchmark_caching_effectiveness()

            # Generate summary
            results['summary'] = self._generate_summary(results)

        except Exception as e:
            print(f"Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

        return results

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {}

        if 'accuracy' in results:
            acc = results['accuracy']['overall_stats']
            summary['accuracy'] = {
                'optimized_f1': acc['optimized']['f1'],
                'legacy_f1': acc['legacy']['f1'],
                'improvement': acc['optimized']['f1'] - acc['legacy']['f1'],
                'meets_80_percent_target': acc['optimized']['f1'] >= 0.8
            }

        if 'speed' in results:
            speed = results['speed']['overall_stats']
            summary['speed'] = {
                'optimized_avg': speed['optimized']['avg_time'],
                'legacy_avg': speed['legacy']['avg_time'],
                'speedup': speed['legacy']['avg_time'] / speed['optimized']['avg_time'] if speed['optimized']['avg_time'] > 0 else float('inf'),
                'meets_2s_target': speed['optimized']['avg_time'] <= 2.0
            }

        if 'caching' in results:
            summary['caching'] = {
                'cache_speedup': results['caching']['cache_speedup'],
                'cache_utilization': results['caching']['cache_stats']['cache_utilization'],
                'relationship_patterns': results['caching']['relationship_stats']['total_patterns']
            }

        return summary

    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        if 'error' in results:
            print(f"❌ Benchmark failed: {results['error']}")
            return

        summary = results.get('summary', {})

        # Accuracy Summary
        if 'accuracy' in summary:
            acc = summary['accuracy']
            print(f"\n📊 ACCURACY RESULTS:")
            print(f"  Optimized System F1: {acc['optimized_f1']:.2%}")
            print(f"  Legacy System F1: {acc['legacy_f1']:.2%}")
            print(f"  Improvement: {acc['improvement']:+.2%}")
            print(f"  80% Target Met: {'✅ YES' if acc['meets_80_percent_target'] else '❌ NO'}")

        # Speed Summary
        if 'speed' in summary:
            speed = summary['speed']
            print(f"\n⚡ SPEED RESULTS:")
            print(f"  Optimized System: {speed['optimized_avg']:.3f}s avg")
            print(f"  Legacy System: {speed['legacy_avg']:.3f}s avg")
            print(f"  Speedup: {speed['speedup']:.1f}x faster")
            print(f"  2s Target Met: {'✅ YES' if speed['meets_2s_target'] else '❌ NO'}")

        # Caching Summary
        if 'caching' in summary:
            cache = summary['caching']
            print(f"\n🚀 CACHING RESULTS:")
            print(f"  Cache Speedup: {cache['cache_speedup']:.1f}x faster")
            print(f"  Cache Utilization: {cache['cache_utilization']:.1%}")
            print(f"  Relationship Patterns: {cache['relationship_patterns']} learned")

        # Overall Assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        accuracy_pass = summary.get('accuracy', {}).get('meets_80_percent_target', False)
        speed_pass = summary.get('speed', {}).get('meets_2s_target', False)

        if accuracy_pass and speed_pass:
            print("  ✅ ALL TARGETS MET - System ready for production!")
        elif accuracy_pass:
            print("  ⚠️  Accuracy target met, but speed needs improvement")
        elif speed_pass:
            print("  ⚠️  Speed target met, but accuracy needs improvement")
        else:
            print("  ❌ Both accuracy and speed targets need improvement")

def main():
    """Run the comprehensive benchmark."""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark()

    # Print summary
    benchmark.print_summary(results)

    # Save detailed results
    output_file = f"benchmark_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Detailed results saved to: {output_file}")

    return 0 if 'error' not in results else 1

if __name__ == "__main__":
    exit(main())
