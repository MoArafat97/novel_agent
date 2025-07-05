#!/usr/bin/env python3
"""
Comprehensive test runner for the Lazywriter system.

This script provides different test execution modes and comprehensive
reporting for all test suites in the system.
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRunner:
    """Comprehensive test runner with multiple execution modes."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        
    def run_unit_tests(self, verbose=False):
        """Run unit tests only."""
        print("🧪 Running Unit Tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "test_agents.py"),
            str(self.test_dir / "test_database.py"),
            str(self.test_dir / "test_utils.py"),
            "-m", "unit",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_integration_tests(self, verbose=False):
        """Run integration tests only."""
        print("🔗 Running Integration Tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "test_integration.py"),
            "-m", "integration",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_performance_tests(self, verbose=False):
        """Run performance tests only."""
        print("⚡ Running Performance Tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "test_performance.py"),
            "-m", "performance",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_quick_tests(self, verbose=False):
        """Run quick tests (excluding slow and API-dependent tests)."""
        print("🚀 Running Quick Tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "-m", "not slow and not requires_api and not requires_models",
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_all_tests(self, verbose=False, coverage=False):
        """Run all tests with optional coverage reporting."""
        print("🎯 Running All Tests...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend([
                "--cov=agents",
                "--cov=database", 
                "--cov=utils",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing"
            ])
        
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_specific_test(self, test_path, verbose=False):
        """Run a specific test file or test function."""
        print(f"🎯 Running Specific Test: {test_path}")
        
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        return subprocess.run(cmd, cwd=self.project_root)
    
    def run_smoke_tests(self):
        """Run basic smoke tests to verify system functionality."""
        print("💨 Running Smoke Tests...")
        
        smoke_tests = [
            "test_agents.py::TestCharacterCreatorAgent::test_initialization",
            "test_database.py::TestWorldState::test_initialization",
            "test_utils.py::TestEntityDetectionUtils::test_initialization"
        ]
        
        for test in smoke_tests:
            cmd = [
                sys.executable, "-m", "pytest",
                str(self.test_dir / test),
                "-v"
            ]
            
            result = subprocess.run(cmd, cwd=self.project_root)
            if result.returncode != 0:
                print(f"❌ Smoke test failed: {test}")
                return result
        
        print("✅ All smoke tests passed!")
        return subprocess.CompletedProcess([], 0)
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("📊 Generating Test Report...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "--tb=short",
            "--cov=agents",
            "--cov=database",
            "--cov=utils",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "--junit-xml=test-results.xml"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0:
            print("✅ Test report generated successfully!")
            print("📁 HTML coverage report: htmlcov/index.html")
            print("📄 JUnit XML report: test-results.xml")
            print("📊 JSON coverage report: coverage.json")
        
        return result
    
    def check_test_environment(self):
        """Check if test environment is properly set up."""
        print("🔍 Checking Test Environment...")
        
        # Check required packages
        required_packages = [
            'pytest', 'pytest-mock', 'pytest-cov', 'pytest-asyncio'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        # Check test files exist
        test_files = [
            'test_agents.py',
            'test_database.py', 
            'test_utils.py',
            'test_integration.py',
            'test_performance.py'
        ]
        
        missing_files = []
        for test_file in test_files:
            if not (self.test_dir / test_file).exists():
                missing_files.append(test_file)
        
        if missing_files:
            print(f"❌ Missing test files: {', '.join(missing_files)}")
            return False
        
        # Check project structure
        required_dirs = ['agents', 'database', 'utils']
        missing_dirs = []
        
        for dir_name in required_dirs:
            if not (self.project_root / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"❌ Missing project directories: {', '.join(missing_dirs)}")
            return False
        
        print("✅ Test environment is properly configured!")
        return True
    
    def run_benchmark_suite(self):
        """Run comprehensive benchmark suite."""
        print("📈 Running Benchmark Suite...")
        
        # Run performance tests with detailed output
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "test_performance.py"),
            "-v",
            "-s",  # Don't capture output so we can see performance metrics
            "--tb=short"
        ]
        
        return subprocess.run(cmd, cwd=self.project_root)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Lazywriter Test Runner")
    
    parser.add_argument(
        'mode',
        choices=['unit', 'integration', 'performance', 'quick', 'all', 'smoke', 'report', 'benchmark', 'check'],
        help='Test execution mode'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Generate coverage report (for all mode)'
    )
    
    parser.add_argument(
        '--test', '-t',
        help='Run specific test file or function'
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Check environment first
    if not runner.check_test_environment():
        print("❌ Test environment check failed!")
        return 1
    
    start_time = time.time()
    
    # Execute requested test mode
    if args.mode == 'unit':
        result = runner.run_unit_tests(args.verbose)
    elif args.mode == 'integration':
        result = runner.run_integration_tests(args.verbose)
    elif args.mode == 'performance':
        result = runner.run_performance_tests(args.verbose)
    elif args.mode == 'quick':
        result = runner.run_quick_tests(args.verbose)
    elif args.mode == 'all':
        result = runner.run_all_tests(args.verbose, args.coverage)
    elif args.mode == 'smoke':
        result = runner.run_smoke_tests()
    elif args.mode == 'report':
        result = runner.generate_test_report()
    elif args.mode == 'benchmark':
        result = runner.run_benchmark_suite()
    elif args.mode == 'check':
        # Environment check already done above
        return 0
    elif args.test:
        result = runner.run_specific_test(args.test, args.verbose)
    else:
        print("❌ Invalid test mode")
        return 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  Test execution completed in {duration:.2f} seconds")
    
    if result.returncode == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
