#!/usr/bin/env python3
"""
Local test runner script for Lazywriter development.

This script provides an easy way to run the same tests that the CI/CD pipeline runs,
helping developers catch issues before pushing to GitHub.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description, check=True):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=check, capture_output=False)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully in {end_time - start_time:.2f}s")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"❌ {description} failed in {end_time - start_time:.2f}s")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure all dependencies are installed: pip install -r requirements-dev.txt")
        return False


def setup_test_environment():
    """Set up test environment variables."""
    test_env = {
        'TINYDB_PATH': './test_data/tinydb',
        'CHROMADB_PATH': './test_data/chromadb',
        'USE_OPENROUTER_EMBEDDINGS': 'false',
        'FLASK_ENV': 'testing',
        'FLASK_DEBUG': 'false'
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    # Create test directories
    os.makedirs('./test_data/tinydb', exist_ok=True)
    os.makedirs('./test_data/chromadb', exist_ok=True)
    
    print("🔧 Test environment configured")


def run_linting():
    """Run code quality checks."""
    print("\n🧹 Running Code Quality Checks")
    
    checks = [
        ([sys.executable, "-m", "ruff", "check", "."], "Ruff linting"),
        ([sys.executable, "-m", "black", "--check", "--diff", "."], "Black formatting check"),
        ([sys.executable, "-m", "isort", "--check-only", "--diff", "."], "Import sorting check"),
    ]
    
    all_passed = True
    for cmd, description in checks:
        if not run_command(cmd, description, check=False):
            all_passed = False
    
    return all_passed


def run_security_checks():
    """Run security scans."""
    print("\n🔒 Running Security Checks")
    
    checks = [
        ([sys.executable, "-m", "safety", "check"], "Safety vulnerability check"),
        ([sys.executable, "-m", "bandit", "-r", ".", "-f", "json"], "Bandit security scan"),
    ]
    
    all_passed = True
    for cmd, description in checks:
        if not run_command(cmd, description, check=False):
            all_passed = False
    
    return all_passed


def run_tests(test_type="all"):
    """Run test suites."""
    print(f"\n🧪 Running Tests: {test_type}")
    
    if test_type == "all" or test_type == "unit":
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_agents.py", "tests/test_database.py", "tests/test_utils.py",
            "-m", "unit and not requires_api and not requires_models",
            "--tb=short", "--disable-warnings",
            "--cov=agents", "--cov=database", "--cov=utils",
            "--cov-report=term-missing"
        ]
        if not run_command(cmd, "Unit Tests", check=False):
            return False
    
    if test_type == "all" or test_type == "integration":
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_integration.py",
            "-m", "integration and not requires_api and not requires_models",
            "--tb=short", "--disable-warnings"
        ]
        if not run_command(cmd, "Integration Tests", check=False):
            return False
    
    if test_type == "all" or test_type == "flask":
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_flask_app.py", "tests/test_routes.py", "tests/test_templates.py",
            "--tb=short", "--disable-warnings",
            "--cov=app", "--cov-report=term-missing"
        ]
        if not run_command(cmd, "Flask Application Tests", check=False):
            return False
    
    return True


def run_build_test():
    """Test application startup."""
    print("\n🚀 Testing Application Build")
    
    test_script = '''
import sys
import os
sys.path.insert(0, ".")

# Set test environment
os.environ["TINYDB_PATH"] = "./test_data/tinydb"
os.environ["CHROMADB_PATH"] = "./test_data/chromadb"
os.environ["USE_OPENROUTER_EMBEDDINGS"] = "false"

try:
    from app import app
    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200
        print("✅ Flask application starts successfully")
        print("✅ Home route responds correctly")
except Exception as e:
    print(f"❌ Application startup failed: {e}")
    sys.exit(1)
'''
    
    cmd = [sys.executable, "-c", test_script]
    return run_command(cmd, "Application Build Test", check=False)


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run Lazywriter tests locally")
    parser.add_argument(
        "--type", 
        choices=["all", "lint", "security", "unit", "integration", "flask", "build"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Skip slow tests and security checks"
    )
    
    args = parser.parse_args()
    
    print("🎯 Lazywriter Local Test Runner")
    print(f"Running tests: {args.type}")
    
    # Setup
    setup_test_environment()
    
    all_passed = True
    
    # Run selected tests
    if args.type == "all":
        if not args.fast:
            all_passed &= run_linting()
            all_passed &= run_security_checks()
        all_passed &= run_tests("all")
        all_passed &= run_build_test()
    elif args.type == "lint":
        all_passed &= run_linting()
    elif args.type == "security":
        all_passed &= run_security_checks()
    elif args.type in ["unit", "integration", "flask"]:
        all_passed &= run_tests(args.type)
    elif args.type == "build":
        all_passed &= run_build_test()
    
    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 All tests passed! Your code is ready for CI/CD.")
    else:
        print("❌ Some tests failed. Please fix the issues before pushing.")
        sys.exit(1)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
