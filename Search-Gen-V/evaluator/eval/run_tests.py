#!/usr/bin/env python3
"""
Test runner for nugget evaluation framework

This script runs all unit tests for the run_id system and related functionality.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py -v                 # Verbose output
    python run_tests.py test_run_id_system # Run specific test module
"""

import sys
import unittest
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="Run tests for nugget evaluation framework"
    )
    parser.add_argument(
        'test_module', 
        nargs='?',
        help='Specific test module to run (e.g., test_run_id_system)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose test output'
    )
    parser.add_argument(
        '--pattern',
        default='test_*.py',
        help='Test file pattern (default: test_*.py)'
    )
    
    args = parser.parse_args()
    
    # Set verbosity level
    verbosity = 2 if args.verbose else 1
    
    if args.test_module:
        # Run specific test module
        try:
            # Import and run specific module
            test_module = __import__(f'tests.{args.test_module}', fromlist=[''])
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
        except ImportError as e:
            print(f" Failed to import test module '{args.test_module}': {e}")
            return 1
    else:
        # Discover and run all tests
        loader = unittest.TestLoader()
        start_dir = Path(__file__).parent / 'tests'
        suite = loader.discover(
            start_dir=str(start_dir),
            pattern=args.pattern,
            top_level_dir=str(Path(__file__).parent)
        )
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    tests_run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    
    print("\n" + "="*60)
    print(f"Tests run: {tests_run}")
    print(f"Failures: {failures}")
    print(f"Errors: {errors}")
    
    if result.wasSuccessful():
        print(" All tests passed!")
        return 0
    else:
        print(" Some tests failed!")
        
        if failures > 0:
            print("\nFailures:")
            for test, traceback in result.failures:
                error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
                print(f"  - {test}: {error_msg}")
        
        if errors > 0:
            print("\nErrors:")
            for test, traceback in result.errors:
                error_msg = traceback.split('\n')[-2]
                print(f"  - {test}: {error_msg}")
        
        return 1


if __name__ == '__main__':
    exit(main())