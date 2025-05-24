"""
Test runner script for the algorithmic trading bot.
"""

import unittest
import sys
import os
from io import StringIO

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests(test_pattern='test_*.py', verbosity=2):
    """
    Run all tests matching the pattern.
    
    Args:
        test_pattern: Pattern to match test files
        verbosity: Test output verbosity level
    """
    # Discover and run tests
    loader = unittest.TestLoader()
    test_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(test_dir, pattern=test_pattern)
    
    # Create test runner
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbosity)
    
    # Run tests
    print(f"Running tests with pattern: {test_pattern}")
    print("=" * 70)
    
    result = runner.run(suite)
    
    # Print results
    output = stream.getvalue()
    print(output)
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    # Return success/failure
    return len(result.failures) == 0 and len(result.errors) == 0

def run_specific_test(test_module, test_class=None, test_method=None):
    """
    Run a specific test module, class, or method.
    
    Args:
        test_module: Module name (e.g., 'test_strategies')
        test_class: Optional class name (e.g., 'TestMeanReversionStrategy')
        test_method: Optional method name (e.g., 'test_initialization')
    """
    if test_method and test_class:
        test_name = f"{test_module}.{test_class}.{test_method}"
    elif test_class:
        test_name = f"{test_module}.{test_class}"
    else:
        test_name = test_module
    
    suite = unittest.TestLoader().loadTestsFromName(test_name)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0

def run_coverage():
    """Run tests with coverage reporting."""
    try:
        import coverage
        
        # Start coverage
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        success = run_tests()
        
        # Stop coverage and report
        cov.stop()
        cov.save()
        
        print("\n" + "=" * 70)
        print("COVERAGE REPORT")
        print("=" * 70)
        cov.report(show_missing=True)
        
        # Generate HTML report
        html_dir = os.path.join(os.path.dirname(__file__), '..', 'coverage_html')
        cov.html_report(directory=html_dir)
        print(f"\nHTML coverage report generated in: {html_dir}")
        
        return success
        
    except ImportError:
        print("Coverage.py not installed. Run: pip install coverage")
        return run_tests()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run trading bot tests')
    parser.add_argument('--pattern', default='test_*.py', 
                       help='Test file pattern to match')
    parser.add_argument('--module', help='Specific test module to run')
    parser.add_argument('--class', dest='test_class', 
                       help='Specific test class to run')
    parser.add_argument('--method', help='Specific test method to run')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage reporting')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    verbosity = 2 if args.verbose else 1
    
    if args.module:
        success = run_specific_test(args.module, args.test_class, args.method)
    elif args.coverage:
        success = run_coverage()
    else:
        success = run_tests(args.pattern, verbosity)
    
    sys.exit(0 if success else 1)
