"""Test runner script for the fish-logging application.

This script runs all tests and generates a coverage report.
"""
import sys
import subprocess


def run_tests():
    """Run all tests with pytest."""
    print("=" * 70)
    print("Running Fish Logging Application Tests")
    print("=" * 70)
    print()

    # Run pytest with coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--color=yes",
    ]

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("ERROR: pytest not found. Install it with: pip install pytest")
        return 1


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    print("=" * 70)
    print("Running Tests with Coverage Report")
    print("=" * 70)
    print()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--cov=.",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-v",
    ]

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print()
            print("=" * 70)
            print("Coverage report generated in htmlcov/index.html")
            print("=" * 70)
        return result.returncode
    except FileNotFoundError:
        print("ERROR: pytest-cov not found. Install it with: pip install pytest-cov")
        return 1


if __name__ == "__main__":
    if "--coverage" in sys.argv or "-c" in sys.argv:
        exit_code = run_tests_with_coverage()
    else:
        exit_code = run_tests()

    sys.exit(exit_code)

