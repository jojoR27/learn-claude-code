#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for hello.py module."""

import unittest
import subprocess
import sys


class TestHello(unittest.TestCase):
    """Test cases for hello.py module."""

    def test_hello_output(self):
        """Test that hello.py prints the correct output."""
        result = subprocess.run(
            [sys.executable, "hello.py"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.stdout.strip(), "Hello World!!")

    def test_hello_exit_code(self):
        """Test that hello.py exits with code 0."""
        result = subprocess.run(
            [sys.executable, "hello.py"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)

    def test_hello_no_stderr(self):
        """Test that hello.py produces no stderr output."""
        result = subprocess.run(
            [sys.executable, "hello.py"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
