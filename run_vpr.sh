#!/bin/bash
# Wrapper script to run Python with torch available

# Use the system Python that has torch installed
PYTHON_BIN="/usr/bin/python"

# Debug: Show what we're using
echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version
echo ""

# Run the Python script with all arguments passed through
exec "$PYTHON_BIN" "$@"

