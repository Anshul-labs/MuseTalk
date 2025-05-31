#!/bin/bash

echo "Starting MuseTalk Realtime API Server..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Install requirements if needed
echo "Installing/updating requirements..."
pip3 install -r requirements_realtime.txt

# Start the server
echo
echo "Starting WebSocket server..."
echo "Open realtime_frontend.html in your browser to test"
echo "Press Ctrl+C to stop the server"
echo

python3 start_realtime_server.py