#!/usr/bin/env python3
"""
Startup script for MuseTalk Realtime API Server
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from realtime_websocket_server import RealtimeWebSocketServer

def check_environment():
    """Check if all required environment variables and dependencies are set"""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    # Check for required directories
    required_dirs = [
        "./models/musetalkV15",
        "./models/whisper",
        "./models/sd-vae",
        "./data/video"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Error: Missing required directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        print("\nPlease ensure MuseTalk models are downloaded and properly set up.")
        return False
    
    # Check for required model files
    required_files = [
        "./models/musetalkV15/unet.pth",
        "./models/musetalkV15/musetalk.json",
        "./models/whisper/config.json",
        "./models/sd-vae/config.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Error: Missing required model files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease run the model download script first.")
        return False
    
    print("✅ Environment check passed!")
    return True

def setup_directories():
    """Create necessary directories for realtime processing"""
    dirs_to_create = [
        "./results/realtime",
        "./temp"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description="MuseTalk Realtime API Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8900, help="Port to bind to (default: 8900)")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment checks")
    
    args = parser.parse_args()
    
    print("🚀 Starting MuseTalk Realtime API Server...")
    print("=" * 50)
    
    # Check environment unless skipped
    if not args.skip_checks:
        if not check_environment():
            sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Start the server
    try:
        print(f"🌐 Starting WebSocket server on {args.host}:{args.port}")
        print("📱 Open realtime_frontend.html in your browser to test")
        print("🔑 Make sure your OpenAI API key is set in environment variables")
        print("=" * 50)
        
        server = RealtimeWebSocketServer(host=args.host, port=args.port)
        asyncio.run(server.start_server())
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()