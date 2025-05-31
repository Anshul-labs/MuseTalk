#!/usr/bin/env python3
"""
Startup script for MuseTalk WebRTC Server
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from webrtc_server import WebRTCMuseTalkServer

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
    """Create necessary directories for WebRTC processing"""
    dirs_to_create = [
        "./results/webrtc",
        "./temp",
        "./logs"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def check_webrtc_dependencies():
    """Check if WebRTC dependencies are installed"""
    try:
        import aiortc
        import av
        print("✅ WebRTC dependencies found")
        return True
    except ImportError as e:
        print("❌ Error: WebRTC dependencies not found")
        print("Please install WebRTC requirements:")
        print("pip install -r requirements_webrtc.txt")
        return False

def main():
    parser = argparse.ArgumentParser(description="MuseTalk WebRTC Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8901, help="Port to bind to (default: 8901)")
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment checks")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    print("🚀 Starting MuseTalk WebRTC Server...")
    print("=" * 60)
    print("🎯 WebRTC Advantages:")
    print("  • Ultra-low latency real-time streaming")
    print("  • Automatic bandwidth adaptation")
    print("  • Built-in audio-video synchronization")
    print("  • NAT/Firewall traversal support")
    print("=" * 60)
    
    # Set debug logging if requested
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print("🐛 Debug logging enabled")
    
    # Check environment unless skipped
    if not args.skip_checks:
        if not check_environment():
            sys.exit(1)
        
        if not check_webrtc_dependencies():
            sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Start the server
    try:
        print(f"🌐 Starting WebRTC server on {args.host}:{args.port}")
        print("📱 Open webrtc_frontend.html in your browser to test")
        print("🔑 Make sure your OpenAI API key is set in environment variables")
        print("🎭 WebRTC provides superior real-time performance compared to WebSocket")
        print("=" * 60)
        
        server = WebRTCMuseTalkServer(host=args.host, port=args.port)
        asyncio.run(server.start_server())
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()