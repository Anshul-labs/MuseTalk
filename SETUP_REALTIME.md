# Quick Setup Guide for MuseTalk Realtime API

## Step 1: Prerequisites

1. **Ensure MuseTalk is working**: Make sure the basic MuseTalk setup is complete and models are downloaded
2. **Get OpenAI API Key**: Sign up at https://platform.openai.com and get your API key
3. **Install FFmpeg**: Ensure FFmpeg is installed and accessible

## Step 2: Install Dependencies

```bash
pip install -r requirements_realtime.txt
```

## Step 3: Set Environment Variables

### Windows:
```cmd
set OPENAI_API_KEY=your-actual-api-key-here
```

### Linux/Mac:
```bash
export OPENAI_API_KEY="your-actual-api-key-here"
```

## Step 4: Prepare Character Videos

Place your character videos in `./data/video/`:
- `girl2.mp4` (or any character name you prefer)
- `man1.mp4`
- `woman1.mp4`

## Step 5: Start the Server

### Windows:
```cmd
start_realtime.bat
```
or
```cmd
python start_realtime_server.py
```

### Linux/Mac:
```bash
./start_realtime.sh
```
or
```bash
python3 start_realtime_server.py
```

## Step 6: Open Frontend

1. Open `realtime_frontend.html` in your web browser
2. Enter a license ID (any string)
3. Select a character
4. Choose a voice
5. Click "Connect"
6. Allow microphone access when prompted
7. Start talking!

## Troubleshooting

### Common Issues:

1. **"OPENAI_API_KEY not set"**
   - Make sure you've set the environment variable correctly
   - Restart your terminal/command prompt after setting it

2. **"Missing model files"**
   - Run the MuseTalk model download script first
   - Check that all models are in the correct directories

3. **"WebSocket connection failed"**
   - Make sure the server is running
   - Check that port 8900 is not blocked by firewall

4. **"No audio/video"**
   - Grant microphone permissions in your browser
   - Check that character videos exist in `./data/video/`

### Performance Tips:

- Use a GPU for faster processing
- Adjust batch_size in config for your hardware
- Close other applications to free up memory

## File Structure After Setup:

```
MuseTalk/
├── realtime_websocket_server.py    # ✅ WebSocket server
├── start_realtime_server.py        # ✅ Server startup
├── realtime_frontend.html          # ✅ Web interface
├── start_realtime.bat              # ✅ Windows launcher
├── start_realtime.sh               # ✅ Linux/Mac launcher
├── requirements_realtime.txt       # ✅ Dependencies
├── README_REALTIME.md              # ✅ Full documentation
├── SETUP_REALTIME.md               # ✅ This guide
├── example_env.txt                 # ✅ Environment example
├── scripts/
│   └── realtime_inference.py       # ✅ Modified inference
├── configs/inference/
│   └── realtime_config.yaml        # ✅ Configuration
└── results/realtime/               # 📁 Will be created
```

## Next Steps

1. **Test the basic setup** with the provided frontend
2. **Customize characters** by adding your own videos
3. **Modify the configuration** in `realtime_config.yaml`
4. **Integrate with your application** using the WebSocket API

For detailed documentation, see `README_REALTIME.md`.

## Support

If you encounter issues:
1. Check the server logs
2. Verify your OpenAI API key and quota
3. Ensure all dependencies are installed
4. Check that MuseTalk works independently first

Happy talking with your digital human! 🎭🤖