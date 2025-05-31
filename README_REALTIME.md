# MuseTalk + OpenAI Realtime API Integration

This integration extends MuseTalk with OpenAI's Realtime API to create a real-time talking digital human with voice interaction capabilities.

## Features

- **Real-time Voice Interaction**: Talk to your digital human using OpenAI's Realtime API
- **Live Video Generation**: MuseTalk generates lip-synced video in real-time
- **WebSocket Communication**: Bidirectional communication between frontend, backend, and OpenAI
- **Audio Processing**: Automatic audio chunking and merging for optimal video generation
- **Streaming Video**: Real-time video streaming using fragmented MP4
- **Multiple Characters**: Support for different digital human characters
- **Voice Selection**: Choose from multiple OpenAI voices

## Architecture

```
Frontend (HTML/JS) ←→ WebSocket Server (Python) ←→ OpenAI Realtime API
                              ↓
                    MuseTalk Inference Process
                              ↓
                    Real-time Video Generation
```

## Prerequisites

1. **MuseTalk Setup**: Ensure MuseTalk is properly installed and models are downloaded
2. **OpenAI API Key**: Get your API key from OpenAI
3. **Python Dependencies**: Install additional requirements for realtime functionality
4. **FFmpeg**: Required for audio/video processing

## Installation

### 1. Install Additional Dependencies

```bash
pip install -r requirements_realtime.txt
```

### 2. Set Environment Variables

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# On Windows:
set OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Prepare Character Videos

Place your character videos in the `./data/video/` directory:
- `girl2.mp4`
- `man1.mp4` 
- `woman1.mp4`

## Usage

### 1. Start the Realtime Server

```bash
python start_realtime_server.py
```

Or with custom settings:
```bash
python start_realtime_server.py --host 0.0.0.0 --port 8900
```

### 2. Open the Frontend

Open `realtime_frontend.html` in your web browser.

### 3. Configure and Connect

1. Enter a license ID (any string for identification)
2. Select a character
3. Choose a voice
4. Click "Connect"

### 4. Start Talking

Once connected, the system will:
1. Initialize the session with OpenAI
2. Start recording your microphone
3. Process your speech in real-time
4. Generate AI responses with voice
5. Create lip-synced video of your digital human
6. Stream the video back to your browser

## File Structure

```
MuseTalk/
├── realtime_websocket_server.py    # Main WebSocket server
├── start_realtime_server.py        # Server startup script
├── realtime_frontend.html          # Web interface
├── requirements_realtime.txt       # Additional dependencies
├── scripts/
│   └── realtime_inference.py       # Modified inference script
├── configs/inference/
│   └── realtime_config.yaml        # Configuration file
└── results/realtime/               # Output directory
    └── {license}/
        ├── audio/                  # Audio chunks
        ├── vid_output/             # Generated videos
        ├── tmp/                    # Temporary files
        └── images/                 # Frame images
```

## Configuration

Edit `configs/inference/realtime_config.yaml` to customize:

- **Server settings**: Host, port, max connections
- **Audio processing**: Sample rate, chunk size, merge thresholds
- **Video settings**: FPS, batch size, parsing mode
- **Character definitions**: Video paths, descriptions, greetings
- **Model paths**: Locations of MuseTalk models

## API Reference

### WebSocket Endpoints

**Main endpoint**: `ws://localhost:8900/api/realtime-api`

**Query parameters**:
- `license`: Unique identifier for the session
- `characterName`: Character to use (girl2, man1, woman1)

### Message Types

#### From Client to Server
- Session configuration messages (forwarded to OpenAI)
- Audio data chunks
- Control messages

#### From Server to Client
- OpenAI response messages
- Binary video data (fragmented MP4)
- Status updates

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not set"**
   - Ensure your OpenAI API key is properly set in environment variables

2. **"Missing model files"**
   - Run the MuseTalk model download script first
   - Verify all required models are in the correct directories

3. **"Unable to access microphone"**
   - Grant microphone permissions in your browser
   - Check browser security settings

4. **"WebSocket connection failed"**
   - Ensure the server is running
   - Check firewall settings
   - Verify the correct port is being used

5. **"No video output"**
   - Check that character video files exist in `./data/video/`
   - Verify FFmpeg is properly installed
   - Check server logs for processing errors

### Performance Optimization

1. **GPU Usage**: Ensure CUDA is available for faster processing
2. **Batch Size**: Adjust batch_size in config for your hardware
3. **Audio Chunking**: Tune audio merge thresholds for responsiveness
4. **Video Buffer**: Adjust max_video_buffer to prevent memory issues

## Development

### Adding New Characters

1. Add character video to `./data/video/`
2. Update `realtime_config.yaml` with character settings
3. Add character option to frontend dropdown

### Customizing Audio Processing

Modify the audio processing logic in `scripts/realtime_inference.py`:
- Adjust merge thresholds
- Change audio quality settings
- Implement custom audio filters

### Extending WebSocket Messages

Add new message types in `realtime_websocket_server.py`:
- Custom control messages
- Additional metadata
- Status notifications

## Security Considerations

1. **API Key Protection**: Never expose your OpenAI API key in client-side code
2. **Input Validation**: Validate all WebSocket messages
3. **Rate Limiting**: Implement rate limiting for production use
4. **File Cleanup**: Regularly clean up temporary files
5. **Access Control**: Add authentication for production deployment

## Performance Metrics

- **Latency**: Typical end-to-end latency is 2-5 seconds
- **Throughput**: Can handle 1-5 concurrent sessions depending on hardware
- **Memory Usage**: ~2-4GB GPU memory per session
- **CPU Usage**: Moderate CPU usage for audio processing

## License

This integration follows the same license as the original MuseTalk project.

## Support

For issues specific to this realtime integration:
1. Check the troubleshooting section above
2. Review server logs in `./logs/realtime.log`
3. Verify your OpenAI API quota and usage

For general MuseTalk issues, refer to the main MuseTalk documentation.