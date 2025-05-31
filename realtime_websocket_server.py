import asyncio
import websockets
import json
import base64
import os
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse
import wave
import numpy as np
from pathlib import Path
import logging
import aiohttp
import ssl
import io
from collections import deque
import concurrent.futures
from threading import Lock
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeWebSocketServer:
    def __init__(self, host="localhost", port=8900):
        self.host = host
        self.port = port
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.active_sessions = {}
        self.openai_ws_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        
        # Memory-based audio/video processing
        self.audio_buffer_lock = Lock()
        self.video_buffer_lock = Lock()
        
        # Thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Audio block standardization (2 seconds target)
        self.target_audio_duration = 2.0
        self.audio_sample_rate = 24000
        self.target_audio_samples = int(self.target_audio_duration * self.audio_sample_rate)
        
    async def handle_client(self, websocket, path):
        """Handle incoming WebSocket connections from clients"""
        try:
            # Parse query parameters
            parsed_url = urlparse(path)
            query_params = parse_qs(parsed_url.query)
            
            license = query_params.get('license', [None])[0]
            character_name = query_params.get('characterName', [None])[0]
            
            if not license:
                await websocket.close(code=1000, reason="Missing license parameter")
                return
                
            if not character_name:
                await websocket.close(code=1000, reason="Missing characterName parameter")
                return
            
            logger.info(f"Client connected: license={license}, character={character_name}")
            
            # Create audio directory
            audio_path = f"./results/realtime/{license}/audio"
            os.makedirs(audio_path, exist_ok=True)
            
            # Store session info with memory buffers
            session_info = {
                'license': license,
                'character_name': character_name,
                'audio_path': audio_path,
                'websocket': websocket,
                'openai_ws': None,
                'python_process': None,
                'is_running': True,
                'audio_buffer': deque(),
                'video_buffer': deque(),
                'audio_accumulator': bytearray(),
                'last_audio_time': time.time(),
                'video_generation_queue': queue.Queue(),
                'processed_audio_count': 0
            }
            
            self.active_sessions[websocket] = session_info
            
            # Connect to OpenAI Realtime API
            await self.connect_to_openai(session_info)
            
        except Exception as e:
            logger.error(f"Error handling client connection: {e}")
            await websocket.close(code=1011, reason="Internal server error")
    
    async def connect_to_openai(self, session_info):
        """Connect to OpenAI Realtime API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            # Create SSL context
            ssl_context = ssl.create_default_context()
            
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    self.openai_ws_url,
                    headers=headers,
                    ssl=ssl_context
                ) as openai_ws:
                    session_info['openai_ws'] = openai_ws
                    logger.info("Connected to OpenAI Realtime API")
                    
                    # Start MuseTalk process
                    await self.start_musetalk_process(session_info)
                    
                    # Start video monitoring
                    asyncio.create_task(self.monitor_video_output(session_info))
                    
                    # Handle bidirectional communication
                    await asyncio.gather(
                        self.handle_client_to_openai(session_info),
                        self.handle_openai_to_client(session_info)
                    )
                    
        except Exception as e:
            logger.error(f"Error connecting to OpenAI: {e}")
            if session_info['websocket'].open:
                await session_info['websocket'].close(code=1011, reason="OpenAI connection failed")
    
    async def handle_client_to_openai(self, session_info):
        """Forward messages from client to OpenAI"""
        try:
            websocket = session_info['websocket']
            openai_ws = session_info['openai_ws']
            
            async for message in websocket:
                if message.type == aiohttp.WSMsgType.TEXT:
                    # Forward text message to OpenAI
                    await openai_ws.send_str(message.data)
                elif message.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {websocket.exception()}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in client to OpenAI communication: {e}")
    
    async def handle_openai_to_client(self, session_info):
        """Forward messages from OpenAI to client and handle audio"""
        try:
            websocket = session_info['websocket']
            openai_ws = session_info['openai_ws']
            audio_path = session_info['audio_path']
            license = session_info['license']
            
            async for message in openai_ws:
                if message.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(message.data)
                    
                    # Handle audio delta messages
                    if data.get('type') == 'response.audio.delta':
                        delta = data.get('delta', '')
                        if delta:
                            # Decode and process audio with 2-second standardization
                            audio_data = base64.b64decode(delta)
                            await self.save_audio_chunk(audio_data, session_info)
                    
                    # Forward all messages to client
                    if websocket.open:
                        await websocket.send(message.data)
                        
                elif message.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"OpenAI WebSocket error: {openai_ws.exception()}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in OpenAI to client communication: {e}")
    
    async def save_audio_chunk(self, audio_data, session_info):
        """Process audio chunk with 2-second standardization and memory buffering"""
        try:
            license = session_info['license']
            
            with self.audio_buffer_lock:
                # Add audio data to accumulator
                session_info['audio_accumulator'].extend(audio_data)
                
                # Calculate current duration in samples (assuming 16-bit PCM)
                current_samples = len(session_info['audio_accumulator']) // 2
                current_duration = current_samples / self.audio_sample_rate
                
                # Check if we have enough audio for 2-second block
                if current_duration >= self.target_audio_duration:
                    # Extract exactly 2 seconds of audio
                    target_bytes = self.target_audio_samples * 2  # 16-bit = 2 bytes per sample
                    audio_block = bytes(session_info['audio_accumulator'][:target_bytes])
                    
                    # Remove processed audio from accumulator
                    session_info['audio_accumulator'] = session_info['audio_accumulator'][target_bytes:]
                    
                    # Process audio block asynchronously
                    await self.process_audio_block(audio_block, session_info)
                    
                    logger.debug(f"Processed 2-second audio block: {len(audio_block)} bytes")
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    async def process_audio_block(self, audio_block, session_info):
        """Process standardized 2-second audio block for video generation"""
        try:
            license = session_info['license']
            audio_count = session_info['processed_audio_count']
            session_info['processed_audio_count'] += 1
            
            # Create audio filename
            audio_filename = f"{license}_{audio_count:08d}.wav"
            
            # Submit to thread pool for parallel processing
            future = self.executor.submit(
                self.generate_video_from_audio,
                audio_block,
                audio_filename,
                session_info
            )
            
            # Store future for tracking
            session_info['video_generation_queue'].put(future)
            
        except Exception as e:
            logger.error(f"Error processing audio block: {e}")
    
    def generate_video_from_audio(self, audio_block, audio_filename, session_info):
        """Generate video from audio block in separate thread"""
        try:
            license = session_info['license']
            
            # Create temporary audio file in memory
            audio_io = io.BytesIO()
            
            # Write WAV header and audio data
            self.write_wav_to_memory(audio_io, audio_block)
            
            # Reset position for reading
            audio_io.seek(0)
            
            # Send audio data directly to MuseTalk process via memory
            self.send_audio_to_musetalk(audio_io.getvalue(), audio_filename, session_info)
            
            logger.debug(f"Generated video for audio block: {audio_filename}")
            
        except Exception as e:
            logger.error(f"Error generating video from audio: {e}")
    
    def write_wav_to_memory(self, audio_io, audio_data):
        """Write WAV format to memory buffer"""
        import struct
        
        # WAV header parameters
        sample_rate = self.audio_sample_rate
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(audio_data)
        
        # Write WAV header
        audio_io.write(b'RIFF')
        audio_io.write(struct.pack('<I', 36 + data_size))
        audio_io.write(b'WAVE')
        audio_io.write(b'fmt ')
        audio_io.write(struct.pack('<I', 16))
        audio_io.write(struct.pack('<H', 1))  # PCM format
        audio_io.write(struct.pack('<H', num_channels))
        audio_io.write(struct.pack('<I', sample_rate))
        audio_io.write(struct.pack('<I', byte_rate))
        audio_io.write(struct.pack('<H', block_align))
        audio_io.write(struct.pack('<H', bits_per_sample))
        audio_io.write(b'data')
        audio_io.write(struct.pack('<I', data_size))
        audio_io.write(audio_data)
    
    def send_audio_to_musetalk(self, wav_data, audio_filename, session_info):
        """Send audio data to MuseTalk process via temporary file (optimized)"""
        try:
            license = session_info['license']
            audio_path = session_info['audio_path']
            
            # Write to temporary file for MuseTalk processing
            temp_audio_file = os.path.join(audio_path, audio_filename)
            with open(temp_audio_file, 'wb') as f:
                f.write(wav_data)
            
            logger.debug(f"Created audio file for processing: {temp_audio_file}")
            
        except Exception as e:
            logger.error(f"Error sending audio to MuseTalk: {e}")
    
    async def start_musetalk_process(self, session_info):
        """Start MuseTalk process for real-time inference"""
        try:
            license = session_info['license']
            character_name = session_info['character_name']
            audio_path = session_info['audio_path']
            
            # Prepare video path and output directory
            video_name = character_name
            mp4_path = video_name
            project_root_dir = os.getcwd()
            output_dir = f"./results/realtime/{license}/vid_output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Start MuseTalk process
            command = [
                "python", "-u", "-m", "scripts.realtime_inference",
                "--avatar_id", license,
                "--preparation", "true",
                "--bbox_shift", "-7",
                "--fps", "25",
                "--batch_size", "6",
                "--audio_folder", audio_path,
                "--mp4_path", mp4_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=project_root_dir
            )
            
            session_info['python_process'] = process
            logger.info(f"Started MuseTalk process for {license}")
            
            # Monitor process output
            asyncio.create_task(self.monitor_process_output(process, license))
            
        except Exception as e:
            logger.error(f"Error starting MuseTalk process: {e}")
    
    async def monitor_process_output(self, process, license):
        """Monitor MuseTalk process output"""
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                logger.info(f"[MuseTalk-{license}] {line.decode().strip()}")
        except Exception as e:
            logger.error(f"Error monitoring process output: {e}")
    
    async def monitor_video_output(self, session_info):
        """Monitor video output directory and send videos to client with optimized streaming"""
        try:
            license = session_info['license']
            output_dir = f"./results/realtime/{license}/vid_output"
            websocket = session_info['websocket']
            
            # Video buffer for ahead-of-time generation
            video_send_queue = asyncio.Queue()
            
            # Start video sender task
            asyncio.create_task(self.video_sender_task(video_send_queue, websocket, session_info))
            
            while session_info['is_running']:
                try:
                    # Check for new video files
                    video_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
                    video_files.sort()
                    
                    for video_file in video_files:
                        video_path = os.path.join(output_dir, video_file)
                        
                        # Process video in thread pool for parallel processing
                        future = self.executor.submit(
                            self.process_video_file,
                            video_path,
                            video_file
                        )
                        
                        # Add to send queue
                        await video_send_queue.put(future)
                        
                        # Delete processed video immediately to free space
                        os.remove(video_path)
                        logger.debug(f"Queued video file: {video_file}")
                
                except FileNotFoundError:
                    pass  # Directory doesn't exist yet
                except Exception as e:
                    logger.error(f"Error monitoring video output: {e}")
                
                await asyncio.sleep(0.3)  # Check every 300ms for faster response
                
        except Exception as e:
            logger.error(f"Error in video monitoring: {e}")
    
    async def video_sender_task(self, video_queue, websocket, session_info):
        """Send videos to client with timing optimization"""
        try:
            video_send_interval = 1.8  # Send slightly faster than 2-second audio blocks
            last_send_time = time.time()
            
            while session_info['is_running']:
                try:
                    # Wait for video with timeout
                    future = await asyncio.wait_for(video_queue.get(), timeout=1.0)
                    
                    # Get processed video data
                    fmp4_data = future.result()
                    
                    if fmp4_data and websocket.open:
                        # Control sending rate to maintain sync
                        current_time = time.time()
                        time_since_last = current_time - last_send_time
                        
                        if time_since_last < video_send_interval:
                            await asyncio.sleep(video_send_interval - time_since_last)
                        
                        # Send video data
                        await websocket.send(fmp4_data)
                        last_send_time = time.time()
                        logger.info(f"Sent video segment at optimized timing")
                        
                except asyncio.TimeoutError:
                    continue  # No video available, continue waiting
                except Exception as e:
                    logger.error(f"Error in video sender: {e}")
                    
        except Exception as e:
            logger.error(f"Error in video sender task: {e}")
    
    def process_video_file(self, video_path, video_file):
        """Process video file in separate thread"""
        try:
            # Read video file into memory
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            # Convert to fragmented MP4 in memory
            fmp4_data = self.convert_to_fragmented_mp4_sync(video_data)
            
            logger.debug(f"Processed video file: {video_file}")
            return fmp4_data
            
        except Exception as e:
            logger.error(f"Error processing video file: {e}")
            return None
    
    def convert_to_fragmented_mp4_sync(self, video_data):
        """Convert video to fragmented MP4 format synchronously"""
        try:
            import tempfile
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as input_file:
                input_file.write(video_data)
                input_path = input_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as output_file:
                output_path = output_file.name
            
            try:
                # Build ffmpeg command for faster processing
                command = [
                    "ffmpeg", "-y", "-v", "quiet",
                    "-i", input_path,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",  # Fastest encoding
                    "-tune", "zerolatency",  # Low latency
                    "-profile:v", "baseline",
                    "-level:v", "3.0",
                    "-movflags", "frag_keyframe+empty_moov+default_base_moof",
                    "-avoid_negative_ts", "make_zero",
                    "-f", "mp4",
                    output_path
                ]
                
                # Run ffmpeg
                result = subprocess.run(command, capture_output=True)
                
                if result.returncode == 0:
                    # Read the fragmented MP4 data
                    with open(output_path, 'rb') as f:
                        fmp4_data = f.read()
                    return fmp4_data
                else:
                    logger.error(f"ffmpeg conversion failed: {result.stderr.decode()}")
                    return None
                    
            finally:
                # Clean up temporary files
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error converting to fragmented MP4: {e}")
            return None
    
    async def convert_to_fragmented_mp4(self, video_path):
        """Convert video to fragmented MP4 format"""
        try:
            temp_dir = "./temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_fmp4_file = os.path.join(temp_dir, f"fragmented_{int(time.time() * 1000)}.mp4")
            
            # Get video duration
            duration = await self.get_video_duration(video_path)
            
            if duration <= 0:
                logger.warning("Video duration is too short, skipping")
                return None
            
            # Build ffmpeg command
            command = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-t", str(duration),
                "-c:v", "libx264",
                "-profile:v", "high",
                "-level:v", "4.0",
                "-c:a", "aac",
                "-movflags", "frag_keyframe+empty_moov+default_base_moof",
                "-avoid_negative_ts", "make_zero",
                temp_fmp4_file
            ]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await process.wait()
            
            if process.returncode != 0:
                logger.error(f"ffmpeg conversion failed with code {process.returncode}")
                return None
            
            # Read the fragmented MP4 data
            with open(temp_fmp4_file, 'rb') as f:
                fmp4_data = f.read()
            
            # Clean up temporary file
            os.remove(temp_fmp4_file)
            
            return fmp4_data
            
        except Exception as e:
            logger.error(f"Error converting to fragmented MP4: {e}")
            return None
    
    async def get_video_duration(self, video_path):
        """Get video duration using ffprobe"""
        try:
            command = [
                "ffprobe", "-i", video_path,
                "-show_entries", "format=duration",
                "-v", "quiet", "-of", "csv=p=0"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, _ = await process.communicate()
            
            if process.returncode == 0:
                return float(stdout.decode().strip())
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting video duration: {e}")
            return 0.0
    
    async def cleanup_session(self, websocket):
        """Clean up session resources"""
        try:
            if websocket in self.active_sessions:
                session_info = self.active_sessions[websocket]
                session_info['is_running'] = False
                
                # Terminate Python process
                if session_info.get('python_process'):
                    session_info['python_process'].terminate()
                    await session_info['python_process'].wait()
                
                # Close OpenAI WebSocket
                if session_info.get('openai_ws'):
                    await session_info['openai_ws'].close()
                
                del self.active_sessions[websocket]
                logger.info(f"Cleaned up session for {session_info['license']}")
                
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async def handle_connection(websocket, path):
            try:
                await self.handle_client(websocket, path)
            finally:
                await self.cleanup_session(websocket)
        
        server = await websockets.serve(
            handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        await server.wait_closed()

async def main():
    server = RealtimeWebSocketServer()
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main())