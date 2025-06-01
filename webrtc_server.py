#!/usr/bin/env python3
"""
WebRTC-based MuseTalk Realtime Server
Replaces WebSocket with WebRTC for optimized audio-video streaming
"""

import asyncio
import json
import logging
import os
import time
import fractions
import numpy as np
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
from threading import Lock
import queue

# WebRTC imports
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc import VideoStreamTrack, AudioStreamTrack
from av import VideoFrame, AudioFrame
import cv2

# WebSocket for signaling
import websockets
from urllib.parse import parse_qs, urlparse

# MuseTalk integration
from scripts.webrtc_inference import get_inference_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleFrameVideoStreamTrack(VideoStreamTrack):
    """
    Custom video track for streaming MuseTalk generated frames
    """
    def __init__(self):
        super().__init__()
        self._current_frame = None
        self._timestamp = 0
        self._time_base = fractions.Fraction(1, 30)  # 30 FPS
        self._lock = asyncio.Lock()
        
    async def update_frame(self, frame):
        """Update the current frame to be streamed"""
        async with self._lock:
            if isinstance(frame, np.ndarray):
                # Convert numpy array to VideoFrame
                self._current_frame = VideoFrame.from_ndarray(frame, format='bgr24')
            else:
                self._current_frame = frame
    
    async def recv(self):
        """Receive method called by WebRTC to get the next frame"""
        async with self._lock:
            if isinstance(self._current_frame, VideoFrame):
                frame = self._current_frame
            elif self._current_frame is not None:
                # Convert numpy array to VideoFrame
                frame = VideoFrame.from_ndarray(self._current_frame, format='bgr24')
            else:
                # Create a black frame if no frame is available
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame = VideoFrame.from_ndarray(black_frame, format='bgr24')
            
            # Set timestamp for the frame
            frame.pts = self._timestamp
            frame.time_base = self._time_base
            
            # Increment timestamp for next frame (30fps = 33.33ms per frame)
            self._timestamp += 3300
            
            return frame

class SingleFrameAudioStreamTrack(AudioStreamTrack):
    """
    Custom audio track for streaming synchronized audio
    """
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_queue = deque()
        self._timestamp = 0
        self._lock = asyncio.Lock()
        
    def push_audio_data(self, audio_data):
        """Push audio data to the queue"""
        self.audio_queue.append(audio_data)
        
    async def recv(self):
        """Receive method called by WebRTC to get the next audio frame"""
        # Wait for audio data to be available
        while not self.audio_queue:
            await asyncio.sleep(0.005)  # 5ms sleep
        
        async with self._lock:
            audio_data = self.audio_queue.popleft()
            
            # Ensure audio_data is 2D (samples, channels)
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)
            
            samples = audio_data.shape[0]
            
            # Create audio frame
            frame = AudioFrame(format="s16", layout="mono", samples=samples)
            frame.sample_rate = self.sample_rate
            frame.time_base = fractions.Fraction(1, self.sample_rate)
            
            # Convert to bytes and update frame
            frame.planes[0].update(audio_data.tobytes())
            frame.pts = self._timestamp
            self._timestamp += samples
            
            return frame

class WebRTCMuseTalkServer:
    """
    WebRTC-based MuseTalk server for real-time audio-video streaming
    """
    def __init__(self, host="localhost", port=8901):
        self.host = host
        self.port = port
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # WebRTC configuration with STUN/TURN servers
        self.ice_servers = [
            RTCIceServer(urls="stun:stun.l.google.com:19302"),
            RTCIceServer(
                urls="turn:freestun.net:3478",
                username="free",
                credential="free"
            )
        ]
        self.rtc_configuration = RTCConfiguration(iceServers=self.ice_servers)
        
        # Active sessions
        self.active_sessions = {}
        
        # Global frame and audio maps for synchronized playback
        self.global_frame_map = {}
        self.global_audio_frame_map = {}
        self.segment_counter = 0
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Memory optimization
        self.memory_lock = Lock()
        
    async def handle_signaling(self, websocket, path=None):
        """Handle WebSocket signaling for WebRTC connection establishment"""
        try:
            # Parse query parameters
            license = None
            character_name = None
            
            if path:
                parsed_url = urlparse(path)
                query_params = parse_qs(parsed_url.query)
                license = query_params.get('license', [None])[0]
                character_name = query_params.get('characterName', [None])[0]
            
            # If parameters are not in path, try to get them from websocket.path
            if (not license or not character_name) and hasattr(websocket, 'path'):
                parsed_url = urlparse(websocket.path)
                query_params = parse_qs(parsed_url.query)
                license = license or query_params.get('license', [None])[0]
                character_name = character_name or query_params.get('characterName', [None])[0]
            
            if not license or not character_name:
                await websocket.close(code=1000, reason="Missing parameters")
                return
            
            logger.info(f"WebRTC signaling started: license={license}, character={character_name}")
            
            # Create WebRTC peer connection
            pc = RTCPeerConnection(self.rtc_configuration)
            
            # Create media tracks
            video_track = SingleFrameVideoStreamTrack()
            audio_track = SingleFrameAudioStreamTrack()
            
            # Add tracks to peer connection
            pc.addTrack(video_track)
            pc.addTrack(audio_track)
            
            # Store session info
            session_info = {
                'license': license,
                'character_name': character_name,
                'pc': pc,
                'video_track': video_track,
                'audio_track': audio_track,
                'websocket': websocket,
                'is_running': True
            }
            
            self.active_sessions[websocket] = session_info
            
            # Start MuseTalk processing
            await self.start_musetalk_processing(session_info)
            
            # Handle signaling messages
            async for message in websocket:
                data = json.loads(message)
                await self.handle_signaling_message(data, session_info)
                
        except Exception as e:
            logger.error(f"Error in signaling: {e}")
        finally:
            await self.cleanup_session(websocket)
    
    async def handle_signaling_message(self, data, session_info):
        """Handle WebRTC signaling messages"""
        try:
            pc = session_info['pc']
            websocket = session_info['websocket']
            
            if data['type'] == 'offer':
                # Handle WebRTC offer
                offer = RTCSessionDescription(sdp=data['sdp'], type=data['type'])
                await pc.setRemoteDescription(offer)
                
                # Create answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                
                # Send answer back
                await websocket.send(json.dumps({
                    'type': 'answer',
                    'sdp': pc.localDescription.sdp
                }))
                
            elif data['type'] == 'ice-candidate':
                # Handle ICE candidate
                candidate = data['candidate']
                await pc.addIceCandidate(candidate)
                
            elif data['type'] == 'audio-data':
                # Handle audio data from OpenAI Realtime API
                audio_data = data['audio']
                await self.process_audio_data(audio_data, session_info)
                
        except Exception as e:
            logger.error(f"Error handling signaling message: {e}")
    
    async def process_audio_data(self, audio_data, session_info):
        """Process audio data and generate video"""
        try:
            license = session_info['license']
            
            # Decode base64 audio data
            import base64
            audio_bytes = base64.b64decode(audio_data)
            
            # Convert to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Store in global audio map
            segment_index = self.segment_counter
            self.global_audio_frame_map[segment_index] = audio_array
            self.segment_counter += 1
            
            # Submit for video generation
            future = self.executor.submit(
                self.generate_video_segment,
                audio_array,
                segment_index,
                session_info
            )
            
            # Don't wait for completion - process asynchronously
            
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
    
    def generate_video_segment(self, audio_array, segment_index, session_info):
        """Generate video segment from audio using MuseTalk (runs in thread pool)"""
        try:
            # Get MuseTalk inference engine
            inference_engine = get_inference_engine()
            
            # Process audio to video using MuseTalk
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                video_frames, processed_audio = loop.run_until_complete(
                    inference_engine.process_audio_to_video(audio_array, segment_index)
                )
                
                # Store frames and audio in global map
                self.global_frame_map[segment_index] = video_frames
                self.global_audio_frame_map[segment_index] = processed_audio
                
                # Start synchronized playback
                asyncio.run_coroutine_threadsafe(
                    self.push_av_segment(segment_index, session_info),
                    loop
                )
                
            finally:
                loop.close()
            
        except Exception as e:
            logger.error(f"Error generating video segment: {e}")
    
    async def push_av_segment(self, segment_index, session_info):
        """Synchronously push audio and video segment"""
        try:
            if segment_index not in self.global_frame_map or segment_index not in self.global_audio_frame_map:
                return
            
            frames = self.global_frame_map[segment_index]
            waveform = self.global_audio_frame_map[segment_index]
            
            video_track = session_info['video_track']
            audio_track = session_info['audio_track']
            
            sample_rate = 24000
            fps = 30
            
            # Calculate audio duration
            audio_duration = len(waveform) / sample_rate
            video_frame_count = min(len(frames), int(audio_duration * fps))
            
            # Audio chunk size (20ms)
            chunk_samples = int(0.02 * sample_rate)
            audio_pos = 0
            
            frame_duration = 1 / fps
            start_time = time.time()
            
            # Synchronized audio-video pushing
            for frame_idx in range(video_frame_count):
                # Update video frame
                await video_track.update_frame(frames[frame_idx])
                
                # Calculate expected audio position
                expected_audio_pos = int(frame_idx * frame_duration * sample_rate)
                
                # Push corresponding audio chunks
                while audio_pos < expected_audio_pos and audio_pos < len(waveform):
                    chunk_end = min(audio_pos + chunk_samples, len(waveform))
                    chunk = waveform[audio_pos:chunk_end]
                    
                    # Pad chunk if necessary
                    if len(chunk) < chunk_samples:
                        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                    
                    # Convert to int16 and push to audio track
                    audio_data = (chunk * 32767).astype(np.int16).reshape(-1, 1)
                    audio_track.push_audio_data(audio_data)
                    
                    audio_pos = chunk_end
                    await asyncio.sleep(0.02)  # 20ms audio frame
                
                # Control video frame rate
                elapsed = time.time() - start_time
                expected_time = (frame_idx + 1) * frame_duration
                if elapsed < expected_time:
                    await asyncio.sleep(expected_time - elapsed)
            
            # Clean up memory
            with self.memory_lock:
                if segment_index in self.global_frame_map:
                    del self.global_frame_map[segment_index]
                if segment_index in self.global_audio_frame_map:
                    del self.global_audio_frame_map[segment_index]
            
            logger.info(f"✅ Segment {segment_index} pushed successfully")
            
        except Exception as e:
            logger.error(f"❌ Segment {segment_index} push failed: {str(e)}")
    
    async def start_musetalk_processing(self, session_info):
        """Start MuseTalk processing for the session"""
        try:
            license = session_info['license']
            character_name = session_info['character_name']
            
            # Create directories
            audio_path = f"./results/webrtc/{license}/audio"
            os.makedirs(audio_path, exist_ok=True)
            
            logger.info(f"Started MuseTalk processing for {license}")
            
            # Here you would integrate with your existing MuseTalk pipeline
            # For now, we'll just log the start
            
        except Exception as e:
            logger.error(f"Error starting MuseTalk processing: {e}")
    
    async def cleanup_session(self, websocket):
        """Clean up session resources"""
        try:
            if websocket in self.active_sessions:
                session_info = self.active_sessions[websocket]
                session_info['is_running'] = False
                
                # Close peer connection
                if session_info.get('pc'):
                    await session_info['pc'].close()
                
                del self.active_sessions[websocket]
                logger.info(f"Cleaned up session for {session_info['license']}")
                
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
    
    async def start_server(self):
        """Start the WebRTC signaling server"""
        logger.info(f"Starting WebRTC server on {self.host}:{self.port}")
        
        async def handle_connection(websocket, path=None):
            try:
                await self.handle_signaling(websocket, path)
            finally:
                await self.cleanup_session(websocket)
        
        server = await websockets.serve(
            handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"WebRTC signaling server started on ws://{self.host}:{self.port}")
        await server.wait_closed()

async def main():
    server = WebRTCMuseTalkServer()
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main())
