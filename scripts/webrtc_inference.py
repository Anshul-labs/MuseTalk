#!/usr/bin/env python3
"""
WebRTC-based MuseTalk Real-time Inference
Integrates MuseTalk with WebRTC for optimized streaming
"""

import argparse
import os
import sys
import asyncio
import numpy as np
import cv2
import torch
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import logging
import json
import glob
from pathlib import Path

# MuseTalk imports
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebRTCMuseTalkInference:
    """
    WebRTC-optimized MuseTalk inference engine
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.load_models()
        
        # Initialize avatar
        self.setup_avatar()
        
        # WebRTC streaming components
        self.video_frame_queue = asyncio.Queue(maxsize=30)
        self.audio_frame_queue = asyncio.Queue(maxsize=100)
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def load_models(self):
        """Load MuseTalk models with optimizations"""
        logger.info("Loading MuseTalk models...")
        
        # Load core models
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=self.args.unet_model_path,
            vae_type=self.args.vae_type,
            unet_config=self.args.unet_config,
            device=self.device
        )
        
        # Optimize for inference
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        
        # Set to eval mode
        self.pe.eval()
        self.vae.vae.eval()
        self.unet.model.eval()
        
        self.timesteps = torch.tensor([0], device=self.device)
        
        # Initialize audio processor and Whisper
        self.audio_processor = AudioProcessor(feature_extractor_path=self.args.whisper_dir)
        self.weight_dtype = self.unet.model.dtype
        self.whisper = WhisperModel.from_pretrained(self.args.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)
        
        # Initialize face parser
        self.fp = FaceParsing(
            left_cheek_width=self.args.left_cheek_width,
            right_cheek_width=self.args.right_cheek_width
        )
        
        logger.info("Models loaded successfully")
    
    def setup_avatar(self):
        """Setup avatar data for inference"""
        logger.info("Setting up avatar...")
        
        # Create avatar directories
        self.avatar_path = f"./results/webrtc/{self.args.avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        
        os.makedirs(self.avatar_path, exist_ok=True)
        os.makedirs(self.full_imgs_path, exist_ok=True)
        os.makedirs(self.mask_out_path, exist_ok=True)
        
        # Load or prepare avatar data
        if self.args.preparation:
            self.prepare_avatar_data()
        else:
            self.load_avatar_data()
        
        logger.info("Avatar setup complete")
    
    def prepare_avatar_data(self):
        """Prepare avatar data from video"""
        logger.info("Preparing avatar data...")
        
        video_path = f"./data/video/{self.args.mp4_path}.mp4"
        
        # Extract frames from video
        if os.path.isfile(video_path):
            self.video2imgs(video_path, self.full_imgs_path)
        
        # Get image list
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        
        # Extract landmarks and prepare data
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.args.bbox_shift)
        
        # Prepare latents
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == (0.0, 0.0, 0.0, 0.0):  # coord_placeholder
                continue
            x1, y1, x2, y2 = bbox
            y2 = y2 + self.args.extra_margin
            y2 = min(y2, frame.shape[0])
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)
        
        # Create cycles for smooth looping
        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        
        # Prepare masks
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []
        
        for i, frame in enumerate(self.frame_list_cycle):
            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=self.fp, mode=self.args.parsing_mode)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)
        
        # Save prepared data
        import pickle
        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        torch.save(self.input_latent_list_cycle, self.latents_out_path)
        
        logger.info("Avatar data preparation complete")
    
    def load_avatar_data(self):
        """Load pre-prepared avatar data"""
        logger.info("Loading avatar data...")
        
        import pickle
        import glob
        
        # Load latents
        self.input_latent_list_cycle = torch.load(self.latents_out_path)
        
        # Load coordinates
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        
        # Load frames
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        
        # Load mask data
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)
        
        logger.info("Avatar data loaded successfully")
    
    def video2imgs(self, vid_path, save_path, ext='.png'):
        """Extract frames from video"""
        cap = cv2.VideoCapture(vid_path)
        count = 0
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
                count += 1
            else:
                break
        cap.release()
    
    async def process_audio_to_video(self, audio_data, segment_id):
        """
        Process audio data and generate corresponding video frames
        Optimized for WebRTC streaming
        """
        try:
            start_time = time.time()
            
            # Convert audio data to the format expected by MuseTalk
            if isinstance(audio_data, bytes):
                # Convert bytes to numpy array (assuming 16-bit PCM)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = audio_data
            
            # Save temporary audio file for processing
            temp_audio_path = f"/tmp/webrtc_audio_{segment_id}.wav"
            import soundfile as sf
            sf.write(temp_audio_path, audio_array, 24000)
            
            # Extract audio features
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(
                temp_audio_path, weight_dtype=self.weight_dtype
            )
            
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.device,
                self.weight_dtype,
                self.whisper,
                librosa_length,
                fps=self.args.fps,
                audio_padding_length_left=self.args.audio_padding_length_left,
                audio_padding_length_right=self.args.audio_padding_length_right,
            )
            
            # Generate video frames
            video_frames = await self.generate_video_frames(whisper_chunks)
            
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.info(f"Processed audio segment {segment_id} in {processing_time:.3f}s, generated {len(video_frames)} frames")
            
            return video_frames, audio_array
            
        except Exception as e:
            logger.error(f"Error processing audio segment {segment_id}: {e}")
            return [], audio_data
    
    async def generate_video_frames(self, whisper_chunks):
        """Generate video frames from whisper chunks"""
        video_frames = []
        frame_idx = 0
        
        # Process in batches for efficiency
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.args.batch_size)
        
        with torch.no_grad():
            for i, (whisper_batch, latent_batch) in enumerate(gen):
                # GPU processing
                audio_feature_batch = self.pe(whisper_batch.to(self.device))
                latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
                
                # Generate predictions
                pred_latents = self.unet.model(
                    latent_batch,
                    self.timesteps,
                    encoder_hidden_states=audio_feature_batch
                ).sample
                
                pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
                recon = self.vae.decode_latents(pred_latents)
                
                # Process each frame
                for res_frame in recon:
                    processed_frame = self.process_single_frame(res_frame.cpu().numpy(), frame_idx)
                    if processed_frame is not None:
                        video_frames.append(processed_frame)
                    frame_idx += 1
                
                # Clear GPU memory periodically
                if i % 5 == 0:
                    torch.cuda.empty_cache()
        
        return video_frames
    
    def process_single_frame(self, res_frame, frame_idx):
        """Process a single frame with blending"""
        try:
            bbox = self.coord_list_cycle[frame_idx % len(self.coord_list_cycle)]
            ori_frame = self.frame_list_cycle[frame_idx % len(self.frame_list_cycle)].copy()
            x1, y1, x2, y2 = bbox
            
            # Resize generated frame
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            
            # Get mask and blend
            mask = self.mask_list_cycle[frame_idx % len(self.mask_list_cycle)]
            mask_crop_box = self.mask_coords_list_cycle[frame_idx % len(self.mask_coords_list_cycle)]
            
            # Blend frames
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            
            return combine_frame
            
        except Exception as e:
            logger.error(f"Error processing frame {frame_idx}: {e}")
            return None
    
    async def stream_frames_to_webrtc(self, video_frames, audio_data, video_track, audio_track):
        """
        Stream generated frames to WebRTC with synchronized audio
        """
        try:
            fps = self.args.fps
            sample_rate = 24000
            frame_duration = 1.0 / fps
            
            # Calculate audio chunks (20ms per chunk)
            audio_chunk_samples = int(0.02 * sample_rate)
            audio_chunks = []
            
            # Split audio into chunks
            for i in range(0, len(audio_data), audio_chunk_samples):
                chunk = audio_data[i:i + audio_chunk_samples]
                if len(chunk) < audio_chunk_samples:
                    # Pad the last chunk
                    chunk = np.pad(chunk, (0, audio_chunk_samples - len(chunk)))
                audio_chunks.append(chunk)
            
            # Stream frames and audio synchronously
            start_time = time.time()
            audio_idx = 0
            
            for frame_idx, frame in enumerate(video_frames):
                # Convert frame to WebRTC format
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Ensure frame is in BGR format for WebRTC
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Convert BGR to RGB for WebRTC
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Update video track
                await video_track.update_frame(frame)
                
                # Stream corresponding audio chunks
                chunks_per_frame = int(frame_duration / 0.02)  # Number of 20ms chunks per frame
                
                for _ in range(chunks_per_frame):
                    if audio_idx < len(audio_chunks):
                        # Convert to int16 and push to audio track
                        audio_chunk = (audio_chunks[audio_idx] * 32767).astype(np.int16).reshape(-1, 1)
                        audio_track.push_audio_data(audio_chunk)
                        audio_idx += 1
                        await asyncio.sleep(0.02)  # 20ms delay
                
                # Control frame rate
                elapsed = time.time() - start_time
                expected_time = (frame_idx + 1) * frame_duration
                if elapsed < expected_time:
                    await asyncio.sleep(expected_time - elapsed)
            
            self.frame_count += len(video_frames)
            logger.info(f"Streamed {len(video_frames)} frames to WebRTC")
            
        except Exception as e:
            logger.error(f"Error streaming to WebRTC: {e}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="WebRTC MuseTalk Inference")
    
    # Basic parameters
    parser.add_argument("--avatar_id", type=str, required=True, help="Avatar ID")
    parser.add_argument("--mp4_path", type=str, required=True, help="MP4 file path")
    parser.add_argument("--preparation", type=str, default="false", help="Whether to prepare avatar data")
    
    # Model parameters
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="VAE type")
    parser.add_argument("--unet_config", type=str, default="./models/musetalkV15/musetalk.json")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper")
    
    # Processing parameters
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--bbox_shift", type=int, default=-7, help="Bbox shift")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin")
    parser.add_argument("--parsing_mode", type=str, default="jaw", help="Parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90)
    parser.add_argument("--right_cheek_width", type=int, default=90)
    parser.add_argument("--audio_padding_length_left", type=int, default=2)
    parser.add_argument("--audio_padding_length_right", type=int, default=2)
    
    return parser.parse_args()

# Global inference engine instance
inference_engine = None

def get_inference_engine():
    """Get or create the global inference engine"""
    global inference_engine
    if inference_engine is None:
        args = parse_args()
        inference_engine = WebRTCMuseTalkInference(args)
    return inference_engine

if __name__ == "__main__":
    # Test the inference engine
    args = parse_args()
    engine = WebRTCMuseTalkInference(args)
    print("WebRTC MuseTalk inference engine initialized successfully")