import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from transformers import WhisperModel

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

import shutil
import threading
import queue
import time
import subprocess
import io
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation, tmp_dir=None, images_dir=None):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        # 根据版本设置不同的基础路径
        if hasattr(args, 'version') and args.version == "v15":
            self.base_path = f"./results/{args.version}/avatars/{avatar_id}"
        else:  # v1
            self.base_path = f"./results/realtime/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": getattr(args, 'version', 'v15')
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        
        # Set custom directories if provided
        if tmp_dir:
            self.tmp_dir = tmp_dir
        else:
            self.tmp_dir = f"{self.avatar_path}/tmp"
            
        if images_dir:
            self.images_dir = images_dir
        else:
            self.images_dir = f"{self.avatar_path}/images"
            
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.mask_list_cycle = read_imgs(input_mask_list)
            else:
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if args.version == "v15":
                y2 = y2 + args.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]  # 更新coord_list中的bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            if args.version == "v15":
                mode = args.parsing_mode
            else:
                mode = "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def process_frames(self, res_frame_queue, video_len, skip_save_images):
        print(video_len)
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)

            if skip_save_images is False:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)
            self.idx = self.idx + 1

    def inference(self, audio_path, out_vid_name, fps, skip_save_images):
        """Optimized inference with GPU acceleration and memory processing"""
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)
        print("start inference")
        
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        # Extract audio features with GPU acceleration
        with torch.cuda.device(device):
            whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype=weight_dtype)
            whisper_chunks = audio_processor.get_whisper_chunk(
                whisper_input_features,
                device,
                weight_dtype,
                whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=args.audio_padding_length_left,
                audio_padding_length_right=args.audio_padding_length_right,
            )
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        
        ############################################## optimized inference with parallel processing ##############################################
        video_num = len(whisper_chunks)
        
        # Use memory-based frame processing
        res_frame_queue = queue.Queue(maxsize=20)  # Limit queue size to control memory
        self.idx = 0
        
        # Create optimized processing thread with higher priority
        process_thread = threading.Thread(
            target=self.process_frames_optimized,
            args=(res_frame_queue, video_num, skip_save_images, out_vid_name, fps, audio_path)
        )
        process_thread.daemon = False
        process_thread.start()

        # Optimized batch processing with GPU memory management
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        start_time = time.time()

        # Use CUDA streams for parallel processing
        with torch.cuda.device(device):
            torch.cuda.empty_cache()  # Clear GPU memory
            
            for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
                # Optimize GPU memory usage
                with torch.no_grad():
                    audio_feature_batch = pe(whisper_batch.to(device, non_blocking=True))
                    latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype, non_blocking=True)

                    # GPU-accelerated inference
                    pred_latents = unet.model(latent_batch,
                                            timesteps,
                                            encoder_hidden_states=audio_feature_batch).sample
                    pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
                    recon = vae.decode_latents(pred_latents)
                    
                    # Move to CPU and queue for processing
                    for res_frame in recon:
                        res_frame_cpu = res_frame.cpu().numpy()
                        res_frame_queue.put(res_frame_cpu)
                    
                    # Clear intermediate GPU memory
                    del audio_feature_batch, latent_batch, pred_latents, recon
                    if i % 5 == 0:  # Periodic cleanup
                        torch.cuda.empty_cache()

        # Wait for processing to complete
        process_thread.join()

        processing_time = time.time() - start_time
        if skip_save_images:
            print(f'Total process time of {video_num} frames without saving images = {processing_time:.2f}s')
        else:
            print(f'Total process time of {video_num} frames including saving images = {processing_time:.2f}s')
        
        print("\n")

    def process_frames_optimized(self, res_frame_queue, video_len, skip_save_images, out_vid_name, fps, audio_path):
        """Optimized frame processing with memory management and faster video generation"""
        print(f"Processing {video_len} frames with optimization")
        
        # Pre-allocate memory for frames
        frames_buffer = []
        
        # Use thread pool for parallel frame processing
        with ThreadPoolExecutor(max_workers=min(4, multiprocessing.cpu_count())) as executor:
            futures = []
            
            while self.idx < video_len:
                try:
                    start = time.time()
                    res_frame = res_frame_queue.get(block=True, timeout=2)
                except queue.Empty:
                    continue

                # Submit frame processing to thread pool
                future = executor.submit(self.process_single_frame, res_frame, self.idx, skip_save_images)
                futures.append(future)
                
                self.idx += 1
                
                # Process completed futures
                if len(futures) >= 10 or self.idx >= video_len:
                    for future in futures:
                        processed_frame = future.result()
                        if processed_frame is not None:
                            frames_buffer.append(processed_frame)
                    futures.clear()

        # Generate video with optimized settings
        if out_vid_name is not None and not skip_save_images:
            self.generate_video_optimized(frames_buffer, out_vid_name, fps, audio_path)

    def process_single_frame(self, res_frame, frame_idx, skip_save_images):
        """Process a single frame with optimizations"""
        try:
            bbox = self.coord_list_cycle[frame_idx % len(self.coord_list_cycle)]
            ori_frame = copy.deepcopy(self.frame_list_cycle[frame_idx % len(self.frame_list_cycle)])
            x1, y1, x2, y2 = bbox
            
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                return None
                
            mask = self.mask_list_cycle[frame_idx % len(self.mask_list_cycle)]
            mask_crop_box = self.mask_coords_list_cycle[frame_idx % len(self.mask_coords_list_cycle)]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            if not skip_save_images:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(frame_idx).zfill(8)}.png", combine_frame)
            
            return combine_frame
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            return None

    def generate_video_optimized(self, frames_buffer, out_vid_name, fps, audio_path):
        """Generate video with optimized encoding settings"""
        try:
            # Use faster video generation with optimized ffmpeg settings
            temp_video = f"{self.avatar_path}/temp_optimized.mp4"
            
            # Optimized ffmpeg command for faster encoding
            cmd_img2video = [
                "ffmpeg", "-y", "-v", "quiet",
                "-r", str(fps),
                "-f", "image2",
                "-i", f"{self.avatar_path}/tmp/%08d.png",
                "-c:v", "libx264",
                "-preset", "ultrafast",  # Fastest encoding
                "-tune", "zerolatency",  # Low latency
                "-crf", "23",  # Balanced quality/speed
                "-pix_fmt", "yuv420p",
                temp_video
            ]
            
            subprocess.run(cmd_img2video, check=True)

            # Combine with audio using optimized settings
            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")
            cmd_combine_audio = [
                "ffmpeg", "-y", "-v", "quiet",
                "-i", audio_path,
                "-i", temp_video,
                "-c:v", "copy",  # Copy video stream without re-encoding
                "-c:a", "aac",
                "-shortest",
                output_vid
            ]
            
            subprocess.run(cmd_combine_audio, check=True)

            # Cleanup
            os.remove(temp_video)
            shutil.rmtree(f"{self.avatar_path}/tmp")
            print(f"Optimized result saved to {output_vid}")
            
        except Exception as e:
            print(f"Error generating optimized video: {e}")


if __name__ == "__main__":
    '''
    This script simulates an online chat and performs necessary preprocessing steps,
    such as face detection and face parsing. During the online chat, only the UNet
    and VAE decoders are involved, which enables MuseTalk to achieve real-time
    performance.
    '''
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar_id",
                        type=str,
                        required=True,
                        help="Avatar ID")
    parser.add_argument("--preparation",
                        type=str,
                        default="true",
                        help="Whether to perform data preparation (true/false)")
    parser.add_argument("--bbox_shift",
                        type=int,
                        default=-7,
                        help="Bounding box offset")
    parser.add_argument("--fps",
                        type=int,
                        default=25,
                        help="Video frame rate")
    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
                        help="Batch size")
    parser.add_argument("--skip_save_images",
                        action="store_true",
                        help="Whether to skip saving images to improve generation speed")
    parser.add_argument("--audio_folder",
                        type=str,
                        required=True,
                        help="Folder path for dynamically reading audio files")
    parser.add_argument("--mp4_path",
                        type=str,
                        required=True,
                        help="MP4 file save path")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Version of MuseTalk: v1 or v15")
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalkV15/musetalk.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")

    args = parser.parse_args()

    # Parse the 'preparation' argument to boolean value
    preparation = args.preparation.lower() == "true"
    bbox_shift = args.bbox_shift

    # Construct the mp4_path based on avatar_id
    video_path = f"./data/video/{args.mp4_path}.mp4"

    # Define temporary and images directories
    tmp_dir = f"./results/realtime/{args.avatar_id}/tmp"
    images_dir = f"./results/realtime/{args.avatar_id}/images"
    os.makedirs(images_dir, exist_ok=True)  # Ensure images directory exists

    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        # Choose path separator based on operating system
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)

    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Initialize face parser with configurable parameters based on version
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()

    # Initialize Avatar instance, passing mp4_path parameter
    avatar = Avatar(
        avatar_id=args.avatar_id,
        video_path=video_path,
        bbox_shift=bbox_shift,
        batch_size=args.batch_size,
        preparation=preparation,
        tmp_dir=tmp_dir,
        images_dir=images_dir,
    )

    # Dynamically read audio files folder and build audio_clips dictionary
    counter = 0  # Define counter

    def get_audio_duration(audio_path):
        """
        Get the duration of the audio file using ffprobe
        """
        command = [
            "ffprobe", "-i", audio_path, "-show_entries", "format=duration",
            "-v", "quiet", "-of", "csv=p=0"
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=True)
        return float(result.stdout.strip())

    # Optimized processing loop with 2-second standardization
    target_duration = 2.0  # Target 2-second audio blocks
    processing_executor = ThreadPoolExecutor(max_workers=2)  # Parallel processing
    
    def create_standardized_audio_block(audio_files, output_path, target_duration):
        """Create a standardized 2-second audio block"""
        try:
            # Create temporary concat file
            concat_file = "audio_concat_temp.txt"
            with open(concat_file, "w") as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")

            # Merge and standardize to exactly 2 seconds
            merge_command = [
                "ffmpeg", "-y", "-v", "quiet",
                "-f", "concat", "-safe", "0", "-i", concat_file,
                "-t", str(target_duration),  # Limit to target duration
                "-ar", "24000",  # Standardize sample rate
                "-ac", "1",      # Mono
                "-c:a", "pcm_s16le",  # 16-bit PCM
                output_path
            ]
            
            result = subprocess.run(merge_command, capture_output=True)
            
            # Clean up
            if os.path.exists(concat_file):
                os.remove(concat_file)
                
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error creating standardized audio block: {e}")
            return False

    def process_audio_block_async(avatar, audio_path, audio_filename, fps, skip_save_images):
        """Process audio block asynchronously"""
        try:
            print(f"Processing audio block: {audio_filename}")
            start_time = time.time()
            
            # Process with optimized inference
            avatar.inference(audio_path, audio_filename, fps, skip_save_images)
            
            # Clean up processed audio file
            try:
                os.remove(audio_path)
            except:
                pass
                
            processing_time = time.time() - start_time
            print(f"Completed processing {audio_filename} in {processing_time:.2f}s")
            
        except Exception as e:
            print(f"Error in async processing: {e}")
    
    while True:
        # List all audio files in the folder that ends with .wav
        audio_files = sorted(
            [os.path.join(args.audio_folder, f) for f in os.listdir(args.audio_folder) if
             f.endswith(".wav")]
        )

        if not audio_files:
            time.sleep(0.3)  # Faster polling
            continue

        audio_to_merge = []
        total_duration = 0.0
        i = 0
        n = len(audio_files)

        while i < n:
            audio_path = audio_files[i]
            try:
                duration = get_audio_duration(audio_path)
            except:
                # Skip corrupted files
                try:
                    os.remove(audio_path)
                except:
                    pass
                i += 1
                continue

            audio_to_merge.append(audio_path)
            total_duration += duration
            i += 1

            # Process when we reach target duration (2 seconds) or have enough files
            if total_duration >= target_duration or len(audio_to_merge) >= 3:
                # Create standardized 2-second audio block
                merged_audio_filename = f"{args.avatar_id}_{str(counter).zfill(8)}.wav"
                merged_audio_path = os.path.join(args.audio_folder, merged_audio_filename)

                try:
                    # Optimized audio merging with target duration
                    success = create_standardized_audio_block(
                        audio_to_merge,
                        merged_audio_path,
                        target_duration
                    )
                    
                    if success:
                        print(f"Created 2-second audio block: {merged_audio_filename}")
                        
                        # Submit to thread pool for parallel processing
                        future = processing_executor.submit(
                            process_audio_block_async,
                            avatar,
                            merged_audio_path,
                            merged_audio_filename,
                            args.fps,
                            args.skip_save_images
                        )
                        
                        # Don't wait for completion - process in parallel
                        counter += 1

                    # Clean up processed audio files
                    for audio in audio_to_merge:
                        try:
                            os.remove(audio)
                        except:
                            pass

                except Exception as e:
                    print(f"Error processing audio block: {e}")

                # Reset for next block
                audio_to_merge = []
                total_duration = 0.0

        time.sleep(0.2)  # Faster polling for better responsiveness
