import os
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yt_dlp
import cv2
from moviepy import VideoFileClip
import logging
import requests
import zipfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WASLProcessor:
    def __init__(self, output_dir="data/wasl_processed"):
        self.output_dir = Path(output_dir)
        self.videos_dir = self.output_dir / "videos"
        self.pose_dir = self.output_dir / "poses"
        self.metadata_dir = self.output_dir / "metadata"
        self.raw_data_dir = self.output_dir / "raw"
        
        # Create necessary directories
        for dir_path in [self.videos_dir, self.pose_dir, self.metadata_dir, self.raw_data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # OpenPose path - update this to your OpenPose installation
        self.openpose_path = Path("path/to/openpose")
        
        # Load metadata cache
        self.metadata_cache = {}
        
    def get_WASL(self, dataset_url=None):
        """
        Download and process the WASL dataset.
        
        Args:
            dataset_url (str, optional): URL to download the WASL dataset. If None, uses default URL.
        """
        logger.info("Starting WASL dataset download and processing...")
        
        # Default WASL dataset URL (update this with the actual URL)
        if dataset_url is None:
            dataset_url = "https://github.com/dxli94/WASL/raw/master/WASL.zip"
            
        try:
            # Download the dataset
            zip_path = self.raw_data_dir / "WASL.zip"
            logger.info(f"Downloading WASL dataset from {dataset_url}")
            
            response = requests.get(dataset_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
                    
            # Extract the dataset
            logger.info("Extracting WASL dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_data_dir)
                
            # Process the dataset structure
            self._process_WASL_structure()
            
            # Clean up
            zip_path.unlink()
            logger.info("WASL dataset download and processing completed!")
            
        except Exception as e:
            logger.error(f"Error downloading or processing WASL dataset: {str(e)}")
            raise
            
    def _process_WASL_structure(self):
        """Process the extracted WASL dataset structure and create metadata."""
        logger.info("Processing WASL dataset structure...")
        
        # Expected WASL directory structure
        wasl_dir = self.raw_data_dir / "WASL"
        if not wasl_dir.exists():
            raise FileNotFoundError("WASL directory not found in extracted files")
            
        # Process annotations
        annotations_file = wasl_dir / "annotations.json"
        if not annotations_file.exists():
            raise FileNotFoundError("WASL annotations file not found")
            
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
            
        # Create metadata structure
        metadata = []
        for entry in annotations:
            video_id = entry.get('video_id')
            if not video_id:
                continue
                
            # Extract word and timestamp information
            word = entry.get('word', '')
            start_time = entry.get('start_time', 0)
            end_time = entry.get('end_time', 0)
            
            if word and start_time < end_time:
                metadata.append({
                    'video_id': video_id,
                    'word': word,
                    'start_time': start_time,
                    'end_time': end_time
                })
                
        # Save processed metadata
        metadata_path = self.metadata_dir / "wasl_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Processed {len(metadata)} entries from WASL dataset")
        
    def _get_word_timestamps(self, video_id):
        """Extract word timestamps from metadata for a given video."""
        if video_id not in self.metadata_cache:
            # Load metadata if not in cache
            metadata_path = self.metadata_dir / "wasl_metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError("WASL metadata file not found")
                
            with open(metadata_path, 'r') as f:
                self.metadata_cache = json.load(f)
                
        # Find entries for this video
        video_entries = [entry for entry in self.metadata_cache 
                        if entry['video_id'] == video_id]
        
        if not video_entries:
            logger.warning(f"No metadata found for video {video_id}")
            return {}
            
        # Create word to timestamp mapping
        word_timestamps = {}
        for entry in video_entries:
            word = entry['word']
            start_time = entry['start_time']
            end_time = entry['end_time']
            word_timestamps[word] = (start_time, end_time)
            
        return word_timestamps
        
    def download_wasl_videos(self, wasl_metadata_path):
        """Download videos from WASL dataset using metadata."""
        logger.info("Starting WASL video downloads...")
        
        with open(wasl_metadata_path, 'r') as f:
            metadata = json.load(f)
            
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit to 720p
            'outtmpl': str(self.videos_dir / '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for entry in tqdm(metadata, desc="Downloading videos"):
                try:
                    video_id = entry['video_id']
                    if not (self.videos_dir / f"{video_id}.mp4").exists():
                        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                except Exception as e:
                    logger.error(f"Error downloading video {video_id}: {str(e)}")
                    
    def split_video_into_words(self, video_path, word_timestamps):
        """Split video into individual word segments."""
        video = VideoFileClip(str(video_path))
        word_clips = []
        
        for word, (start, end) in word_timestamps.items():
            clip = video.subclip(start, end)
            output_path = self.videos_dir / f"{video_path.stem}_{word}.mp4"
            clip.write_videofile(str(output_path), codec='libx264', audio=False)
            word_clips.append((word, output_path))
            
        video.close()
        return word_clips
        
    def process_with_openpose(self, video_path):
        """Process video with OpenPose to generate pose data."""
        output_path = self.pose_dir / f"{video_path.stem}.json"
        
        # OpenPose command
        cmd = [
            str(self.openpose_path / "build/examples/openpose/openpose.bin"),
            "--video", str(video_path),
            "--write_json", str(self.pose_dir),
            "--display", "0",
            "--render_pose", "0"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Processed {video_path} with OpenPose")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing {video_path} with OpenPose: {str(e)}")
            
    def create_word_pose_mapping(self, metadata):
        """Create mapping between words and their corresponding pose data."""
        mapping = {}
        
        for entry in metadata:
            word = entry['word']
            video_id = entry['video_id']
            pose_file = self.pose_dir / f"{video_id}.json"
            
            if pose_file.exists():
                with open(pose_file, 'r') as f:
                    pose_data = json.load(f)
                mapping[word] = pose_data
                
        # Save mapping
        with open(self.metadata_dir / "word_pose_mapping.json", 'w') as f:
            json.dump(mapping, f, indent=2)
            
    def process_dataset(self, wasl_metadata_path):
        """Main pipeline to process the entire WASL dataset."""
        logger.info("Starting WASL dataset processing...")
        
        # Download videos
        self.download_wasl_videos(wasl_metadata_path)
        
        # Process each video
        for video_path in tqdm(list(self.videos_dir.glob("*.mp4")), desc="Processing videos"):
            # Get word timestamps from metadata
            word_timestamps = self._get_word_timestamps(video_path.stem)
            
            # Split video into words
            word_clips = self.split_video_into_words(video_path, word_timestamps)
            
            # Process each word clip with OpenPose
            for word, clip_path in word_clips:
                self.process_with_openpose(clip_path)
                
        # Create final word-pose mapping
        with open(wasl_metadata_path, 'r') as f:
            metadata = json.load(f)
        self.create_word_pose_mapping(metadata)
        
        logger.info("WASL dataset processing completed!")

if __name__ == "__main__":
    processor = WASLProcessor()
    # First get the WASL dataset
    processor.get_WASL()
    # Then process it
    processor.process_dataset("data/wasl_processed/metadata/wasl_metadata.json") 