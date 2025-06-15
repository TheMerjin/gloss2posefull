# Gloss2PoseFull - WASL Video Processing Pipeline

This project processes videos from the Worldwide American Sign Language (WASL) dataset to create a mapping between sign language words and their corresponding pose data using OpenPose.

## Prerequisites

1. Python 3.8 or higher
2. OpenPose installed and built (https://github.com/CMU-Perceptual-Computing-Lab/openpose)
3. FFmpeg installed (required for video processing)

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Update the OpenPose path in `scripts/wasl_processor.py`:
```python
self.openpose_path = Path("path/to/your/openpose/installation")
```

## Usage

1. Prepare your WASL metadata file in JSON format with the following structure:
```json
[
    {
        "video_id": "youtube_video_id",
        "word": "sign_word",
        "start_time": start_time_in_seconds,
        "end_time": end_time_in_seconds
    },
    ...
]
```

2. Run the processing script:
```bash
python scripts/wasl_processor.py
```

The script will:
- Download videos from YouTube using the provided metadata
- Split videos into individual word segments
- Process each segment with OpenPose to generate pose data
- Create a mapping between words and their corresponding pose data

## Output Structure

The processed data will be organized in the following structure:
```
data/wasl_processed/
├── videos/           # Original and split video files
├── poses/           # OpenPose JSON output files
└── metadata/        # Word-pose mapping and other metadata
```

## Notes

- The script limits video downloads to 720p quality to save bandwidth and processing time
- OpenPose processing can be resource-intensive. Consider processing videos in batches if needed
- Make sure you have sufficient disk space for video storage and processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
