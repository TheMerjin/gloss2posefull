import fiftyone as fo
import fiftyone.utils.huggingface as fouh
import os
import urllib.request

python_version = r" C:\Users\Sreek\AppData\Local\Programs\Python\Python310\python.exe"
# Load WLASL dataset from Hugging Face
print("Loading WLASL from Hugging Face...")
dataset = fouh.load_from_hub("Voxel51/WLASL")

# Optionally launch GUI to inspect dataset

# Create output directory for videos
os.makedirs("data/raw_videos", exist_ok=True)

# Loop through FiftyOne samples to download videos
for sample in dataset:
    gloss = sample["gloss"]
    url = sample["video"]
    instance_id = sample["instance_id"]
    out_path = f"data/raw_videos/{gloss}_{instance_id}.mp4"

    if not os.path.exists(out_path):
        try:
            print(f"Downloading: {gloss} (instance {instance_id})")
            urllib.request.urlretrieve(url, out_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
