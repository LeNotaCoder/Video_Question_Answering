import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import av
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# Initialize CLIP globally (will be loaded in main.py)
clip_model = None
processor = None
device = None

def init_clip(model_name="openai/clip-vit-base-patch32", dev="cuda"):
    global clip_model, processor, device
    device = torch.device(dev if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

def read_video_frames(video_path, resize=(224, 224)):
    global clip_model, processor, device
    frames = []
    try:
        container = av.open(video_path)
    except Exception as e:
        return None

    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        frames.append(img)

    if not frames:
        return None

    encoded_frames = []
    for frame in frames:
        pil_frame = Image.fromarray(frame)
        inputs = processor(images=pil_frame, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, dim=-1)

        encoded_frames.append(image_features.squeeze())

    return torch.stack(encoded_frames).to(device)


class MyDataset(Dataset):
    def __init__(self, data_keys, data_dict, path):
        self.data_keys = data_keys
        self.data_dict = data_dict
        self.path = path

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        key = self.data_keys[idx]
        video_path = self.path + key.split("_")[0] + ".mp4"
        video = read_video_frames(video_path)
        
        if video is None:
            return None

        options = self.data_dict[key]['options']
        true_label = torch.tensor(self.data_dict[key]['answer'], dtype=torch.long)
        
        return video, options, true_label


def custom_collate(batch):
    batch = [(v, o, l) for v, o, l in batch if v is not None]
    
    if len(batch) == 0:
        return None
    
    videos, options, labels = zip(*batch)
    max_len = max(v.shape[0] for v in videos)
    padded_videos = []

    for v in videos:
        device = v.device
        if v.shape[0] < max_len:
            pad_size = max_len - v.shape[0]
            padding = torch.zeros((pad_size, v.shape[1]), device=device)
            v = torch.cat([v, padding], dim=0)
        padded_videos.append(v)

    videos = torch.stack(padded_videos)
    labels = torch.tensor(labels, device=videos.device)
    return videos, list(options), labels
