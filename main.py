import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import pandas as pd
import os
from models import Final_Model
from functions import init_clip, MyDataset, custom_collate, device

# ================================
# Configuration
# ================================
VIDEO_PATH = "" # Path to the videos
JSON_PATH = "" # Path to the metadata
CHECKPOINT_PATH = "one.pth"

BATCH_SIZE = 5
NUM_TIMES = 8
EPOCHS_PER_ROUND = 20
LR = 1e-5
HIDDEN_DIM = 512
NUM_LAYERS = 1

# ================================
# Load and Prepare Data
# ================================
with open(JSON_PATH, 'r') as f:
    all_train = json.load(f)

labels = pd.DataFrame(all_train)

videos = []
for _, _, files in os.walk(VIDEO_PATH):
    for file in files:
        if file.endswith(".mp4") and file[:-4] in labels.keys():
            videos.append(file)

final_dataset = {}
for label in labels.columns:
    video_file = label + ".mp4"
    if video_file in videos:
        for i in range(len(labels[label][5])):
            item = labels[label][5][i]
            context = item['question']
            options = item['options']
            answer = item['answer_id']

            final_options = [context + ", " + str(opt) for opt in options]
            key = f"{label}_{i}"
            final_dataset[key] = {'options': final_options, 'answer': answer}

all_keys = sorted(list(final_dataset.keys()))
print(f"Total samples: {len(all_keys)}")

# ================================
# Initialize CLIP
# ================================
init_clip()

# ================================
# Training Loop
# ================================
model = Final_Model(input_dim=512, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

start_round = 0
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("Loaded checkpoint.")
    # Optionally resume from last round
    # start_round = ...

for num in range(start_round, NUM_TIMES):
    print(f"\n=== Training Round {num+1}/{NUM_TIMES} ===")
    
    # Slice data
    start_idx = 300 * num
    end_idx = 300 * (num + 1)
    batch_keys = all_keys[start_idx:end_idx]
    
    dataset = MyDataset(batch_keys, final_dataset, VIDEO_PATH + "/")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

    for epoch in range(EPOCHS_PER_ROUND):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch in loader:
            if batch is None:
                continue

            videos, options_batch, true_labels = batch
            videos = videos.to(device)
            true_labels = true_labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)  # (B, 512)

            all_logits = []
            for i in range(len(options_batch)):
                text_inputs = processor(text=options_batch[i], return_tensors="pt", 
                                        padding=True, truncation=True).to(device)
                with torch.no_grad():
                    text_features = clip_model.get_text_features(**text_inputs)
                    text_features = F.normalize(text_features, dim=-1)
                
                video_feat = outputs[i].unsqueeze(0)
                logits = video_feat @ text_features.T  # (1, num_options)
                all_logits.append(logits)

            logits_batch = torch.cat(all_logits, dim=0)
            loss = criterion(logits_batch, true_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        avg_loss = running_loss / batch_count if batch_count > 0 else float('inf')
        print(f"Round {num+1}, Epoch {epoch+1}/{EPOCHS_PER_ROUND}, Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, CHECKPOINT_PATH)
    print(f"Checkpoint saved after round {num+1}.")
