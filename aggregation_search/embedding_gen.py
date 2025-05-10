import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import faiss
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict

# Configure environment for optimal CPU performance
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUM_THREADS"] = "8"
faiss.omp_set_num_threads(8)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Load CLIP model with memory-efficient settings
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32)
model_clip = model_clip.to(device).eval()
processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Collect images per place using a dictionary (key: place, value: list of image paths)
place_images = defaultdict(list)
base_folder = os.path.join("..", "places_images")
for root, _, files in os.walk(base_folder):
    place = os.path.basename(root)
    for file in files:
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            place_images[place].append(os.path.join(root, file))

# Set the optimal batch size based on your experiments
batch_size = 256

aggregated_embeddings = []  # One aggregated embedding per place
place_to_rep_image = {}     # Map from place to a representative image (for display)
places_order = []           # List of place names in the order of aggregation

# Process each place: use batching to compute embeddings
for place, image_paths in tqdm(place_images.items(), desc="Aggregating embeddings per place"):
    embeddings_list = []
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        # Load images in the batch
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
            except Exception as e:
                print(f"Error loading {path}: {e}")
                img = Image.new('RGB', (224, 224))  # Fallback image
            batch_images.append(img)
        
        # Process the batch through CLIP
        inputs = processor_clip(images=batch_images, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            batch_emb = model_clip.get_image_features(**inputs)
        batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
        # Detach the tensor and convert to numpy
        embeddings_list.append(batch_emb.detach().cpu().numpy())
    
    # If the place has any images, compute the centroid (average embedding)
    if embeddings_list:
        place_emb = np.mean(np.vstack(embeddings_list), axis=0, keepdims=True)
        aggregated_embeddings.append(place_emb)
        # Use the first image as the representative image for display purposes
        place_to_rep_image[place] = image_paths[0]
        places_order.append(place)

# Stack all aggregated embeddings and prepare the FAISS index.
aggregated_embeddings = np.vstack(aggregated_embeddings).astype(np.float32)
faiss.normalize_L2(aggregated_embeddings)  # ensure normalization for cosine similarity

index_place = faiss.IndexFlatIP(aggregated_embeddings.shape[1])
index_place.add(aggregated_embeddings)

# Save the aggregated index and mappings.
faiss.write_index(index_place, "aggregated_clip.index")
with open('place_mapping.json', 'w') as f:
    json.dump(place_to_rep_image, f)
with open('places_order.json', 'w') as f:
    json.dump(places_order, f)

print(f"Aggregated index created with {index_place.ntotal} entries.")
