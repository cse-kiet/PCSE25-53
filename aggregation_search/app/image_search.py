import streamlit as st # For st.error
import torch
import numpy as np
from PIL import Image # For Image.open

from . import resources


# ----------------------------
# Image Search Function
# ----------------------------
def search_places(query_files, processed_images=None, top_k=5):
    device, model_clip, processor_clip, index_place, place_mapping, places_order = resources.get_loaded_resources()
    
    query_embeddings = []
    if processed_images is None:
        processed_images = [Image.open(file) for file in query_files]
    
    with torch.inference_mode():
        for img in processed_images:
            try:
                inputs = processor_clip(images=img, return_tensors="pt").to(device)
                embedding = model_clip.get_image_features(**inputs)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                query_embeddings.append(embedding.cpu().numpy())
            except Exception as e:
                st.error(f"Error processing image: {e}")
                query_embeddings.append(np.zeros((1, 512), dtype=np.float32))
    query_embeddings = np.vstack(query_embeddings).astype(np.float32)
    distances, indices = index_place.search(query_embeddings, top_k)
    
    results = []
    for i, q_indices in enumerate(indices):
        result = []
        for idx in q_indices:
            if idx < 0:
                continue
            place = places_order[idx]
            rep_image = place_mapping[place]
            result.append({
                "place": place,
                "distance": distances[i][0],
                "rep_image": rep_image
            })
        results.append(result)
    return results