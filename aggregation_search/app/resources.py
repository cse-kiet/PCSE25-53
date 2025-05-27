import os
import streamlit as st
import torch
import faiss
import json
from transformers import AutoProcessor, CLIPModel

# Global variables to store loaded resources
_device, _model_clip, _processor_clip, _index_place, _place_mapping, _places_order = None, None, None, None, None, None


# ----------------------------
# Global Marker Styling
# ----------------------------
# Revised mapping with supported FontAwesome v4.7 icon names.
place_styles = {
    "accounting": {"icon": "briefcase", "color": "gray"},
    "airport": {"icon": "plane", "color": "blue"},
    "amusement_park": {"icon": "smile-o", "color": "orange"},
    "aquarium": {"icon": "tint", "color": "aqua"},
    "art_gallery": {"icon": "paint-brush", "color": "purple"},
    "atm": {"icon": "credit-card", "color": "lightblue"},
    "bakery": {"icon": "birthday-cake", "color": "pink"},
    "bank": {"icon": "usd", "color": "teal"},
    "bar": {"icon": "glass", "color": "darkred"},
    "beauty_salon": {"icon": "scissors", "color": "pink"},
    "bicycle_store": {"icon": "bicycle", "color": "blue"},
    "book_store": {"icon": "book", "color": "blue"},
    "bowling_alley": {"icon": "futbol-o", "color": "green"},
    "bus_station": {"icon": "bus", "color": "orange"},
    "cafe": {"icon": "coffee", "color": "orange"},
    "campground": {"icon": "tree", "color": "green"},
    "car_dealer": {"icon": "car", "color": "red"},
    "car_rental": {"icon": "car", "color": "blue"},
    "car_repair": {"icon": "wrench", "color": "gray"},
    "car_wash": {"icon": "tint", "color": "blue"},
    "casino": {"icon": "dice", "color": "darkgreen"},
    "cemetery": {"icon": "tint", "color": "gray"},
    "church": {"icon": "cross", "color": "purple"},
    "city_hall": {"icon": "building", "color": "gray"},
    "clothing_store": {"icon": "shopping-cart", "color": "blue"},
    "convenience_store": {"icon": "shopping-cart", "color": "blue"},
    "courthouse": {"icon": "balance-scale", "color": "gray"},
    "dentist": {"icon": "medkit", "color": "red"},
    "department_store": {"icon": "shopping-cart", "color": "blue"},
    "doctor": {"icon": "user-md", "color": "red"},
    "drugstore": {"icon": "plus", "color": "red"},
    "electrician": {"icon": "bolt", "color": "yellow"},
    "electronics_store": {"icon": "tv", "color": "blue"},
    "embassy": {"icon": "globe", "color": "darkblue"},
    "fire_station": {"icon": "fire", "color": "red"},
    "florist": {"icon": "leaf", "color": "green"},
    "funeral_home": {"icon": "heart", "color": "black"},
    "furniture_store": {"icon": "bed", "color": "brown"},
    "gas_station": {"icon": "tint", "color": "blue"},
    "gym": {"icon": "heartbeat", "color": "red"},
    "hair_care": {"icon": "scissors", "color": "pink"},
    "hardware_store": {"icon": "wrench", "color": "gray"},
    "hindu_temple": {"icon": "fire", "color": "orange"},
    "home_goods_store": {"icon": "home", "color": "blue"},
    "hospital": {"icon": "plus-sign", "color": "darkred"},
    "insurance_agency": {"icon": "shield", "color": "blue"},
    "jewelry_store": {"icon": "diamond", "color": "pink"},
    "laundry": {"icon": "tint", "color": "blue"},
    "lawyer": {"icon": "gavel", "color": "gray"},
    "library": {"icon": "book", "color": "blue"},
    "light_rail_station": {"icon": "train", "color": "orange"},
    "liquor_store": {"icon": "glass", "color": "darkred"},
    "local_government_office": {"icon": "building", "color": "gray"},
    "locksmith": {"icon": "key", "color": "orange"},
    "lodging": {"icon": "home", "color": "green"},
    "meal_delivery": {"icon": "truck", "color": "red"},
    "meal_takeaway": {"icon": "cutlery", "color": "red"},
    "mosque": {"icon": "building", "color": "green"},
    "movie_rental": {"icon": "film", "color": "purple"},
    "movie_theater": {"icon": "film", "color": "purple"},
    "moving_company": {"icon": "truck", "color": "blue"},
    "museum": {"icon": "university", "color": "purple"},
    "night_club": {"icon": "glass", "color": "darkred"},
    "painter": {"icon": "paint-brush", "color": "purple"},
    "park": {"icon": "tree", "color": "darkgreen"},
    "parking": {"icon": "car", "color": "blue"},
    "pet_store": {"icon": "paw", "color": "orange"},
    "pharmacy": {"icon": "plus-square", "color": "red"},
    "physiotherapist": {"icon": "heartbeat", "color": "red"},
    "plumber": {"icon": "wrench", "color": "gray"},
    "police": {"icon": "shield", "color": "blue"},
    "post_office": {"icon": "envelope", "color": "blue"},
    "primary_school": {"icon": "graduation-cap", "color": "blue"},
    "real_estate_agency": {"icon": "home", "color": "green"},
    "restaurant": {"icon": "cutlery", "color": "red"},
    "roofing_contractor": {"icon": "building", "color": "gray"},
    "rv_park": {"icon": "home", "color": "green"},
    "school": {"icon": "graduation-cap", "color": "blue"},
    "secondary_school": {"icon": "graduation-cap", "color": "blue"},
    "shoe_store": {"icon": "shopping-cart", "color": "blue"},
    "shopping_mall": {"icon": "shopping-cart", "color": "blue"},
    "spa": {"icon": "leaf", "color": "green"},
    "stadium": {"icon": "futbol-o", "color": "orange"},
    "storage": {"icon": "archive", "color": "gray"},
    "store": {"icon": "shopping-cart", "color": "blue"},
    "subway_station": {"icon": "train", "color": "orange"},
    "supermarket": {"icon": "shopping-cart", "color": "blue"},
    "synagogue": {"icon": "star", "color": "purple"},
    "taxi_stand": {"icon": "taxi", "color": "yellow"},
    "tourist_attraction": {"icon": "star", "color": "darkblue"},
    "train_station": {"icon": "train", "color": "orange"},
    "transit_station": {"icon": "train", "color": "orange"},
    "travel_agency": {"icon": "suitcase", "color": "blue"},
    "university": {"icon": "graduation-cap", "color": "blue"},
    "veterinary_care": {"icon": "paw", "color": "green"},
    "zoo": {"icon": "paw", "color": "green"}
}


@st.cache_resource()
def load_resources():
    global _device, _model_clip, _processor_clip, _index_place, _place_mapping, _places_order
    
    data_dir = "data" # Relative to CWD of final_app.py (aggregation_search/)
    index_path = os.path.join(data_dir, "aggregated_clip.index")
    mapping_path = os.path.join(data_dir, "place_mapping.json")
    order_path = os.path.join(data_dir, "places_order.json")

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    index = faiss.read_index(index_path)
    with open(mapping_path, "r") as f:
        place_mapping = json.load(f)
    with open(order_path, "r") as f:
        places_order = json.load(f)
    
    _device, _model_clip, _processor_clip, _index_place, _place_mapping, _places_order = \
        device, model, processor, index, place_mapping, places_order
    
    return device, model, processor, index, place_mapping, places_order

def get_loaded_resources():
    if _device is None: # Ensure loaded
        load_resources()
    return _device, _model_clip, _processor_clip, _index_place, _place_mapping, _places_order


def get_marker_style(place):
    types = place.get("types", [])
    for t in types:
        if t in place_styles:
            return place_styles[t]
    return {"icon": "map-marker", "color": "blue"}
