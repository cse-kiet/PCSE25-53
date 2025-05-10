import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import datetime
import time
import streamlit as st
import torch
import faiss
import numpy as np
import json
import requests
import math
import folium
import pandas as pd
from PIL import Image
from streamlit_folium import st_folium
from transformers import AutoProcessor, CLIPModel
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import concurrent.futures

st.set_page_config(page_title="Travel Planner", layout="wide")

# ----------------------------
# Constants and Helper Functions
# ----------------------------
ALL_PLACE_TYPES = [
    "accounting", "airport", "amusement_park", "aquarium", "art_gallery", "atm",
    "bakery", "bank", "bar", "beauty_salon", "bicycle_store", "book_store",
    "bowling_alley", "bus_station", "cafe", "campground", "car_dealer",
    "car_rental", "car_repair", "car_wash", "casino", "cemetery", "church",
    "city_hall", "clothing_store", "convenience_store", "courthouse", "dentist",
    "department_store", "doctor", "drugstore", "electrician", "electronics_store",
    "embassy", "fire_station", "florist", "funeral_home", "furniture_store",
    "gas_station", "gym", "hair_care", "hardware_store", "hindu_temple",
    "home_goods_store", "hospital", "insurance_agency", "jewelry_store", "laundry",
    "lawyer", "library", "light_rail_station", "liquor_store", "local_government_office",
    "locksmith", "lodging", "meal_delivery", "meal_takeaway", "mosque", "movie_rental",
    "movie_theater", "moving_company", "museum", "night_club", "painter", "park",
    "parking", "pet_store", "pharmacy", "physiotherapist", "plumber", "police",
    "post_office", "primary_school", "real_estate_agency", "restaurant", "roofing_contractor",
    "rv_park", "school", "secondary_school", "shoe_store", "shopping_mall", "spa",
    "stadium", "storage", "store", "subway_station", "supermarket", "synagogue",
    "taxi_stand", "tourist_attraction", "train_station", "transit_station", "travel_agency",
    "university", "veterinary_care", "zoo"
]

MEAL_SLOTS = {
    "Breakfast": (480, 540),   # 8:00 - 9:00
    "Lunch": (720, 780),       # 12:00 - 13:00
    "Dinner": (1140, 1200)     # 19:00 - 20:00
}

def get_purpose(place):
    types = place.get("types", [])
    if any(t in types for t in ["restaurant", "cafe", "bakery", "meal_takeaway"]):
        return "Eating"
    elif any(t in types for t in ["museum", "tourist_attraction"]):
        return "Sightseeing"
    elif "park" in types:
        return "Recreation"
    elif "shopping_mall" in types:
        return "Shopping"
    elif "lodging" in types:
        return "Stay"
    else:
        return "Visit"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a)) * 1000  # meters

# ----------------------------
# Global Resource Loading
# ----------------------------
@st.cache_resource()
def load_resources():
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float32)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    index = faiss.read_index("aggregated_clip.index")
    with open("place_mapping.json", "r") as f:
        place_mapping = json.load(f)
    with open("places_order.json", "r") as f:
        places_order = json.load(f)
    return device, model, processor, index, place_mapping, places_order

device, model_clip, processor_clip, index_place, place_mapping, places_order = load_resources()

# ----------------------------
# API and Place Details Functions
# ----------------------------
def geocode_place(place_name):
    api_key = st.secrets["GOOGLE_MAPS_API_KEY"]
    geocode_url = "https://maps.gomaps.pro/maps/api/geocode/json"
    params = {"address": place_name, "key": api_key}
    response = requests.get(geocode_url, params=params)
    data = response.json()
    if data.get("status") != "OK":
        st.error(f"Error geocoding {place_name}: {data.get('error_message', 'Unknown error')}")
        return None
    return data["results"][0]["geometry"]["location"]

def get_basic_places(location, radius, place_types):
    api_key = st.secrets["GOOGLE_MAPS_API_KEY"]
    all_places = []
    for place_type in place_types:
        url = "https://maps.gomaps.pro/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{location['lat']},{location['lng']}",
            "radius": radius,
            "type": place_type,
            "key": api_key
        }
        while True:
            response = requests.get(url, params=params)
            data = response.json()
            if data.get("status") != "OK":
                break
            all_places.extend([{
                "place_id": p["place_id"],
                "name": p.get("name"),
                "types": p.get("types", []),
                "geometry": p.get("geometry"),
                "vicinity": p.get("vicinity")
            } for p in data.get("results", [])])
            if "next_page_token" in data:
                params["pagetoken"] = data["next_page_token"]
                time.sleep(2)
            else:
                break
    return all_places

def get_detailed_place(place_id):
    api_key = st.secrets["GOOGLE_MAPS_API_KEY"]
    url = "https://maps.gomaps.pro/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,formatted_address,opening_hours,rating,user_ratings_total,types,geometry",
        "key": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data.get("status") == "OK":
        result = data.get("result", {})
        return {
            "place_id": place_id,
            "name": result.get("name"),
            "types": result.get("types", []),
            "geometry": result.get("geometry", {}),
            "formatted_address": result.get("formatted_address"),
            "opening_hours": result.get("opening_hours", {}),
            "rating": result.get("rating"),
            "user_ratings_total": result.get("user_ratings_total")
        }
    return None

# ----------------------------
# Core Algorithm Functions
# ----------------------------
def create_distance_matrix(places, speed=500):
    size = len(places)
    matrix = [[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i != j:
                distance = haversine(
                    places[i]["geometry"]["location"]["lat"],
                    places[i]["geometry"]["location"]["lng"],
                    places[j]["geometry"]["location"]["lat"],
                    places[j]["geometry"]["location"]["lng"]
                )
                travel_time = distance / speed
                matrix[i][j] = int(travel_time)
    return matrix

def parse_opening_hours(place):
    periods = place.get("opening_hours", {}).get("periods", [])
    today = datetime.datetime.today().weekday()
    for period in periods:
        open_info = period.get("open", {})
        close_info = period.get("close", {})
        if open_info.get("day") == today and close_info:
            try:
                open_time = int(open_info["time"])
                close_time = int(close_info["time"])
                if open_time < close_time:
                    return (
                        (open_time // 100)*60 + (open_time % 100),
                        (close_time // 100)*60 + (close_time % 100)
                    )
            except Exception:
                continue
    return (540, 1020)  # Default 9:00-17:00

def create_time_aware_data_model(places, available_time, day_start_minutes):
    visit_duration = 60  # minutes per place
    time_matrix = create_distance_matrix(places)
    time_windows = []
    meal_assignments = {}
    
    eateries = [idx for idx, p in enumerate(places) if idx != 0 and 
                any(t in p.get("types", []) for t in ["restaurant", "cafe", "bakery", "meal_takeaway"])]
    
    for meal, (m_start, m_end) in MEAL_SLOTS.items():
        best_eatery = None
        best_score = -1
        for idx in eateries:
            place = places[idx]
            raw_window = parse_opening_hours(place)
            if raw_window[0] <= m_start and raw_window[1] >= m_end:
                rating = place.get("rating", 3.0)
                reviews = math.log(place.get("user_ratings_total", 1))
                score = rating * reviews
                if score > best_score:
                    best_score = score
                    best_eatery = idx
        if best_eatery is not None:
            meal_assignments[meal] = best_eatery
            time_windows.insert(best_eatery, (m_start - day_start_minutes, m_end - day_start_minutes))
            eateries.remove(best_eatery)
    
    for idx in range(len(places)):
        if idx == 0:  # Depot/lodging
            time_windows.append((0, available_time))
            continue
        if idx in meal_assignments.values():
            continue
        raw_window = parse_opening_hours(places[idx])
        adjusted_window = (
            max(0, raw_window[0] - day_start_minutes),
            max(0, raw_window[1] - day_start_minutes)
        )
        if adjusted_window[1] - adjusted_window[0] < visit_duration:
            adjusted_window = (adjusted_window[0], adjusted_window[0] + visit_duration + 30)
        time_windows.insert(idx, adjusted_window)
    
    data = {
        "places": places,
        "time_matrix": time_matrix,
        "time_windows": time_windows,
        "num_vehicles": 1,
        "depot": 0,
        "visit_duration": visit_duration,
        "available_time": available_time,
        "meal_assignments": meal_assignments
    }
    return data

def solve_prize_collecting_vrptw(data, penalties):
    num_nodes = len(data["time_matrix"])
    manager = pywrapcp.RoutingIndexManager(num_nodes, data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)
    
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["time_matrix"][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    time_dimension_name = "Time"
    routing.AddDimension(
        transit_callback_index,
        60,
        data["available_time"],
        False,
        time_dimension_name
    )
    time_dimension = routing.GetDimensionOrDie(time_dimension_name)
    
    for node in range(1, num_nodes):
        index = manager.NodeToIndex(node)
        time_dimension.SlackVar(index).SetValue(data["visit_duration"])
    
    for location_idx, window in enumerate(data["time_windows"]):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(window[0], window[1])
    
    for node in range(1, num_nodes):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalties[node])
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(10)
    
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        route, schedule = [], []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            schedule.append(solution.Min(time_dimension.CumulVar(index)))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        schedule.append(solution.Min(time_dimension.CumulVar(index)))
        return route, solution.ObjectiveValue(), schedule
    return None, None, None

# ----------------------------
# Image Search Function
# ----------------------------
def search_places(query_files, top_k=5):
    query_embeddings = []
    with torch.inference_mode():
        for file in query_files:
            try:
                image = Image.open(file).convert("RGB")
                inputs = processor_clip(images=image, return_tensors="pt").to(device)
                embedding = model_clip.get_image_features(**inputs)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                query_embeddings.append(embedding.cpu().numpy())
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
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

def get_marker_style(place):
    types = place.get("types", [])
    for t in types:
        if t in place_styles:
            return place_styles[t]
    return {"icon": "map-marker", "color": "blue"}

# ----------------------------
# UI State Initialization
# ----------------------------
def initialize_session_state():
    if 'selected_place' not in st.session_state:
        st.session_state.selected_place = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'dest_coords' not in st.session_state:
        st.session_state.dest_coords = None
    if 'basic_places' not in st.session_state:
        st.session_state.basic_places = None
    if 'detailed_places' not in st.session_state:
        st.session_state.detailed_places = None
    if 'daily_itineraries' not in st.session_state:
        st.session_state.daily_itineraries = None

initialize_session_state()

# ----------------------------
# Common Map Rendering Function (using folium.Icon)
# ----------------------------
def render_map(center, markers, polyline=None, width=1200, height=600):
    m = folium.Map(location=[center['lat'], center['lng']], zoom_start=13)
    for marker in markers:
        style = get_marker_style(marker["place"])
        popup_text = marker["popup"]
        folium.Marker(
            location=marker["location"],
            popup=popup_text,
            tooltip=marker["tooltip"],
            icon=folium.Icon(color=style["color"], icon=style["icon"], prefix="fa")
        ).add_to(m)
    if polyline:
        folium.PolyLine(locations=polyline, color="blue", weight=5, opacity=0.7).add_to(m)
    st_folium(m, width=width, height=height)

# ----------------------------
# UI Tabs Rendering Functions
# ----------------------------
def render_tab1():
    st.title("Image-based Place Search")
    uploaded_files = st.sidebar.file_uploader("Upload travel photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files and st.button("Search Similar Places"):
        with st.spinner("Analyzing images..."):
            results = search_places(uploaded_files)
            st.session_state.search_results = results

    if st.session_state.search_results:
        st.subheader("Search Results")
        for i, result in enumerate(st.session_state.search_results):
            cols = st.columns(5)
            for j, (col, place) in enumerate(zip(cols, result)):
                with col:
                    st.image(place["rep_image"], use_container_width=True)
                    # Use both i and j to create a unique key for each button.
                    if st.button(f"Select {place['place'].replace('_', ' ')}", key=f"select_{i}_{j}_{place['place']}"):
                        st.session_state.selected_place = place['place'].replace('_', ' ')
                        # Reset dependent state so new coordinates and places are fetched.
                        st.session_state.dest_coords = None
                        st.session_state.basic_places = None
                        st.session_state.detailed_places = None
                        st.session_state.daily_itineraries = None
                        st.rerun()

    st.subheader("Manual Destination Input")
    manual_dest = st.text_input("Or enter destination manually:")
    if manual_dest:
        if manual_dest != st.session_state.selected_place:
            st.session_state.selected_place = manual_dest
            st.session_state.dest_coords = None
            st.session_state.basic_places = None
            st.session_state.detailed_places = None
            st.session_state.daily_itineraries = None
            st.rerun()

def render_tab2():
    if st.session_state.selected_place:
        st.title(f"Places near {st.session_state.selected_place}")
        if not st.session_state.dest_coords:
            with st.spinner("Locating destination..."):
                st.session_state.dest_coords = geocode_place(st.session_state.selected_place)
        if st.session_state.dest_coords:
            radius = st.slider("Search radius (km)", 1, 50, 5) * 1000
            selected_types = st.multiselect("Place types", ALL_PLACE_TYPES, default=["tourist_attraction", "park", "museum", "lodging", "restaurant"])
            if st.button("Discover Places"):
                with st.spinner("Finding nearby places..."):
                    basic_places = get_basic_places(st.session_state.dest_coords, radius, selected_types)
                    st.session_state.basic_places = basic_places
            if st.session_state.basic_places:
                st.subheader(f"Found {len(st.session_state.basic_places)} places")
                # Show map first
                markers = []
                for place in st.session_state.basic_places:
                    lat = place["geometry"]["location"]["lat"]
                    lng = place["geometry"]["location"]["lng"]
                    markers.append({
                        "location": [lat, lng],
                        "popup": f"<b>{place['name']}</b><br><i>Types:</i> {'<br>'.join(place.get('types', []))}<br><i>Address:</i> {place.get('vicinity', 'Unknown')}",
                        "tooltip": place["name"],
                        "place": place
                    })
                render_map(st.session_state.dest_coords, markers)
                # Then show table listing
                table_data = []
                for place in st.session_state.basic_places:
                    table_data.append({
                        "Name": place.get("name"),
                        "Types": ", ".join(place.get("types", [])),
                        "Vicinity": place.get("vicinity", "Unknown")
                    })
                st.markdown("### Places Listing")
                st.table(pd.DataFrame(table_data))

def render_tab3():
    st.title("Generate Detailed Itinerary")
    col1, col2 = st.columns(2)
    with col1:
        # start_date = st.date_input("Trip Start Date", datetime.date.today())
        start_date = st.date_input("Trip Start Date", datetime.date.today() + datetime.timedelta(days=1))
        day_start = st.time_input("Daily Start Time", datetime.time(9, 0))
    with col2:
        end_date = st.date_input("Trip End Date", datetime.date.today() + datetime.timedelta(days=3))
        day_end = st.time_input("Daily End Time", datetime.time(20, 0))
    generate_col, clear_col = st.columns([3, 1])
    with generate_col:
        generate_enabled = st.session_state.basic_places is not None and st.session_state.dest_coords is not None
        if st.button("✨ Generate Optimized Itinerary", disabled=not generate_enabled, help="Requires completed Place Discovery"):
            with st.spinner("🧭 Building optimal route..."):
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(get_detailed_place, p["place_id"]) for p in st.session_state.basic_places]
                        detailed_places = [f.result() for f in concurrent.futures.as_completed(futures)]
                    detailed_places = [p for p in detailed_places if p is not None]
                    st.session_state.detailed_places = detailed_places
                    lodging = next((p for p in detailed_places if "lodging" in p.get("types", [])), None)
                    if not lodging:
                        st.warning("⚠️ No lodging found - using first place as start/end point")
                        lodging = detailed_places[0]
                    total_days = (end_date - start_date).days + 1
                    day_start_minutes = day_start.hour * 60 + day_start.minute
                    daily_minutes = (day_end.hour * 60 + day_end.minute) - day_start_minutes
                    daily_itineraries = []
                    remaining_places = [p for p in detailed_places if p != lodging]
                    for day_num in range(total_days):
                        if not remaining_places:
                            break
                        day_places = [lodging] + remaining_places[:10]
                        penalties = [0] + [int((5 - (p.get("rating") or 3.0)) * 1000) for p in day_places[1:]]
                        data = create_time_aware_data_model(day_places, daily_minutes, day_start_minutes)
                        route, cost, schedule = solve_prize_collecting_vrptw(data, penalties)
                        if route and len(route) > 2:
                            valid_route = [i for i in route if i != 0][:-1]
                            visited_indices = [i-1 for i in valid_route if i > 0]
                            daily_itineraries.append({
                                "date": start_date + datetime.timedelta(days=day_num),
                                "places": [day_places[i] for i in route],
                                "schedule": schedule,
                                "data_model": data
                            })
                            remaining_places = [p for idx, p in enumerate(remaining_places) if idx not in visited_indices]
                    st.session_state.daily_itineraries = daily_itineraries
                    st.rerun()
                except Exception as e:
                    st.error(f"🚨 Itinerary generation failed: {str(e)}")
    with clear_col:
        if st.button("🧹 Clear Itinerary"):
            st.session_state.daily_itineraries = None
            st.rerun()
    if st.session_state.daily_itineraries:
        st.success("✅ Itinerary Generated Successfully!")
        for day_idx, day_plan in enumerate(st.session_state.daily_itineraries):
            with st.expander(f"📅 Day {day_idx+1}: {day_plan['date'].strftime('%A, %b %d')}", expanded=True):
                lodging = next((p for p in day_plan['places'] if "lodging" in p.get("types", [])), None)
                if lodging:
                    st.markdown(f"""
                    ### 🏨 Overnight Stay
                    **{lodging['name']}**  
                    {lodging.get('formatted_address', '')}
                    """)
                map_col, timeline_col = st.columns([1, 2])
                with map_col:
                    itinerary_coords = []
                    markers = []
                    for idx, place in enumerate(day_plan['places']):
                        try:
                            lat = place["geometry"]["location"]["lat"]
                            lng = place["geometry"]["location"]["lng"]
                            itinerary_coords.append((lat, lng))
                            popup_text = f"Stop {idx+1}: {place['name']}"
                            if idx == 0:
                                popup_text = f"Start / Lodging: {place['name']}"
                            elif idx == len(day_plan['places']) - 1 and place == lodging:
                                popup_text = f"Return to Lodging: {place['name']}"
                            markers.append({
                                "location": [lat, lng],
                                "popup": popup_text,
                                "tooltip": f"Stop {idx+1}",
                                "place": place
                            })
                        except KeyError:
                            continue
                    render_map(st.session_state.dest_coords, markers, polyline=itinerary_coords, width=400, height=400)
                with timeline_col:
                    st.markdown("### 🕒 Daily Schedule")
                    day_start_dt = datetime.datetime.combine(day_plan['date'], datetime.time(day_start.hour, day_start.minute))
                    prev_end_time = day_start_dt
                    for idx, (place, arrival_min) in enumerate(zip(day_plan['places'], day_plan['schedule'])):
                        if idx == 0 or (idx == len(day_plan['places'])-1 and place == lodging):
                            continue
                        arrival_time = day_start_dt + datetime.timedelta(minutes=arrival_min)
                        departure_time = arrival_time + datetime.timedelta(minutes=day_plan['data_model']['visit_duration'])
                        travel_time = arrival_time - prev_end_time if idx > 1 else datetime.timedelta(0)
                        travel_min = int(travel_time.total_seconds() // 60) if travel_time.total_seconds() > 0 else 0
                        meal_type = next((meal for meal, p_idx in day_plan['data_model']['meal_assignments'].items() if p_idx == idx), None)
                        purpose = get_purpose(place)
                        details = []
                        if place.get('rating'):
                            details.append(f"⭐ {place['rating']} ({place.get('user_ratings_total', '?')} reviews)")
                        if place.get('opening_hours'):
                            details.append(f"🕒 {place['opening_hours'].get('weekday_text', [''])[0]}")
                        step_html = f"""
                        <div style="border: 1px solid #ddd; border-radius:5px; padding: 10px; margin-bottom:10px;">
                          <strong>{arrival_time.strftime('%H:%M')} to {departure_time.strftime('%H:%M')}</strong><br>
                          {"🍴 <strong>" + meal_type + " Break</strong> - " + place['name'] if meal_type else "📍 <strong>" + place['name'] + "</strong> - " + purpose}<br>
                          {"🚗 " + str(travel_min) + " min travel from previous" if travel_time.total_seconds() > 0 else ""}<br>
                          {" • ".join(details) if details else ""}
                        </div>
                        """
                        st.markdown(step_html, unsafe_allow_html=True)
                        prev_end_time = departure_time
                    st.markdown(f"🏁 <strong>{prev_end_time.strftime('%H:%M')}</strong> - Return to: 🏨 {lodging['name']}", unsafe_allow_html=True)

# ----------------------------
# Main Application with Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Image Search", "🗺️ Place Discovery", "📅 Itinerary Planner"])
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
