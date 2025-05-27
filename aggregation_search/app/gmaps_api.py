import streamlit as st # For st.secrets
import requests
import time
import math


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