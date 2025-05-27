import math
from PIL import Image


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


def resize_image(image, max_width=800, max_height=800):
    width, height = image.size
    aspect_ratio = width / height
    
    if width > max_width or height > max_height:
        if width > height:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        return image.resize((new_width, new_height))
    else:
        return image
