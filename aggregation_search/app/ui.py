import streamlit as st
import datetime
import pandas as pd
from PIL import Image # For Image.open in render_tab1
import cv2 # For homography in render_tab1
import numpy as np # For homography in render_tab1
from streamlit_image_coordinates import streamlit_image_coordinates
import folium
from streamlit_folium import st_folium
import concurrent.futures # For render_tab3

from . import resources # For get_marker_style
from . import gmaps_api
from . import routing
from . import image_search # For search_places
from . import utils # For resize_image, get_purpose
from .constants import ALL_PLACE_TYPES # For render_tab2


# ----------------------------
# UI State Initialization
# ----------------------------
def initialize_session_state():

    if 'image_points' not in st.session_state:
        st.session_state.image_points = {}
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

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


def render_map(center, markers, polyline=None, width=1200, height=600):
    m = folium.Map(location=[center['lat'], center['lng']], zoom_start=13)
    for marker in markers:
        style = resources.get_marker_style(marker["place"])
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
    uploaded_files = st.sidebar.file_uploader("Upload travel photos", 
                                            type=["jpg", "jpeg", "png"], 
                                            accept_multiple_files=True)
    
    # Homography controls
    warp_enabled = st.checkbox("Enable Perspective Correction (Select 4 points)")
    if warp_enabled and uploaded_files:
        st.info("🔍 Click on images in order: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
        
        # Create tabs for each image
        tabs = st.tabs([f"Image {i+1}" for i in range(len(uploaded_files))])
        
        for i, (tab, file) in enumerate(zip(tabs, uploaded_files)):
            with tab:
                col1, col2 = tab.columns(2)
                with col1:
                    if file.name not in st.session_state.image_points:
                        st.session_state.image_points[file.name] = []
                    
                    points = st.session_state.image_points[file.name]
                    img = Image.open(file)
                    
                    img = utils.resize_image(img, max_width=800, max_height=800)
                    
                    # Display image with click coordinates
                    value = streamlit_image_coordinates(img, key=f"coord_{i}")
                    
                    if value is not None:
                        x, y = value["x"], value["y"]
                        if len(points) < 4 and (x, y) not in points:
                            points.append((x, y))
                            st.session_state.image_points[file.name] = points
                            st.rerun()
                    
                    st.caption(f"Selected points: {len(points)}/4")
                    
                with col2:
                    if len(points) == 4:
                        try:
                            # Perform homography warping preview
                            src_pts = np.float32(points)
                            width = abs(points[1][0] - points[0][0])
                            height = abs(points[3][1] - points[0][1])
                            dst_pts = np.float32([[0, 0], [width, 0], 
                                                [width, height], [0, height]])
                            
                            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                            img_warped = cv2.warpPerspective(np.array(img), matrix, 
                                                           (int(width), int(height)))
                            st.image(img_warped, caption="Warped Preview")
                        except Exception as e:
                            st.error(f"Error in warping: {str(e)}")

    # Existing search functionality with warping
    if uploaded_files and st.button("Search Similar Places"):
        with st.spinner("Analyzing images..."):
            processed_images = []
            for file in uploaded_files:
                try:
                    img = Image.open(file)
                    points = st.session_state.image_points.get(file.name, [])
                    
                    if len(points) == 4:
                        # Apply homography transformation
                        src_pts = np.float32(points)
                        width = abs(points[1][0] - points[0][0])
                        height = abs(points[3][1] - points[0][1])
                        dst_pts = np.float32([[0, 0], [width, 0], 
                                            [width, height], [0, height]])
                        
                        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        img_warped = cv2.warpPerspective(np.array(img), matrix, 
                                                       (int(width), int(height)))
                        processed_images.append(Image.fromarray(img_warped))
                    else:
                        processed_images.append(img)
                
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
                    processed_images.append(img)
            
            # Perform search with processed images
            results = image_search.search_places(uploaded_files, processed_images)
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
                st.session_state.dest_coords = gmaps_api.geocode_place(st.session_state.selected_place)
        if st.session_state.dest_coords:
            radius = st.slider("Search radius (km)", 1, 50, 5) * 1000
            selected_types = st.multiselect("Place types", ALL_PLACE_TYPES, default=["tourist_attraction", "park", "museum", "lodging", "restaurant"])
            if st.button("Discover Places"):
                with st.spinner("Finding nearby places..."):
                    basic_places = gmaps_api.get_basic_places(st.session_state.dest_coords, radius, selected_types)
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
                        futures = [executor.submit(gmaps_api.get_detailed_place, p["place_id"]) for p in st.session_state.basic_places]
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
                        data = routing.create_time_aware_data_model(day_places, daily_minutes, day_start_minutes)
                        route, cost, schedule = routing.solve_prize_collecting_vrptw(data, penalties)
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
                        purpose = utils.get_purpose(place)
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
