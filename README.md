# PCSE25-53
# Padharo Sa: Tourism Itinerary Generation Based on Image Similarity

**Project Report submitted as partial fulfillment for the award of BACHELOR OF TECHNOLOGY DEGREE (Session 2024-25) in Computer Science and Engineering.**

By:
*   Rovince Gangwar (2100290100139)
*   Yash Goswami (2100290100195)
*   Mohd. Uzair Khan (2100290100099)

Under the supervision of: Dr. Seema Maitrey
KIET Group of Institutions, Ghaziabad
Affiliated to Dr. A.P.J. Abdul Kalam Technical University, Lucknow

---

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Features](#features)
3.  [System Architecture](#system-architecture)
4.  [Repository Structure](#repository-structure)
5.  [Setup and Installation](#setup-and-installation)
6.  [Running the Project](#running-the-project)
    *   [Step 1: Embedding Generation (One-time setup)](#step-1-embedding-generation-one-time-setup)
    *   [Step 2: Running the Main Application](#step-2-running-the-main-application)
7.  [Key Technologies Used](#key-technologies-used)
8.  [Project Documents](#project-documents)

---

## 1. Project Overview

"Padharo Sa" is an innovative tourism itinerary generation system that leverages the power of visual similarity. Users can upload an image representing their desired travel aesthetic or a specific destination vibe. The system then uses advanced computer vision (CLIP model) and similarity search techniques (FAISS) to identify visually similar tourist attractions from a database. 

Once a primary location is identified and selected, the system integrates with the Google Maps API to gather details about nearby attractions and optimize a multi-day itinerary using Google OR-Tools (solving a Vehicle Routing Problem with Time Windows - VRPTW). The final itinerary is presented to the user via an interactive Streamlit web application, complete with a map visualization (Folium).

The project aims to bridge the gap between visual inspiration and practical travel planning, offering a more intuitive and personalized way to discover and plan trips.

---

## 2. Features

*   **Image-Based Destination Discovery:** Upload an image to find visually similar tourist locations.
*   **Personalized Itinerary Parameters:** Customize travel dates, types of attractions, search radius, and daily travel pace.
*   **Nearby Attraction Discovery:** Utilizes Google Maps API to find relevant points of interest around a selected primary location.
*   **Optimized Itinerary Generation:** Employs Google OR-Tools to create efficient multi-day itineraries considering travel times, opening hours, and meal breaks.
*   **Interactive User Interface:** Built with Streamlit for easy image upload, preference setting, and itinerary viewing.
*   **Map Visualization:** Displays the generated itinerary on an interactive Folium map.

---

## 3. System Architecture

The system follows a multi-stage pipeline:

1.  **Image Input & Preprocessing:** User uploads an image. (Optional: Homographic warping for image refinement).
2.  **Embedding Generation (Query):** The input image is processed by the CLIP model to generate a high-dimensional embedding vector.
3.  **Similarity Search:** The query embedding is compared against a precomputed FAISS index of location embeddings (centroids) using cosine similarity.
4.  **Location Selection & Customization:** User selects a primary destination from the visual matches and provides travel preferences.
5.  **Data Enrichment & Nearby POI Discovery:** Google Maps API is used to geocode locations, fetch details (ratings, opening hours), and find nearby relevant attractions.
6.  **Itinerary Optimization (VRPTW):** Google OR-Tools solves the Vehicle Routing Problem with Time Windows to create daily schedules, considering travel times, service durations, and constraints.
7.  **Output Presentation:** The final itinerary is displayed in the Streamlit app with an interactive Folium map.

---

## 4. Repository Structure

```
.
├── aggregation_search/
│   ├── embedding_gen.py            # Script to generate embeddings and FAISS index for location images
│   ├── final_app.py                # Main Streamlit application script
│   ├── requirements.txt            # Python dependencies
│   ├── aggregated_clip.index       # (Generated) FAISS index for location embeddings
│   ├── place_mapping.json          # (Generated) Maps place names to representative image paths
│   └── places_order.json           # (Generated) Ordered list of place names for FAISS index mapping
│
├── ICETISD-2025/                   # Directory for conference paper
│   └── [Research Paper Document Name].pdf # Example: ICETISD_Paper_PadharoSa.pdf (Actual name may vary)
│
├── final_project_report/
│   └── [Project Report Document Name].pdf # Example: BTech_Project_Report_PadharoSa.pdf (Actual name may vary)
│
├── Presentation certificate Rovince Gangwar/
│   └── [Certificate File Name].pdf   # Example: Certificate_Rovince_Gangwar_ICETISD.pdf (Actual name may vary)
│
├── Tourism_Itinerary_Generation.pptx # Conference presentation slides
│
├── places_images/
│   └── 50 most famous places in the world.zip/ # Zipped directory containing images
│       ├── [Place Name 1]/
│       │   ├── image1.jpg
│       │   └── ...
│       ├── [Place Name 2]/
│       │   ├── image1.jpg
│       │   └── ...
│       └── ... (directories for all 50 places)
│   └── Pseudo marking/                   # Directory containing unused .csv marking files
│       ├── file1.csv
│       └── ...
│
└── README.md                           # This file
```

**Note on Generated Files:**
The files `aggregated_clip.index`, `place_mapping.json`, and `places_order.json` inside the `aggregation_search/` directory are generated by the `embedding_gen.py` script. They are essential for the `final_app.py` to run. If you are cloning this repository, these files might be included if they were committed. If not, you will need to run `embedding_gen.py` first.

---

## 5. Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/padharo-sa.git # Replace with your actual repo URL
    cd padharo-sa
    ```

2.  **Set up a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Navigate to the `aggregation_search` directory and install the required Python packages.
    ```bash
    cd aggregation_search
    pip install -r requirements.txt
    ```
    **Note:** `requirements.txt` should include libraries like `torch`, `transformers`, `faiss-cpu` (or `faiss-gpu` if you have a compatible GPU and CUDA setup), `streamlit`, `folium`, `numpy`, `Pillow`, `googlemaps`, `ortools`, etc.

4.  **Prepare Image Dataset:**
    *   Navigate to the `places_images/` directory from the root of the project.
    *   Unzip the `50 most famous places in the world.zip` file.
    *   This should create a directory structure like `places_images/50 most famous places in the world/[Place Name]/image.jpg`.
    *   The `embedding_gen.py` script expects the images to be organized in subdirectories where each subdirectory's name is the name of the tourist place.
        *   **IMPORTANT:** The `embedding_gen.py` script in its current form (as per the provided code) uses `base_folder = os.path.join("..", "places_images")`. This means it expects the `places_images` directory to be one level *above* the `aggregation_search` directory (i.e., in the project root). Ensure your `places_images` folder containing the unzipped place image directories is at the root level of your project. If you unzipped it to `places_images/50 most famous places in the world/`, you might need to adjust the `base_folder` path in `embedding_gen.py` to point correctly to the directories containing the actual images (e.g., `os.path.join("..", "places_images", "50 most famous places in the world")`).

5.  **Google Maps API Key:**
    *   The `final_app.py` script will require a Google Maps API key for geocoding, finding nearby places, and potentially for fetching travel times.
    *   You will need to obtain an API key from the Google Cloud Platform Console. Make sure to enable the following APIs for your key:
        *   Geocoding API
        *   Places API
        *   Directions API (if used for travel times)
    *   You'll need to securely manage this API key. The application might expect it as an environment variable or a configuration file (this detail should be in `final_app.py`'s implementation). For example, you might need to set an environment variable:
        ```bash
        # On Linux/macOS
        export GOOGLE_MAPS_API_KEY="YOUR_API_KEY"
        # On Windows (PowerShell)
        $env:GOOGLE_MAPS_API_KEY="YOUR_API_KEY"
        ```
        Or modify `final_app.py` to read it from a config file. **Do not commit your API key directly into the code.**

---

## 6. Running the Project

### Step 1: Embedding Generation (One-time setup)

This step processes all the images in your `places_images` dataset, generates CLIP embeddings for them, aggregates them per location, and creates the FAISS index and mapping files. **You only need to run this once**, or whenever your image dataset changes.

1.  Ensure you are in the `aggregation_search/` directory in your terminal.
2.  Run the `embedding_gen.py` script:
    ```bash
    python embedding_gen.py
    ```
3.  This script will:
    *   Scan the image dataset (path configured within the script, likely `../places_images/`).
    *   Generate embeddings for all images (this can take a while depending on the number of images and your hardware).
    *   Create `aggregated_clip.index` (the FAISS index).
    *   Create `place_mapping.json` (maps places to representative images).
    *   Create `places_order.json` (ordered list of places for the index).

    These three files will be saved in the `aggregation_search/` directory.

### Step 2: Running the Main Application

Once the embeddings and index are generated (or if they are already present), you can run the Streamlit web application.

1.  Ensure you are in the `aggregation_search/` directory.
2.  Make sure your Google Maps API key is correctly set up (see Setup section).
3.  Run the `final_app.py` script using Streamlit:
    ```bash
    streamlit run final_app.py
    ```
4.  This will start a local web server, and your default web browser should open automatically to the application's URL (usually `http://localhost:8501`).
5.  You can now interact with the application:
    *   Upload an image.
    *   View visually similar locations.
    *   Select a primary location and customize itinerary preferences.
    *   Generate and view the optimized itinerary with an interactive map.

---

## 7. Key Technologies Used

*   **Python:** Core programming language.
*   **Streamlit:** For building the interactive web application UI.
*   **Hugging Face Transformers:** For accessing the pre-trained CLIP model.
*   **PyTorch:** Deep learning framework used by the CLIP model.
*   **FAISS (Facebook AI Similarity Search):** For efficient similarity search of high-dimensional embeddings.
*   **NumPy:** For numerical operations, especially with embeddings.
*   **Pillow (PIL):** For image processing.
*   **Google OR-Tools:** For solving the Vehicle Routing Problem with Time Windows (VRPTW) to optimize itineraries.
*   **Google Maps API:** For geocoding, fetching place details (ratings, opening hours), finding nearby places, and route information.
*   **Folium:** For creating interactive map visualizations.
*   **JSON:** For storing mapping data.

---

## 8. Project Documents

This repository also contains important documentation related to the project:

*   **`final_project_report/`**: Contains the detailed B.Tech project report (`[Project Report Document Name].pdf`). This document provides an in-depth explanation of the project's background, methodology, implementation, results, and conclusions.
*   **`ICETISD-2025/`**: Contains the research paper (`[Research Paper Document Name].pdf`) submitted/presented at the ICETISD-2025 conference.
*   **`Presentation certificate Rovince Gangwar/`**: Contains the certificate (`[Certificate File Name].pdf`) received for the conference presentation.
*   **`Tourism_Itinerary_Generation.pptx`**: The PowerPoint presentation slides used for the conference or project defense.

These documents provide comprehensive insights into the academic and research aspects of the "Padharo Sa" project.

---

For any issues or contributions, please open an issue or a pull request in this repository.