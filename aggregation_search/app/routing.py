import datetime
import math # For math.log in create_time_aware_data_model
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .utils import haversine # Relative import
from .constants import MEAL_SLOTS # Relative import


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
