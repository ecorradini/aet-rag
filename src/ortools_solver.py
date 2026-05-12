"""OR-Tools wrapper for CVRPTW used by all three synchronization scenarios."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    ORTOOLS_AVAILABLE = True
except Exception:                                                # pragma: no cover
    ORTOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SolverResult:
    routes: List[List[int]]
    arrival_times: Dict[int, float]
    edge_to_vehicle: Dict[Tuple[int, int], int]
    objective: float
    total_travel: float
    total_lateness: float
    unserved: int
    runtime_seconds: float
    feasible: bool


def _build_data_model(
    travel_time: np.ndarray,
    instance,
) -> dict:
    return {
        "time_matrix": np.rint(travel_time).astype(int).tolist(),
        "time_windows": [(int(a), int(b)) for a, b in instance.time_windows],
        "service_time": [int(s) for s in instance.service_times],
        "demands": [int(d) for d in instance.demands],
        "vehicle_capacities": [int(instance.vehicle_capacity)] * instance.num_vehicles,
        "num_vehicles": int(instance.num_vehicles),
        "depot": int(instance.depot),
    }


def solve(
    instance,
    travel_time: np.ndarray,
    time_limit_seconds: int = 30,
    first_solution: str = "PATH_CHEAPEST_ARC",
    metaheuristic: str = "GUIDED_LOCAL_SEARCH",
) -> SolverResult:
    if not ORTOOLS_AVAILABLE:
        raise RuntimeError("ortools is not installed; please pip install -r requirements.txt")

    data = _build_data_model(travel_time, instance)
    manager = pywrapcp.RoutingIndexManager(
        len(data["time_matrix"]), data["num_vehicles"], data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return data["time_matrix"][i][j] + data["service_time"][i]

    transit_idx = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    horizon = max(b for _, b in data["time_windows"]) + 60
    routing.AddDimension(
        transit_idx,
        slack_max=horizon,
        capacity=horizon,
        fix_start_cumul_to_zero=False,
        name="Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")
    for node, (a, b) in enumerate(data["time_windows"]):
        if node == data["depot"]:
            continue
        index = manager.NodeToIndex(node)
        time_dim.CumulVar(index).SetRange(int(a), int(b))
    for v in range(data["num_vehicles"]):
        index = routing.Start(v)
        time_dim.CumulVar(index).SetRange(0, horizon)

    def demand_callback(from_index):
        return data["demands"][manager.IndexToNode(from_index)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, data["vehicle_capacities"], True, "Capacity"
    )

    # Allow customers to be dropped at a penalty (unserved penalty)
    penalty = 10_000
    for node in range(1, len(data["time_matrix"])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, first_solution
    )
    params.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic
    )
    params.time_limit.seconds = int(time_limit_seconds)

    t0 = time.perf_counter()
    solution = routing.SolveWithParameters(params)
    runtime = time.perf_counter() - t0
    if solution is None:
        return SolverResult([], {}, {}, float("inf"), 0, 0, instance.n_customers, runtime, False)

    routes: List[List[int]] = []
    arrivals: Dict[int, float] = {}
    edge_to_v: Dict[Tuple[int, int], int] = {}
    total_travel = 0.0
    lateness = 0.0

    for v in range(data["num_vehicles"]):
        index = routing.Start(v)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            time_var = time_dim.CumulVar(index)
            arrivals[node] = float(solution.Min(time_var))
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            edge_to_v[(node, next_node)] = v
            total_travel += data["time_matrix"][node][next_node]
            index = next_index
        route.append(manager.IndexToNode(index))
        routes.append(route)

    served = set()
    for r in routes:
        for n in r[1:-1]:
            served.add(n)
    unserved = instance.n_customers - len(served)
    for n in served:
        deadline = instance.time_windows[n, 1]
        lateness += max(0.0, arrivals.get(n, 0.0) - float(deadline))

    objective = float(solution.ObjectiveValue())
    return SolverResult(
        routes=routes,
        arrival_times=arrivals,
        edge_to_vehicle=edge_to_v,
        objective=objective,
        total_travel=total_travel,
        total_lateness=lateness,
        unserved=unserved,
        runtime_seconds=runtime,
        feasible=True,
    )
