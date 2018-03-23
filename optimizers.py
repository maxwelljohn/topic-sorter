import numpy as np
import tsp


def greedy(problem):
    soln = problem.solution_type(problem)
    while soln.feasible_edges.any():
        target_value = np.min(problem.costs[soln.feasible_edges])
        assert target_value != -1
        target_edges = np.array(
            soln.feasible_edges & (problem.costs == target_value),
            dtype=np.bool
        )
        chosen_edge = np.argmax(target_edges)
        soln.add_edge(
            chosen_edge // soln.dimension,
            chosen_edge % soln.dimension
        )
    return soln
