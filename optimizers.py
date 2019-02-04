import heapq
import numpy as np
import random
import tsp


def greedy(problem):
    soln = problem.solution_type(problem)
    while not soln.complete:
        target_value = np.min(problem.costs[soln.feasible_edges])
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

def genetic(problem, popsize, n_generations, n_parents=2, tournament_size=2,
        n_elites=2):
    dim = problem.dimension
    edge_indices = np.arange(dim*dim).reshape((dim, dim))
    def tournament_select(values, tournament_size):
        best = random.randrange(len(values))
        for i in range(1, tournament_size):
            challenger = random.randrange(len(values))
            if values[challenger] < values[best]:
                best = challenger
        return best
    def complete_randomly(soln):
        while not soln.complete:
            total = np.sum(soln.feasible_edges)
            chosen_edge = np.random.choice(
                dim*dim, p=soln.feasible_edges.flatten()/total
            )
            soln.add_edge(chosen_edge // dim, chosen_edge % dim)
        return soln
    def crossover_with_mutation(parents, p_mutation):
        # Algebra gets us a background that yields p_mutation% mutations.
        background = p_mutation*2*dim/(dim*dim - p_mutation*dim*dim)
        background = max(background, np.finfo(np.float32).eps)
        selection_odds = np.full(
            (dim, dim), background, dtype=np.float32
        )
        for parent in parents:
            selection_odds += parent.edges_added
        result = problem.solution_type(problem)
        while not result.complete:
            this_selection_odds = selection_odds * result.feasible_edges
            total = np.sum(this_selection_odds)
            chosen_edge = np.random.choice(
                dim*dim, p=this_selection_odds.reshape((-1,))/total
            )
            result.add_edge(chosen_edge // dim, chosen_edge % dim)
        return result
    pop = [complete_randomly(problem.solution_type(problem))
           for i in range(popsize)]
    mutation_schedule = np.linspace(0.3, 0, n_generations)
    elites = []
    for i in range(n_elites):
        elite = complete_randomly(problem.solution_type(problem))
        heapq.heappush(elites, (-elite.cost, elite))
    pop.extend([elite[1] for elite in elites])
    for p_mutation in mutation_schedule:
        print(p_mutation)
        scores = [soln.cost for soln in pop]
        kids = []
        while len(kids) < popsize:
            parents = [
                pop[tournament_select(scores, tournament_size)]
                for i in range(n_parents)
            ]
            kid = crossover_with_mutation(parents, p_mutation)
            kids.append(kid)
            kid_cost = kid.cost
            if kid_cost < -elites[0][0]:
                print(kid_cost)
                heapq.heapreplace(elites, (-kid_cost, kid))
        pop = kids
        pop.extend([elite[1] for elite in elites])
    return max(elites)[1]
