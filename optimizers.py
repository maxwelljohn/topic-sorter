import heapq
import numpy as np
import random
import sys
import tsp

from bloomfilter import BloomFilter


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


def genetic(problem, n_commoners, n_elites, n_generations, tournament_size=2,
            mutation_rate_hist=8, n_parents=2):
    dim = problem.dimension
    edge_indices = np.arange(dim * dim).reshape((dim, dim))

    edge_costs = problem.solution_type(problem).valid_edges * problem.costs
    mean_edge_cost = np.mean(edge_costs)
    edge_cost_sd = np.std(edge_costs)
    # Softmax action selection on standardized edge costs.
    baseline_edge_odds = np.e**(-(edge_costs-mean_edge_cost)/edge_cost_sd)
    baseline_edge_odds /= np.sum(baseline_edge_odds)
    assert np.isclose(np.sum(baseline_edge_odds), 1)

    def tournament_select(values, tournament_size):
        best = random.randrange(len(values))
        for _ in range(1, tournament_size):
            challenger = random.randrange(len(values))
            if values[challenger] < values[best]:
                best = challenger
        return best

    def complete_randomly(soln, edge_odds):
        while not soln.complete:
            remaining_odds = soln.feasible_edges * edge_odds
            total = np.sum(remaining_odds)
            chosen_edges = np.random.choice(
                dim * dim, size=soln.additions_needed, replace=False,
                p=remaining_odds.flatten()/total
            )
            for edge in chosen_edges:
                node_a = edge // dim
                node_b = edge % dim
                if soln.feasible_edges[node_a, node_b]:
                    soln.add_edge(node_a, node_b)
        return soln

    def crossover_with_mutation(parents, p_mutation):
        # Algebra gets us a background that yields p_mutation% mutations.
        background = p_mutation * 2 * dim / (1 - p_mutation)
        edge_odds = background * baseline_edge_odds

        # Edges with 0 odds can't be selected, making problems unsolvable.
        # Fix this by ensuring each edge has odds of at least epsilon.
        eps = np.full(
            (dim, dim), np.finfo(np.float).eps, dtype=np.float
        )
        edge_odds = np.max(np.array([edge_odds, eps]), axis=0)

        for parent in parents:
            edge_odds += parent.edges_added
        assert np.isclose(
            p_mutation, background / np.sum(edge_odds), atol=0.1
        )

        result = problem.solution_type(problem)
        return complete_randomly(result, edge_odds)

    def tune_mutation_rate(mutation_rate_data):
        X = mutation_rate_data[:, :2]
        y = mutation_rate_data[:, 2]
        # Linear regression
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        return max(min((0.8 - theta[0]) / theta[1], 0.99), 0.01)

    seen = BloomFilter(n_generations * (n_commoners + n_elites))
    stuck_threshold = 10 * (n_commoners + n_elites)

    pop = [complete_randomly(problem.solution_type(problem),
                             baseline_edge_odds)
           for _ in range(n_commoners)]

    elites = [greedy(problem)]
    for _ in range(n_elites - 1):
        elite = complete_randomly(problem.solution_type(problem),
                                  baseline_edge_odds)
        heapq.heappush(elites, elite)
    pop.extend(elites)

    # Permanent data points at (1, 1) and (0, 0) serve as prior knowledge
    # and ensure matrix inversion is always possible during regression.
    mutation_rate_data = np.ones((mutation_rate_hist + 2, 3))
    mutation_rate_data[1, 1:] = 0
    p_mutation = 0.3

    for g in range(n_generations):
        sys.stderr.write('.')
        sys.stderr.flush()

        scores = [soln.cost for soln in pop]

        n_attempted_kids = 0
        n_failed_kids = 0
        kids = []

        while len(kids) < n_commoners:
            parents = [
                pop[tournament_select(scores, tournament_size)]
                for _ in range(n_parents)
            ]
            kid = crossover_with_mutation(parents, p_mutation)

            if kid.cost <= elites[0].cost:
                # The bloom filter is approximate.
                # Do exact dup prevention for elites.
                dup = [(elite.edges_added == kid.edges_added).all()
                       for elite in elites]
                if not any(dup):
                    heapq.heapreplace(elites, kid)
            elif repr(hash(kid)) not in seen:
                kids.append(kid)
                if kid.cost > sum([p.cost for p in parents]) / len(parents):
                    n_failed_kids += 1

            seen.add(repr(hash(kid)))
            n_attempted_kids += 1
            if n_attempted_kids > stuck_threshold:
                sys.stderr.write('\n')
                sys.stderr.write(
                    'Optimization seems stuck. Returning best so far...\n'
                )
                sys.stderr.flush()
                return max(elites)

        mutation_rate_data[2 + g % mutation_rate_hist] = \
            [1, p_mutation, n_failed_kids / n_attempted_kids]
        p_mutation = tune_mutation_rate(mutation_rate_data)

        pop = kids
        pop.extend(elites)

    sys.stderr.write('\n')
    sys.stderr.flush()

    return max(elites)  # Comparison operator overload makes cheap solns "big".
