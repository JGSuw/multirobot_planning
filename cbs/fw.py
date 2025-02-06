
from collections import defaultdict
import heapq
import numpy as np
import networkx as nx

def frank_wolfe_step(game):
    new_flows = defaultdict(float)
    
    for (start, end), demand in game.demands.items():
        if demand > 0:
            path = nx.dijkstra_path(game.net, start, end, 'cost')
            if path:
                for i in range(len(path) - 1):
                    key = (path[i], path[i+1])
                    new_flows[key] += demand 
    return new_flows

def run_frank_wolfe(game, 
                    max_iterations=50, 
                    sample_rate=1, 
                    ftol_abs = 1., 
                    ftol_rel = 1e-3):
    costs = []
    last_cost = np.inf
    for iteration in range(max_iterations):

        new_flows = frank_wolfe_step(game)
        
        step_size = 2 / (iteration + 2)
        
        for edge, attrs in game.net.edges.items():
            old_flow = attrs['flow']
            key = edge
            attrs['flow'] = (1 - step_size) * old_flow + step_size * new_flows.get(key, 0)
        
        game.update_edge_costs()
        current_cost = game.total_system_cost()
        costs.append(current_cost)
        
        print(f"Iteration {iteration + 1}: Total System Cost = {current_cost:.4f}")

        if abs(current_cost - last_cost) < ftol_abs:
            print(f'Frank-Wolfe Terminated due to ftol_abs at iteration {iteration+1}')
            break

        if abs(current_cost - last_cost)/last_cost < ftol_rel:
            print(f'Frank-Wolfe Terminated due to ftol_rel at iteration {iteration+1}')
            break

        last_cost = current_cost

    return costs