
from collections import defaultdict
import heapq
import numpy as np

def dijkstra_shortest_path(graph, start, end, edge_costs):
    distances = {start: 0}
    previous = {}
    pq = [(0, start)] 
    visited = set()

    while pq:
        dist, current = heapq.heappop(pq)
        if current == end:  
            break
        if current in visited:  
            continue
        visited.add(current)
        
        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue
                
            edge = frozenset((current, neighbor))
            new_dist = dist + edge_costs[edge]
            
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))
    
    if end not in previous and start != end:
        return None
        
    path = []
    current = end
    while current in previous:
        path.append(current)
        current = previous[current]
    path.append(start)
    
    return path[::-1]

def frank_wolfe_step(game):
    new_flows = defaultdict(float)
    
    for (start, end), demand in game.demands.items():
        if demand > 0:

            path = dijkstra_shortest_path(game.net, start, end, game.edge_costs)
            
            if path:
                for i in range(len(path) - 1):
                    edge = frozenset((path[i], path[i+1]))
                    new_flows[edge] += demand 
    return new_flows

def run_frank_wolfe(game, 
                    max_iterations=50, 
                    sample_rate=1, 
                    ftol_abs = 1., 
                    ftol_rel = 1e-3):
    costs = []
    all_flows = []
    last_cost = np.inf
    for iteration in range(max_iterations):

        new_flows = frank_wolfe_step(game)
        
        step_size = 2 / (iteration + 2)
        
        max_diff = 0
        for edge in game.flow_state:
            old_flow = game.flow_state[edge]
            new_flow = (1 - step_size) * old_flow + step_size * new_flows.get(edge, 0)
            game.flow_state[edge] = new_flow
            max_diff = max(max_diff, abs(new_flow - old_flow))
        
        game.update_edge_costs()
        current_cost = game.total_system_cost()
        costs.append(current_cost)
        
        if iteration % sample_rate == 0:
            max_flow = max(game.flow_state.values())
            normalized_flows = {edge: flow/max_flow
                              for edge, flow in game.flow_state.items()}
            all_flows.append((iteration + 1, normalized_flows))
        
        print(f"Iteration {iteration + 1}: Total System Cost = {current_cost:.4f}")
        
        # if max_diff < threshold:
            # print(f"Converged after {iteration + 1} iterations")
            # break

        if abs(current_cost - last_cost) < ftol_abs:
            print(f'Frank-Wolfe Terminated due to ftol_abs at iteration {iteration+1}')
            break

        if abs(current_cost - last_cost)/last_cost < ftol_rel:
            print(f'Frank-Wolfe Terminated due to ftol_rel at iteration {iteration+1}')
            break

        last_cost = current_cost
    
    return {edge: flow/max(game.flow_state.values()) 
            for edge, flow in game.flow_state.items()}, costs, all_flows