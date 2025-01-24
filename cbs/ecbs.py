import cbs
from mapf import *
import time
import numpy as np
import heapq

class ECBSNode(cbs.CBSNode):
    def __init__(self, x: dict):
        super().__init__(x)
        self.lower_bounds = {}
        self.lower_bound = 0
        self.occupied_vertexes = {}
        self.traversed_edges = {}

    def compute_lower_bound(self):
        self.lower_bound = sum(self.lower_bounds[id] for id in self.lower_bounds)
    
    def compute_cost(self):
        self.cost = sum(len(self.paths[id]) for id in self.paths)
    
def focal_astar(
    action_generator: ActionGenerator,
    occupied_vertexes: dict,
    traversed_edges: dict,
    start: PathVertex,
    goal: Goal,
    constraints: dict,
    omega: float):

    # queues
    OPEN = []
    open_finder = {}
    FOCAL = []
    focal_finder = {}

    # dictionary to track predecessors of each vertex
    predecessors = {}

    # admissible heuristic for goals
    h = lambda loc: goal.heuristic(loc)

    # scores
    d = {} # d[v] = # of conflicts from start to v
    g = {} # g[v] = distance from start to v
    f = {} # f[v] = g[v] + h(v)

    v = start
    if v in constraints:
        print('A* infeasibility')
        return None, np.inf
    
    predecessors[v] = None
    d_score = 0
    g_score = 1
    f_score = g_score + h(v.pos)
    d[v] = d_score
    g[v] = g_score
    f[v] = f_score 

    entry = [f_score, v]
    open_finder[v] = entry
    f_best = f_score
    heapq.heappush(OPEN, entry)

    entry = [d_score, f_score, v]
    focal_finder[v] = entry
    heapq.heappush(FOCAL, entry)

    while len(FOCAL) > 0 or len(OPEN) > 0:
    # while len(FOCAL) > 0:
        # check to see if f_best has changed, if so, reform FOCAL 
        if OPEN[0][0] != f_best:
            f_best = OPEN[0][0]
            OLD_FOCAL = FOCAL
            FOCAL = []
            while len(OLD_FOCAL) > 0:
                entry = heapq.heappop(OLD_FOCAL)
                d_score = entry[0]
                f_score = entry[1]
                v = entry[2]
                focal_finder.pop(v)
                if f_score <= omega*f_best:
                    focal_finder[v] = entry
                    heapq.heappush(FOCAL, entry)
        
        if len(FOCAL) > 0:
            entry = FOCAL[0]
            d_score = entry[0]
            f_score = entry[1]
            v = entry[2]
            if goal.satisfied(v.pos): 
                # reconstruct the path
                vertexes = []
                while predecessors[v] != None:
                    vertexes.append(v)
                    v = predecessors[v]
                vertexes.append(v)
                vertexes.reverse()
                path = Path(vertexes)
                return Path(vertexes), len(path), OPEN[0][0]
            else:
                heapq.heappop(FOCAL)
                focal_finder.pop(v)
        else:
            entry = OPEN[0]
            f_score = entry[0]
            v = entry[1]
            if goal.satisfied(v.pos): 
                # reconstruct the path
                vertexes = []
                while predecessors[v] != None:
                    vertexes.append(v)
                    v = predecessors[v]
                vertexes.append(v)
                vertexes.reverse()
                path = Path(vertexes)
                return Path(vertexes), len(path), OPEN[0][0]
            else:
                heapq.heappop(OPEN)
                open_finder.pop(v)

        # get new nodes
        new_nodes = []
        for (u,e) in action_generator.actions(v):
            if e in constraints:
                continue # skip this vertex
            new_nodes.append(u)

        # update scores for new nodes
        for u in new_nodes:
            if u in g:
                if g[v] + 1 < g[u]:
                    predecessors[u] = v
                    d_score = d[v]
                    g_score = g[v] + 1
                    f_score = g_score + h(u.pos)
                    e = PathEdge(v.pos, u.pos, v.t)
                    if u in occupied_vertexes:
                        d_score += len(occupied_vertexes[u])
                    if e.compliment() in traversed_edges:
                        d_score += len(traversed_edges[e.compliment()])
                    d[u] = d_score
                    g[u] = g_score
                    f[u] = f_score
                    if u not in open_finder:
                        open_entry = [f_score, u]
                        open_finder[u] = open_entry
                        heapq.heappush(OPEN, open_entry)
                    else:
                        open_entry = open_finder[u]
                        if f_score != open_entry[0]:
                            open_entry[0] = f_score
                            heapq.heapify(OPEN)
                    if f_score <= OPEN[0][0]*omega:
                        if u not in focal_finder:
                            focal_entry = [d_score, f_score, u]
                            focal_finder[u] = focal_entry
                            heapq.heappush(FOCAL, focal_entry)
                        else:
                            focal_entry = focal_finder[u]
                            if focal_entry[0] != d_score or focal_entry[1] != f_score:
                                focal_entry[0] = d_score
                                focal_entry[1] = f_score
                                heapq.heapify(FOCAL)
            else:
                predecessors[u] = v 
                d_score = d[v]
                g_score = g[v] + 1
                f_score = g_score + h(u.pos)
                e = PathEdge(v.pos, u.pos, v.t)
                if u in occupied_vertexes:
                    d_score += len(occupied_vertexes[u])
                if e.compliment() in traversed_edges:
                    d_score += len(traversed_edges[e.compliment()])
                d[u] = d_score
                g[u] = g_score
                f[u] = f_score
                entry = [f_score, u]
                open_finder[u] = entry
                heapq.heappush(OPEN, entry)
                if f_score <= OPEN[0][0]*omega:
                    entry = [d_score, f_score, u]
                    focal_finder[u] = entry
                    heapq.heappush(FOCAL, entry)

    # del queue
    return None, np.inf, np.inf

def update_paths(node: ECBSNode, action_generators: dict, goals: dict, agents: list, omega: float):
    for id in agents:
        # remove this agent's path from the occupied vertices and traversed edges
        path = node.paths[id]
        for v in path.vertexes:
            try:
                ids = node.occupied_vertexes[v]
                ids.remove(id)
            except:
                pass
        for e in path.generate_edges():
            try:
                ids = node.traversed_edges[e]
                ids.remove(id)
            except:
                pass

        # run focal search to update the agents path
        new_path, cost, lb = focal_astar(
            action_generators[id],
            node.occupied_vertexes,
            node.traversed_edges,
            path[0],
            goals[id],
            node.constraints[id],
            omega
        )
        if new_path is None:
            node.cost = np.inf
            node.lower_bound = np.inf
            return
        node.paths[id] = new_path
        node.lower_bounds[id] = lb
        # update occupied_vertexes and occupied_edges with new_path
        for v in new_path.vertexes:
            try:
                node.occupied_vertexes[v].append(id)
            except:
                node.occupied_vertexes[v] = [id]
        for e in new_path.generate_edges():
            try:
                node.traversed_edges[e].append(id)
            except:
                node.traversed_edges[e] = [id]
    node.compute_lower_bound()
    node.compute_cost()
            
def enhanced_cbs(root: ECBSNode, action_generators: dict, goals: dict, omega: float, maxtime=30., verbose=False):
    clock_start = time.time()
    root.detect_conflicts()
    OPEN = [[root.lower_bound, root]]
    FOCAL = [[len(root.conflicts), root]]
    best_lb = root.lower_bound
    while len(FOCAL) > 0 or len(OPEN) > 0:
        if OPEN[0][0] != best_lb:
            best_lb = OPEN[0][0]
            if verbose:
                print(f'lower bound updated to {best_lb}')
            OLD_FOCAL = FOCAL
            FOCAL = []
            for i in range(len(OLD_FOCAL)):
                conflict_count, node = heapq.heappop(OLD_FOCAL)
                if node.cost <= omega*OPEN[0][0]:
                    heapq.heappush(FOCAL, [conflict_count, node])

        if len(FOCAL) > 0:
            if verbose:
                print('Retrieving node from FOCAL')
            conflict_count, node = heapq.heappop(FOCAL)
        else:
            if verbose:
                print('Retrieving node from OPEN')
            lower_bound, node = heapq.heappop(OPEN)
            conflict_count = len(node.conflicts)
        if conflict_count > 0:
            conflicts = node.conflicts[0]
            for i,(id, c) in enumerate(conflicts):
                if verbose:
                    print(f'Applying constraint {c} to {id}')
                if i == 0 and len(conflicts) > 1:
                    new_node = node.branch(id, c, copy_node=True)
                else:
                    new_node = node.branch(id, c, copy_node=False)
                update_paths(new_node, action_generators, goals, [id], omega)
                new_node.detect_conflicts()
                if new_node.cost < np.inf:
                    heapq.heappush(OPEN, [new_node.lower_bound, new_node])
                    if new_node.cost <= omega*OPEN[0][0]:
                        if verbose:
                            print(f'inserting new node into FOCAL with cost {new_node.cost}')
                        heapq.heappush(FOCAL, [len(new_node.conflicts), new_node])
                else:
                    if verbose:
                        print('abandoning node')
                    del new_node
        else:
            # we are done
            # return both the current node and the lower bound on the solution value
            return node, OPEN, FOCAL
        
        if time.time()-clock_start > maxtime:
            print('ECBS timeout')
            return None, OPEN, FOCAL
        
    if verbose:
        print('infeasible problem')
    return None, OPEN, FOCAL

