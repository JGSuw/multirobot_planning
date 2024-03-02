import cbs
from mapf import *
import time
import numpy as np
import heapq

class ECBSNode(cbs.CBSNode):
    def __init__(self,
                 paths: dict,
                 goals: dict, 
    ):
        super().__init__(paths, goals)
        self.lower_bounds = {}
        self.lower_bound = 0
        self.conflict_count = 0 
        self.occupied_vertexes = {}
        self.traversed_edges = {}
        self.conflicts = []

    def compute_lower_bound(self):
        self.lower_bound = sum(self.lower_bounds[id] for id in self.lower_bounds)
    
    def compute_cost(self):
        self.cost = sum(len(self.paths[id]) for id in self.paths)

    def detect_conflicts(self):
        self.conflicts = cbs.detect_conflicts(self.paths)
        self.conflict_count = len(self.conflicts)

def focal_astar(
    action_gen: ActionGenerator.actions,
    occupied_vertexes: dict,
    traversed_edges: dict,
    path: cbs.Path,
    goal: cbs.Goal,
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

    v = path[0]
    if v in constraints:
        print('A* infeasibility')
        return None, np.inf
    
    predecessors[v] = None
    d_score = 0
    if v in occupied_vertexes:
        d_score += 1
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
                v = entry[-1]
                focal_finder.pop(v)
                if f_score <= omega*f_best:
                    focal_finder[v] = entry
                    heapq.heappush(FOCAL, [d_score, f_score, v])
        
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
        for (u,e) in action_gen(v):
            if u in constraints or e in constraints or e.compliment() in constraints:
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
                        d_score += 1
                    if e.compliment() in traversed_edges:
                        d_score += 1
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
                    d_score += 1
                if e.compliment() in traversed_edges:
                    d_score += 1
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

def low_level_solve(action_generator: ActionGenerator.actions, node: ECBSNode, agents, omega: float):
    for id in agents:
        path = node.paths[id]
        goal = node.goals[id]
        constraints = copy.deepcopy(node.path_constraints)
        if id in node.agent_constraints:
            constraints.update(node.agent_constraints[id])
        new_path, cost, lb = focal_astar(
            lambda v: action_generator.actions(v),
            node.occupied_vertexes,
            node.traversed_edges,
            path,
            goal,
            constraints,
            omega
        )
        if new_path is None:
            node.cost = np.inf
            node.lower_bound = np.inf
            return
        node.paths[id] = new_path
        node.lower_bounds[id] = lb

    # update dictionaries for tracking conflicts
    occupied_vertexes = {}
    traversed_edges = {}
    for id in node.paths:
        path = node.paths[id]
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            e = PathEdge(u.pos, v.pos, u.t)
            if u in occupied_vertexes:
                occupied_vertexes[u].append(id)
            else:
                occupied_vertexes[u] = [id]
            if e in traversed_edges:
                traversed_edges[e] = id
            if v in occupied_vertexes:
                occupied_vertexes[v].append(id)
            else:
                occupied_vertexes[v] = [id]
    node.occupied_vertexes = occupied_vertexes
    node.traversed_edges = traversed_edges
    node.compute_lower_bound()
    node.compute_cost()
    node.detect_conflicts()
            
def enhanced_cbs(action_generator: ActionGenerator.actions, node: ECBSNode, omega: float, maxtime=30., verbose=False):
    clock_start = time.time()
    OPEN = [[node.lower_bound, node]]
    # FOCAL = [[node.conflict_count, node]]
    FOCAL = [[node.conflict_count, node]]
    best_lb = node.lower_bound
    while len(FOCAL) > 0 or len(OPEN) > 0:
    # while len(FOCAL) > 0:
        if OPEN[0][0] != best_lb:
            best_lb = OPEN[0][0]
            if verbose:
                print(f'lower bound updated to {best_lb}')
            OLD_FOCAL = FOCAL
            FOCAL = []
            for i in range(len(OLD_FOCAL)):
                conflict_count, node = heapq.heappop(OLD_FOCAL)
                if node.cost <= omega*OPEN[0][0]:
                    heapq.heappush(FOCAL, [node.conflict_count, node])

        if len(FOCAL) > 0:
            if verbose:
                print('Retrieving node from FOCAL')
            conflict_count, node = heapq.heappop(FOCAL)
        else:
            if verbose:
                print('Retrieving node from OPEN')
            lower_bound, node = heapq.heappop(OPEN)
        if node.conflict_count > 0:
            (id1, c1, id2, c2) = node.conflicts[0]
            if verbose:
                print(f'Conflict between {id1} and {id2} with constraints {c1}, {c2}')
            ids = [id for id in (id1, id2) if id is not None]
            constraints = [c for c in (c1,c2) if c is not None]
            if len(ids) == 2 :
                if verbose:
                    print('branching')
                new_node = copy.deepcopy(node)
                id = ids[1]
                c = constraints[1]
                new_node.apply_agent_constraint(id, c)
                low_level_solve(action_generator, new_node, [id], omega)
                if new_node.cost < np.inf:
                    heapq.heappush(OPEN, [new_node.lower_bound, new_node])
                    if new_node.cost <= omega*OPEN[0][0]:
                        if verbose:
                            print(f'inserting new node into FOCAL with cost {new_node.cost}')
                        heapq.heappush(FOCAL, [new_node.conflict_count, new_node])
                else:
                    if verbose:
                        print('abandoning node')
                    del new_node
            id = ids[0]
            c = constraints[0]
            node.apply_agent_constraint(id, c)
            low_level_solve(action_generator, node,[id], omega)
            if node.cost < np.inf:
                heapq.heappush(OPEN, [node.lower_bound, node])
                if node.cost <= omega*OPEN[0][0]:
                    if verbose:
                        print(f'inserting new node into FOCAL with cost {node.cost}')
                    heapq.heappush(FOCAL, [node.conflict_count, node])
            else:
                if verbose:
                    print('abandoning node')
                del node
        else:
            # we are done
            # return both the current node and the lower bound on the solution value
            return node, OPEN[0][0]
        
        if time.time()-clock_start > maxtime:
            print('ECBS timeout')
            return None
        
    if verbose:
        print('infeasible problem')
    return None

