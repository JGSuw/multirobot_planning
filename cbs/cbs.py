from mapf import *
from heapq import heapify, heappush, heappop
import copy
import time

def astar(
    action_generator: ActionGenerator.actions,
    v: PathVertex,
    goal: Goal,
    constraints: dict):


    # admissible heuristic for goals
    h = lambda loc: goal.heuristic(loc)

    # scores
    g = {} # g[v] = distance from start to v
    f = {} # f[v] = g[v] + h(v)

    # priority queue
    OPEN = []
    open_finder = {}

    # dictionary to track predecessors of each vertex
    predecessors = {}

    if v in constraints:
        print('A* infeasibility')
        return None, np.inf
    
    predecessors[v] = None
    g_score = 1
    f_score = g_score + h(v.pos)
    g[v] = g_score
    f[v] = f_score 

    entry = [f_score, v]
    open_finder[v] = entry
    heappush(OPEN, entry)

    while len(OPEN) > 0:
        entry = OPEN[0]
        f_score, v = entry[0], entry[1]
        if goal.satisfied(v.pos): 
            # reconstruct the path
            vertexes = []
            while predecessors[v] != None:
                vertexes.append(v)
                v = predecessors[v]
            vertexes.append(v)
            vertexes.reverse()
            path = Path(vertexes)
            return Path(vertexes), len(path)
        else:
            heappop(OPEN) 
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
                    g_score = g[v] + 1
                    f_score = g_score + h(u.pos)
                    g[u] = g_score
                    f[u] = f_score
                    if u not in open_finder:
                        open_entry = [f_score, u]
                        open_finder[u] = open_entry
                        heappush(OPEN, open_entry)
                    else:
                        open_entry = open_finder[u]
                        open_entry[0] = f_score
                        heapify(OPEN)
            else:
                predecessors[u] = v 
                g_score = g[v] + 1
                f_score = g_score + h(u.pos)
                g[u] = g_score
                f[u] = f_score
                entry = [f_score, u]
                open_finder[u] = entry
                heappush(OPEN, entry)

    print('A* infeasibility (empty open queue)')
    return None, np.inf

class CBSNode:
    def __init__(self,
                 x: dict,       # start vertexes for agents
    ):
        self.x = x              # key = id, value = PathVertex
        self.constraints = dict((id, {}) for id in self.x)
        self.paths = dict((id, Path([x[id]])) for id in self.x)
        self.conflicts = []
        self.conflict_count = 0
        self.cost = 0

    def detect_conflicts(self):
        vertexes = {}
        edges = {}
        conflicts = []
        for id, start in self.x.items(): # key by x to ignore paths of agents not part of the current problem?
            # if start in vertexes:
            #     other = vertexes[start]
            #     conflicts.append([other])
            # else:
            #     vertexes[start] = (id, None)
            path = self.paths[id]
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                e = PathEdge(u.pos, v.pos, u.t)
                if v in vertexes:
                    other = vertexes[v]
                    conflicts.append([(id,e),other])
                else:
                    vertexes[v] = (id,e)
                if e.compliment() in edges:
                    other = edges[e.compliment()]
                    conflicts.append([(id,e),other])
                else:
                    edges[e] = (id,e)
        self.conflicts = conflicts
        self.conflict_count = len(conflicts)
    
    def branch(self, id: int, c: Constraint, copy_node=True):
        if copy_node:
            new_node = copy.deepcopy(self)
            new_node.constraints[id][c] = True
        else:
            new_node = self
            new_node.constraints[id][c] = True
        return new_node
    
    def compute_cost(self):
        self.cost = sum(len(path) for id, path in self.paths.items())

    def __lt__(self, other):
        return self.cost < other.cost
    
def update_paths(node: CBSNode, action_generators: dict, goals: dict, agents: list):
    # for id, start in node.x.items():
    for id in agents:
        start = node.x[id]
        constraints = node.constraints[id]
        path, cost = astar(
            action_generators[id], 
            start, 
            goals[id], 
            constraints
        )
        if path is not None:
            node.paths[id] = path
        else:
            node.paths[id] = None
            node.cost = np.inf
            return # skip other agents due to infeasible subproblem
    node.compute_cost()

def conflict_based_search(
        root: CBSNode, 
        action_generators: dict, 
        goals: dict,
        maxtime=60.,
        verbose=False):
    clock_start = time.time()
    root.detect_conflicts()
    O = [root]
    while len(O) > 0:
        node = heappop(O)
        if time.time() - clock_start > maxtime:
            if verbose:
                print('CBS timeout')
            node.cost = np.inf
            return node, node.cost
        if node.conflict_count > 0:
            if verbose:
                print(f'Current conflict count {node.conflict_count}')
            conflicts = node.conflicts[0]
            for (id, c) in conflicts:
                if verbose:
                    print(f'Applying constraint {c} to {id}')
                new_node = node.branch(id, c)
                update_paths(new_node, action_generators, goals, [id])
                new_node.detect_conflicts()
                if new_node.cost < np.inf:
                    heappush(O, new_node)
        else:
            if verbose:
                print('CBS solution found')
            return node, node.cost
    if verbose:
        print('Infeasible CBS problem')
    node.cost = np.inf
    return node, np.inf