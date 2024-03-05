from mapf import *
from heapq import heappush, heappop, heapify
import numpy as np
import networkx as nx
import copy
import time


def column_lattice_obstacles(h: int, w: int, dy: int, dx: int, obstacle_rows: int, obstacle_cols: int):
    obstacles = []
    for i in range(obstacle_rows):
        for j in range(obstacle_cols):
            row = i*(w+2*dx)+dx
            col = j*(h+2*dy)+dy
            obstacles+=[(row+k, col+l) for k in range(h) for l in range(w)]
    return obstacles

class GridRegion(Environment):
    def __init__(self, grid_world: GridWorld, location: tuple, size: tuple):
        self.size=size
        self.location = location
        self.boundary = []
        nodes = []
        for node in grid_world.G.nodes:
            if location[0] <= node[0] < location[0]+size[0]:
                if location[1] <= node[1] < location[1]+size[1]:
                    nodes.append(node)
                    if location[0] == node[0]:
                        self.boundary.append(node)
                    elif location[0]+size[0]-1 == node[0]:
                        self.boundary.append(node)
                    elif location[1] == node[1]:
                        self.boundary.append(node)
                    elif location[1]+size[1]-1 == node[1]:
                        self.boundary.append(node)

        self.G = nx.subgraph(grid_world.G, nodes)

    def contains_node(self, u: tuple):
        return u in self.G.nodes
    
    def contains_edge(self, u: tuple, v: tuple):
        return (u,v) in self.G.nodes
    
class RegionActionGenerator(ActionGenerator):
    def __init__(self, world: GridWorld, region: GridRegion, constraints = {}):
        self.world = world
        self.region = region
        self.constraints = constraints

    def actions(self, v:PathVertex):
        if self.region.contains_node(v.pos):
            for pos in self.world.G.adj[v.pos]:
                u = PathVertex(pos, v.t+1)
                e = PathEdge(v.pos, pos, v.t)
                if u in self.constraints:
                    continue
                if e in self.constraints:
                    continue
                if e.compliment() in self.constraints:
                    continue
                yield (u,e)

class RegionalEnvironment(Environment):
    def __init__(self, 
                 gridworld: GridWorld,  # the full world
                 region_graph: nx.Graph
        ):
        self.gridworld = gridworld
        self.region_graph = region_graph
        self.action_generators = {}
        for R in self.region_graph.nodes:
            region = self.region_graph.nodes[R]['env']
            self.action_generators[R] = RegionActionGenerator(gridworld, region)

    def contains_node(self, u: tuple):
        return self.gridworld.contains_node(u)
    
    def contains_edge(self, u: tuple, v: tuple):
        return self.gridworld.contains_edge(u,v)
    
    def dense_matrix(self):
        return self.gridworld.dense_matrix()    
            
class ColumnLatticeEnvironment(RegionalEnvironment):
    def __init__(self, 
                 nrows: int,    # number of rows of subregions
                 ncols: int,    # number of columns of subregions
                 column_h: int, # height of obstacle columns
                 column_w: int, # width of obstacle columns
                 dy: int,       # vertical free-space around columns
                 dx: int,       # horizontal free-space around columns
                 obstacle_rows: int,
                 obstacle_cols: int):
        self.nrows = nrows
        self.ncols = ncols

        # construct the GridWorld
        cell_obstacles = column_lattice_obstacles(column_h, column_w, dy, dx, obstacle_rows, obstacle_cols)
        cell_size = (obstacle_rows*(column_h+2*dy), obstacle_cols*(column_w+2*dx))
        obstacles = []
        for i in range(nrows):
            for j in range(ncols):
                loc = (i*cell_size[0], j*cell_size[1])
                for o in cell_obstacles:
                    obstacles.append((o[0]+loc[0], o[1]+loc[1]))
        world_size = (nrows*cell_size[0],ncols*cell_size[1])
        gridworld = GridWorld(world_size, obstacles)

        # construct the GridRegions
        region_graph = nx.Graph()
        for i in range(nrows):
            for j in range(ncols):
                loc = (i*cell_size[0], j*cell_size[1])
                env = GridRegion(gridworld, loc, cell_size)
                region_graph.add_node((i,j), env=env)
                neighbors = []
                if i > 0:
                    neighbors.append((i-1,j))
                if j > 0:
                    neighbors.append((i,j-1))
                for other in neighbors: 
                    other_env = region_graph.nodes[other]['env']
                    edges = [
                        (u,v) for u in env.boundary for v in other_env.boundary
                        if gridworld.contains_edge(u,v)
                    ]
                    region_graph.add_edge((i,j), other, boundary=edges)

        super().__init__(gridworld, region_graph)

class BoundaryGoal(Goal):
    def __init__(self, env: RegionalEnvironment, source: tuple, dest: tuple, final_goal: tuple):
        if (source, dest) not in env.region_graph.edges:
            raise ValueError(f"source {source} and dest {dest} are not connected in the region graph")
        edges = env.region_graph.edges[source,dest]['boundary']
        region = env.region_graph.nodes[source]['env']
        nodes = [v for e in edges for v in e if not region.contains_node(v)]
        self.set_goal = SetGoal(nodes)
        self.final_goal = LocationGoal(final_goal)

    def heuristic(self, p: tuple):
        return self.set_goal.heuristic(p)
    def satisfied(self, p: tuple):
        return self.set_goal.satisfied(p)
    
def focal_astar(
    action_gen: ActionGenerator.actions,
    V: dict,
    E: dict,
    v: PathVertex,
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

    if v in constraints:
        print('A* infeasibility')
        return None, np.inf
    
    predecessors[v] = None
    d_score = 0
    if v in V:
        d_score += 1
    g_score = 1
    f_score = g_score + h(v.pos)
    d[v] = d_score
    g[v] = g_score
    f[v] = f_score 

    entry = [f_score, v]
    open_finder[v] = entry
    f_best = f_score
    heappush(OPEN, entry)

    entry = [d_score, f_score, v]
    focal_finder[v] = entry
    heappush(FOCAL, entry)

    while len(FOCAL) > 0 or len(OPEN) > 0:
    # while len(FOCAL) > 0:
        # check to see if f_best has changed, if so, reform FOCAL 
        if OPEN[0][0] != f_best:
            f_best = OPEN[0][0]
            OLD_FOCAL = FOCAL
            FOCAL = []
            while len(OLD_FOCAL) > 0:
                entry = heappop(OLD_FOCAL)
                v = entry[-1]
                focal_finder.pop(v)
                if f_score <= omega*f_best:
                    focal_finder[v] = entry
                    heappush(FOCAL, [d_score, f_score, v])
        
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
                heappop(FOCAL)
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
                heappop(OPEN)
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
                    if v in V:
                        d_score += 1
                    if e in E or e.compliment() in E:
                        d_score += 1
                    d[u] = d_score
                    g[u] = g_score
                    f[u] = f_score
                    if u not in open_finder:
                        open_entry = [f_score, u]
                        open_finder[u] = open_entry
                        heappush(OPEN, open_entry)
                    else:
                        open_entry = open_finder[u]
                        if f_score != open_entry[0]:
                            open_entry[0] = f_score
                            heapify(OPEN)
                    if f_score <= OPEN[0][0]*omega:
                        if u not in focal_finder:
                            focal_entry = [d_score, f_score, u]
                            focal_finder[u] = focal_entry
                            heappush(FOCAL, focal_entry)
                        else:
                            focal_entry = focal_finder[u]
                            if focal_entry[0] != d_score or focal_entry[1] != f_score:
                                focal_entry[0] = d_score
                                focal_entry[1] = f_score
                                heapify(FOCAL)
            else:
                predecessors[u] = v 
                d_score = d[v]
                g_score = g[v] + 1
                f_score = g_score + h(u.pos)
                e = PathEdge(v.pos, u.pos, v.t)
                if v in V:
                    d_score += 1
                if e in E or e.compliment() in E:
                    d_score += 1
                d[u] = d_score
                g[u] = g_score
                f[u] = f_score
                entry = [f_score, u]
                open_finder[u] = entry
                heappush(OPEN, entry)
                if f_score <= OPEN[0][0]*omega:
                    entry = [d_score, f_score, u]
                    focal_finder[u] = entry
                    heappush(FOCAL, entry)

    # del queue
    return None, np.inf, np.inf

class CBSNode:
    def __init__(self,
                 x: dict,       # start vertexes for agents
                 goals: dict,   # goals for agents
    ):
        self.x = x              # key = id, value = PathVertex
        self.goals = goals      # key = id, value = Goal
        self.constraints = {}   # key = id, value = hash table of constraints
        self.l = {}             # key = id, value = path length lowerbound (int)
        self.paths = {}         # key = id, value = Path
        self.vertexes = {}      # key = id, value = dict of vertexes 
        self.edges = {}         # key = is, value = dict of edges
        self.conflicts = []
        self.conflict_count = 0
        self.cost = 0
        self.lower_bound = 0
    
    def apply_constraint(self, id: int, c: Constraint):
        if id in self.constraints:
            self.constraints[id][c] = True
        else:
            self.constraints[id] = {c: True}

    def detect_conflicts(self):
        vertexes = {}
        edges = {}
        conflicts = []
        for id in self.paths:
            path = self.paths[id]
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                e = PathEdge(u.pos, v.pos, u.t)
                if v in vertexes:
                    other = vertexes[v]
                    if other[0] != id:
                        conflicts.append([(id,v), other])
                else:
                    vertexes[v] = (id,e)
                if e.compliment() in edges:
                    other = edges[e.compliment()]
                    if other[0] != id:
                        conflicts.append([(id,e), other])
                else:
                    edges[e] = (id,e)
        self.conflicts = conflicts
        self.conflict_count = len(conflicts)
    
    def compute_cost(self):
        self.cost = sum(len(self.paths[id]) for id in self.paths)
        self.lower_bound = sum(self.l[id] for id in self.l)
    
    def branch(self, id: int, c: Constraint):
        node = copy.deepcopy(self)
        node.apply_constraint(id, c)
        node.paths = copy.deepcopy(self.paths)
        node.l = copy.deepcopy(self.l)
        return node

    def __lt__(self, other):
        if type(other) != CBSNode:
            raise ValueError(f'Cannot compare CBSNode to other of type {type(other)}')
        return self.cost < other.cost
    
def update_paths(node: CBSNode, ids: list, a: ActionGenerator.actions, omega: float):
    # delete vertex/edge sets for any agents to be updated
    for id in ids:
        if id in node.vertexes: del node.vertexes[id]
        if id in node.edges: del node.edges[id]
    # create vertex and edge sets to check during A* for conflicts - 
    # this generates a heuristic for sorting A* nodes in the FOCAL heap
    V = {v: True for id in node.vertexes for v in node.vertexes[id]}
    E = {e: True for id in node.edges for e in node.edges[id]}
    for id in ids:
        goal = node.goals[id]
        if id in node.constraints:
            c = node.constraints[id]
        else:
            c = {} # empty hash table of constraints
        path, cost, lb = focal_astar(a, V, E, node.x[id], goal, c, omega)
        node.vertexes[id] = {}
        node.edges[id] = {}
        if path is not None:
            node.paths[id] = path
            node.l[id] = lb
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                e = PathEdge(u.pos, v.pos, u.t)
                node.vertexes[id][v] = True
                node.edges[id][e] = True
        else:
            node.paths[id] = Path([node.x[id]])
            node.l[id] = np.inf
            node.cost = np.inf
            break # can skip other agents because node contains infeasible A* problem

def conflict_based_search(
        root: CBSNode, 
        a: ActionGenerator.actions, 
        omega: float, 
        maxtime=60.,
        verbose=False):
    clock_start = time.time()
    ids = [id for id in root.goals]
    update_paths(root, ids, a, omega)
    root.detect_conflicts()
    root.compute_cost()
    O = [[root.lower_bound, root]]
    F = [[root.conflict_count, root]]
    while len(O) > 0 or len(F) > 0:

        node = None
        while len(F) > 0:
            entry = heappop(F)
            if entry[1].cost <= omega*O[0][0]:
                node = entry[1]
                if verbose:
                    print('CBS popped from F')
                break

        if node is None:
            lower_bound, node = heappop(O)
            if verbose:
                print('CBS popped from O')

        if time.time() - clock_start > maxtime:
            if verbose:
                print('CBS timeout')
            node.cost = np.inf
            return node, np.inf 

        if node.conflict_count > 0:
            if verbose:
                print(f'Current conflict count {node.conflict_count}')
            conflicts = node.conflicts[0]
            for (id, c) in conflicts:
                if verbose:
                    print(f'Applying constraint {c} to {id}')
                new_node = node.branch(id, c)
                update_paths(new_node, [id], a, omega)
                new_node.detect_conflicts()
                new_node.compute_cost()
                if new_node.lower_bound < np.inf:
                    heappush(O, [new_node.lower_bound, new_node])
                    if new_node.cost <= omega * O[0][0]:
                        heappush(F, [new_node.conflict_count, new_node])
        else:
            if verbose:
                print('CBS solution found')
            if len(O) > 0:
                lb = O[0][0]
            else:
                lb = node.lower_bound
            return node, lb
    if verbose:
        print('Infeasible CBS problem')
    node.cost = np.inf
    return node, np.inf
                
class RCBSNode:
    def __init__(self, x: dict, final_goals: dict, region_paths: dict):
        self.x = x
        self.final_goals = final_goals
        self.region_paths = region_paths
        self.partial_paths = {}
        self.trip_idx = dict((id, 0) for id in final_goals)
        self.region_conflicts = []
        self.conflict_count = 0
        self.cost = 0
        self.goal_cost = 0
        self.lb = {}
        self.lower_bound = 0
        self.cbs_nodes = {}
        self.constraints = {}
        
    def detect_conflicts(self):
        conflict_count = 0
        conflicts = []
        vertexes = {}
        edges = {}
        for r in self.cbs_nodes:
            N = self.cbs_nodes[r]
            for id in N.paths:
                path = N.paths[id]
                for i in range(len(path)-1):
                    u = path[i]
                    v = path[i+1]
                    e = PathEdge(u.pos, v.pos, u.t)
                    if v in vertexes:
                        other = vertexes[v]
                        if other[1] != id:
                            conflicts.append([(r,id,e), other])
                            conflict_count += 1
                    else:
                        vertexes[v] = (r,id,e)
                    if e.compliment() in edges:
                        other = edges[e.compliment()]
                        if other[1] != id:
                            conflicts.append([(r,id,e), other])
                            conflict_count += 1
                    else:
                        edges[e] = (r,id,e)
        self.region_conflicts = conflicts
        self.conflict_count = conflict_count

    def compute_cost(self):
        self.goal_cost = sum(-self.trip_idx[id] for id in self.trip_idx)
        self.cost = 0
        self.lower_bound = 0

        for id in self.region_paths:
            region_path = self.region_paths[id]
            trip_idx = self.trip_idx[id]
            for r in region_path[0:trip_idx]:
                path_cost = len(self.partial_paths[r][id])
                self.lower_bound += path_cost
                self.cost += path_cost
        for r in self.cbs_nodes:
            N = self.cbs_nodes[r]
            self.cost += N.cost
        for r in self.lb:
            self.lower_bound += self.lb[r]

    def make_solution(self):
        paths = {}
        for id in self.region_paths:
            region_path = self.region_paths[id]
            for r in region_path:
                path = copy.copy(self.partial_paths[r][id])
                if id not in paths:
                    paths[id] = path
                else:
                    paths[id] += path
        return MAPFSolution(paths)

    def __lt__(self, other):
        pass

def init_rcbs(env: RegionalEnvironment, x: dict, final_goals: dict, region_paths):
    root = RCBSNode(x, final_goals, region_paths)
    cbs_nodes = {r: CBSNode({},{}) for r in env.region_graph.nodes}
    root.cbs_nodes = cbs_nodes
    for r in cbs_nodes:
        N = cbs_nodes[r]
        renv = env.region_graph.nodes[r]['env']
        agents = [id for id in x if renv.contains_node(x[id].pos)]
        for id in agents:
            N.x[id] = x[id]
            region_path = region_paths[id]
            if len(region_path) == 1:
                N.goals[id] = LocationGoal(final_goals[id])
            else:
                r2 = region_path[1]
                N.goals[id] = BoundaryGoal(env, r, r2, final_goals[id])
    return root

def update_region(M: RCBSNode, env: RegionalEnvironment, r:tuple, omega: float, cbs_maxtime: float):
    if r in M.constraints:
        constraints = M.constraints[r]
    else:
        constraints = M.constraints[r] = {}
    action_gen = RegionActionGenerator(env.gridworld, 
                                       env.region_graph.nodes[r]['env'], 
                                       constraints=constraints)
    N, lb = conflict_based_search(M.cbs_nodes[r], action_gen.actions, omega, maxtime=cbs_maxtime)
    M.lb[r] = lb
    M.cbs_nodes[r] = N
    paths = copy.deepcopy(N.paths)
    if r in M.partial_paths:
        M.partial_paths[r].update(paths)
    else:
        M.partial_paths[r] = paths

def branch_rcbs(node: RCBSNode, 
                r: tuple, 
                id: int, 
                c: Constraint):
    # allocate new nodes
    M = copy.deepcopy(node)
    N = M.cbs_nodes[r]
    N.apply_constraint(id, c)
    return M

def advance_agents(node: RCBSNode, env: RegionalEnvironment):
    M = copy.deepcopy(node)
    # advance the agent trip index
    for id in M.trip_idx:
        region_path = M.region_paths[id]
        if M.trip_idx[id] < len(region_path):
            M.trip_idx[id] += 1

    # create new CBS nodes
    M.cbs_nodes = {r : CBSNode({},{}) for r in env.region_graph.nodes}
    M.lb = {}
    for id in M.trip_idx:
        trip_idx = M.trip_idx[id]
        region_path = M.region_paths[id]
        if trip_idx == len(region_path):
            continue # this agent has finished all trips, so skip
        # apply path constraints to r1, the last region
        r1 = region_path[trip_idx-1]
        path = M.partial_paths[r1][id]
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            e = PathEdge(u.pos, v.pos, u.t)
            M.constraints[r1][v] = True
            M.constraints[r1][e] = True
            M.constraints[r1][e.compliment()] = True
        if trip_idx == len(region_path):
            continue # skip the following 
        # set the starting position in r2
        r2 = region_path[trip_idx]
        M.cbs_nodes[r2].x[id] = path[-1]
        # set the goal in r2
        if r2 == region_path[-1]:
            M.cbs_nodes[r2].goals[id] = LocationGoal(M.final_goals[id])
        else:
            r3 = region_path[trip_idx+1]
            M.cbs_nodes[r2].goals[id] = BoundaryGoal(env, r2, r3, M.final_goals[id])
    return M

def regional_cbs(root: RCBSNode, env: RegionalEnvironment, omega: float, maxtime=60., cbs_maxtime=30., verbose=False):
    clock_start = time.time()

    for r in root.cbs_nodes:
        update_region(root, env, r, omega, cbs_maxtime)
    root.compute_cost()
    root.detect_conflicts()

    O = [[root.goal_cost, root.lower_bound, root]]
    F = [[root.goal_cost, root.conflict_count, root]]
    while len(O) > 0 or len(F) > 0:

        if time.time() - clock_start > maxtime:
            print('RCBS timeout')
            # return O, F # return the queue for inspection / completing partial solutions
            return None

        M = None
        while len(F) > 0:
            goal_cost, conflict_count, M = heappop(F)
            if M.cost <= omega * O[0][1]:
                break
            M = None
        if M is None: 
            gc, f, M = heappop(O)

        if M.conflict_count > 0:
            conflict = M.region_conflicts[0]
            for (r, id, c) in conflict:
                if verbose:
                    print(f'Branching at region {r} with constraint {c} applied to agent {id}')
                new_node = branch_rcbs(M, r, id, c)
                update_region(new_node, env, r, omega, cbs_maxtime)
                new_node.compute_cost()
                new_node.detect_conflicts() 
                if new_node.lower_bound < np.inf:
                    heappush(O,[new_node.goal_cost, new_node.lower_bound, new_node])
                    if new_node.cost <= omega * O[0][1]:
                        heappush(F,[new_node.goal_cost, new_node.conflict_count, new_node])
        else:
            if all(M.trip_idx[id] == len(M.region_paths[id]) for id in M.trip_idx):
                if verbose:
                    print('RCBS successful')
                return M
            else:
                if verbose:
                    print(f'# of completed trips {-M.goal_cost}')
                    print('advancing agents...')
                new_node = advance_agents(M, env)
                for r in new_node.cbs_nodes:
                    update_region(new_node, env, r, omega, cbs_maxtime)
                
                new_node.compute_cost()
                new_node.detect_conflicts()
                if new_node.lower_bound < np.inf:
                    heappush(O,[new_node.goal_cost, new_node.lower_bound, new_node])
                    if new_node.cost <= omega * O[0][1]:
                        heappush(F,[new_node.goal_cost, new_node.conflict_count, new_node])
    return None
                
def random_problem(N_agents: int, env: ColumnLatticeEnvironment, path_cutoff=10, rng=np.random.default_rng()):
    # assign start locations to agents
    start_regions = {}
    start_pos = {}
    nodes = list(env.region_graph.nodes)
    for id in range(N_agents):
        start_regions[id] = R = nodes[rng.choice(len(nodes))]
        sub_env = env.region_graph.nodes[R]['env']
        locs = [p 
                for p in sub_env.G.nodes if p not in start_pos.values() and 
                    all(sub_env.contains_node(u) 
                    for u in env.gridworld.G.adj[p])]
        start_pos[id] = locs[rng.choice(len(locs))]

    # assign random final goal regions
    final_goal_regions = {}
    final_goals = {}
    shortest_path_lens = dict(nx.shortest_path_length(env.region_graph))
    for id in start_regions:
        R1 = start_regions[id]
        choices = [R2 for R2 in shortest_path_lens[R1] if shortest_path_lens[R1][R2] < path_cutoff]
        final_goal_regions[id] = R2 = choices[rng.choice(len(choices))]
        sub_env = env.region_graph.nodes[R2]['env']
        locs = [p 
                for p in sub_env.G.nodes if p not in final_goals.values() and
                all(sub_env.contains_node(u)
                    for u in env.gridworld.G.adj[p])]
        final_goals[id] = locs[rng.choice(len(locs))]

    # assemble trip graph with 1-weight edges initially
    trip_graph = nx.Graph()
    for v1 in env.region_graph.nodes:
        edges = []
        for v2 in env.region_graph.adj[v1]:
            edges.append((v1,v2,10))
        trip_graph.add_weighted_edges_from(edges, weight='c')

    # generate regional paths for agents
    region_paths = {}
    for id in start_regions:
        R1 = start_regions[id]
        R2 = final_goal_regions[id]
        if R1 == R2:
            region_paths[id] = [R1]
            continue
        region_paths[id] = path = [R for R in nx.shortest_path(trip_graph, R1, R2, weight='c')]
        for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                e = (u,v)
                trip_graph.edges[e]['c']+=1
    x = {id : PathVertex(start_pos[id], 0) for id in start_pos}
    return x, final_goals, region_paths