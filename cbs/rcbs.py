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
    
def astar(
    action_gen: ActionGenerator.actions,
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
        f_score, v = heappop(OPEN)
        open_finder.pop(v)
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
                 goals: dict,   # goals for agents
    ):
        self.x = x              # key = id, value = PathVertex
        self.goals = goals      # key = id, value = Goal
        self.constraints = dict((id, {}) for id in self.x)
        self.paths = dict((id, Path([x[id]])) for id in self.x)
        self.conflicts = []
        self.conflict_count = 0
        self.cost = 0

    def detect_conflicts(self):
        vertexes = {}
        edges = {}
        conflicts = []
        for id in self.paths:
            path = self.paths[id]
            v = path[0]
            # if v in vertexes:
                # other = vertexes[v]
                # conflicts.append([other])
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
        return conflicts
    
    def branch(self, id: int, c: Constraint):
        new_node = copy.deepcopy(self)
        new_node.constraints[id][c] = True
        return new_node

    def __lt__(self, other):
        if type(other) != CBSNode:
            raise ValueError(f'Cannot compare CBSNode to other of type {type(other)}')
        return self.cost < other.cost
    
def update_paths(node: CBSNode, a: ActionGenerator.actions):
    for id, start in node.x.items():
        goal = node.goals[id]
        constraints = node.constraints[id]
        path, cost = astar(a, start, goal, constraints)
        if path is not None:
            node.paths[id] = path
            node.cost = cost
        else:
            node.paths[id] = Path([node.x[id]])
            node.cost = np.inf
            break # can skip other agents because node contains infeasible A* problem

def conflict_based_search(
        root: CBSNode, 
        a: ActionGenerator.actions, 
        maxtime=60.,
        verbose=False):
    clock_start = time.time()
    update_paths(root, a)
    root.detect_conflicts()
    O = [root]
    node = None
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
                update_paths(new_node, a)
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
        self.cbs_nodes = {}
        self.agent_constraints = dict((id, {}) for id in x)
        self.path_constraints = {}

    def compute_cost(self):
        self.goal_cost = sum(-self.trip_idx[id] for id in self.trip_idx)
        self.cost = 0

        for id in self.region_paths:
            region_path = self.region_paths[id]
            trip_idx = self.trip_idx[id]
            for r in region_path[0:trip_idx]:
                path_cost = len(self.partial_paths[r][id])
                self.cost += path_cost
        for r, cbs_node in self.cbs_nodes.items():
            self.cost += cbs_node.cost

    def make_solution(self):
        paths = {}
        for id in self.region_paths:
            region_path = self.region_paths[id]
            for r in region_path:
                path = copy.deepcopy(self.partial_paths[r][id])
                if id not in paths:
                    paths[id] = path
                else:
                    paths[id] += path
        return MAPFSolution(paths)
    
    def __lt__(self, other):
        if self.goal_cost < other.goal_cost:
            return True
        elif self.goal_cost > other.goal_cost:
            return False
        else:
            return self.cost <= other.cost

def detect_boundary_conflicts(env: RegionalEnvironment, node: RCBSNode):
    """
    New conflict detection logic...

    For each agent, examine their path, marking all locations they occupy or edges they traverse
    that are part of a region boundary.
    """
    vertexes = {}
    edges = {}
    node.region_conflicts = []
    node.conflict_count = 0
    for id, trip_idx in node.trip_idx.items():
        region_path = node.region_paths[id]
        # for each trip,
        for trip in range(trip_idx+1):
            # get the region and path
            region = region_path[trip]
            path = node.partial_paths[region][id]
            # get the region boundary from the environment
            boundary = env.region_graph.nodes[region]['env'].boundary
            # iterate over the path and check vertexes and edges whenever
            # the agent is occupying a boundary node
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                e = PathEdge(u.pos,v.pos,u.t)
                if u.pos in boundary or v.pos in boundary:
                    if v in vertexes:
                        other = vertexes[v]
                        node.region_conflicts.append([(id,region,e), other])
                        node.conflict_count += 1
                    else:
                        vertexes[v] = (id,region,e)
                    if e.compliment() in edges:
                        other = edges[e.compliment()]
                        node.region_conflicts.append([(id,region,e), other])
                        node.conflict_count += 1
                    else:
                        edges[e] = (id,region,e)

def init_rcbs(env: RegionalEnvironment, x: dict, final_goals: dict, region_paths):
    root = RCBSNode(x, final_goals, region_paths)
    cbs_nodes = {}
    for r in env.region_graph.nodes:
        # N = cbs_nodes[r]
        renv = env.region_graph.nodes[r]['env']
        agents = [id for id in x if renv.contains_node(x[id].pos)]
        # start positions and goals
        cbs_node_x = {}
        cbs_node_goals = {}
        for id in agents:
            cbs_node_x[id] = x[id]
            region_path = region_paths[id]
            if len(region_path) == 1:
                cbs_node_goals[id] = LocationGoal(final_goals[id])
            else:
                r2 = region_path[1]
                cbs_node_goals[id] = BoundaryGoal(env, r, r2, final_goals[id])
        cbs_nodes[r] = CBSNode(cbs_node_x, cbs_node_goals)
    root.cbs_nodes = cbs_nodes
    return root

def update_region(env: RegionalEnvironment, node: RCBSNode, r: tuple, cbs_maxtime: float):
    action_gen = RegionActionGenerator(env.gridworld, 
                                       env.region_graph.nodes[r]['env'], 
                                       constraints=node.path_constraints)
    cbs_node, cost = conflict_based_search(node.cbs_nodes[r], action_gen.actions, maxtime=cbs_maxtime)
    node.cbs_nodes[r] = cbs_node
    paths = copy.deepcopy(cbs_node.paths)
    try:
        node.partial_paths[r].update(paths)
    except KeyError:
        node.partial_paths[r] = paths

def branch_rcbs(env: RegionalEnvironment,
                node: RCBSNode, 
                id: int, 
                r: tuple, 
                c: Constraint):
    # allocate new node
    new_node = copy.deepcopy(node)
    # compare current region of agent to r
    region_path = node.region_paths[id]
    trip_idx = node.trip_idx[id]
    current_r = node.region_paths[id][trip_idx]

    # in this case, we have to revert the agent to an earlier trip index
    if current_r != r:
        print('agent must revert!')
        # 1) apply constraint to agent in RCBS node
        new_node.agent_constraints[id][c] = True

        # 2) re-initialize CBS node of current region to exclude the agent
        old_cbs_node = new_node.cbs_nodes[current_r]
        del old_cbs_node.x[id]
        del old_cbs_node.goals[id]
        new_node.cbs_nodes[current_r] = CBSNode(old_cbs_node.x, old_cbs_node.goals)
        new_cbs_node = CBSNode(old_cbs_node.x, old_cbs_node.goals)
        for id in new_cbs_node.x:
            new_cbs_node.constraints[id] = copy.deepcopy(new_node.agent_constraints[id])
        new_node.cbs_nodes[current_r] = new_node

        # 3) re-initialize CBS node of the last region to include the agent
        old_cbs_node = new_node.cbs_nodes[r]
        last_path = new_node.partial_paths[r][id]
        old_cbs_node.x[id] = last_path[0]
        old_cbs_node.goals[id] = BoundaryGoal(env, r, current_r, new_node.final_goals[id])
        new_cbs_node = CBSNode(old_cbs_node.x, old_cbs_node.goals)
        for id in new_cbs_node.x:
            new_cbs_node.constraints[id] = copy.deepcopy(new_node.agent_constraints)
        new_node.cbs_nodes[r] = new_node

        # 4) get the new trip idx
        new_trip_idx = next(i for i in range(len(region_path)) if r == region_path[i])
        new_node.trip_idx[id] = new_trip_idx

        # 5) Remove path constraints imposed by the agent's partial paths from regions inbetween
        for idx in range(new_trip_idx, trip_idx):
            old_region = region_path[idx]
            old_path = new_node.partial_paths[old_region][id]
            for i in range(len(old_path)-1):
                u = last_path[i]
                v = last_path[i+1]
                e = PathEdge(u.pos,v.pos,u.t)
                del new_node.path_constraints[v]
                del new_node.path_constraints[e]
            # old_path is no longer needed in the partial_paths of old_region
            # del new_node.partial_paths[old_region][id]
    else:
        # apply constraint on agent to current region
        new_node.agent_constraints[id][c] = True
        new_node.cbs_nodes[current_r].constraints[id][c] = True

    return new_node

def advance_agents(env: RegionalEnvironment, node: RCBSNode):
    new_node = copy.deepcopy(node)

    # loop over agents to apply path constraints and update trip_idx
    for id, trip_idx in new_node.trip_idx.items():
        region_path = new_node.region_paths[id]
        # applying path constraints from agent's last partial path
        if trip_idx < len(region_path)-1:
            last_r = region_path[trip_idx]
            last_path = new_node.partial_paths[last_r][id]
            for i in range(len(last_path)-1):
                u = last_path[i]
                v = last_path[i+1]
                e = PathEdge(u.pos, v.pos, u.t)
                new_node.path_constraints[v] = True
                new_node.path_constraints[e] = True
        # incrementing the agent's trip index
        new_node.trip_idx[id] = min(trip_idx+1, len(region_path)-1)

    # loop over regions, initializing new cbs nodes
    new_cbs_nodes = {}
    for region in new_node.cbs_nodes:
        x = {}
        goals = {}
        agent_constraints = {}
        for id, trip_idx in new_node.trip_idx.items():
            region_path = new_node.region_paths[id]
            last_r = region_path[max(trip_idx-1,0)]
            current_r = region_path[trip_idx]
            if current_r == region:
                # apply the start position of the agent in this region to x
                if last_r != current_r:
                    x[id] = new_node.partial_paths[last_r][id][-1]
                else:
                    x[id] = new_node.x[id]
                # determine the agent's goal (boundary goal if intermediate trip, location goal if final trip)
                if trip_idx < len(region_path)-1:
                    next_r = region_path[trip_idx+1]
                    goals[id] = BoundaryGoal(env, current_r, next_r, new_node.final_goals[id])
                else:
                    goals[id] = LocationGoal(new_node.final_goals[id])
                agent_constraints[id] = copy.deepcopy(new_node.agent_constraints[id])
        new_cbs_nodes[region] = CBSNode(x,goals)
        new_cbs_nodes[region].constraints = agent_constraints
    new_node.cbs_nodes = new_cbs_nodes
    return new_node

def regional_cbs(root: RCBSNode, env: RegionalEnvironment, omega: float, maxtime=60., cbs_maxtime=30., verbose=False):
    clock_start = time.time()

    for r in root.cbs_nodes:
        update_region(env, root, r, cbs_maxtime)
    root.compute_cost()
    detect_boundary_conflicts(env, root)

    O = [root]
    node = None
    while len(O) > 0:

        if time.time() - clock_start > maxtime:
            print('RCBS timeout')
            # return O, F # return the queue for inspection / completing partial solutions
            return None, node, O

        node = heappop(O)

        if node.conflict_count > 0:
            conflict = node.region_conflicts[0]
            for (id, r, c) in conflict:
                if verbose:
                    print(f'Branching at region {r} with constraint {c} applied to agent {id}')
                new_node = branch_rcbs(env, node, id, r, c)
                update_region(env, new_node, r, cbs_maxtime)
                new_node.compute_cost()
                detect_boundary_conflicts(env, new_node)
                if new_node.cost < np.inf:
                    heappush(O,new_node)
                elif verbose:
                    print('Discarding node due to infeasible subproblem')
        else:
            if all(node.trip_idx[id] == len(node.region_paths[id])-1 for id in node.trip_idx):
                if verbose:
                    print('RCBS successful')
                return node.make_solution(), node, O
            else:
                if verbose:
                    print(f'# of completed trips {-node.goal_cost}')
                    print('advancing agents...')
                new_node = advance_agents(env, node)
                for r in new_node.cbs_nodes:
                    update_region(env, new_node, r, cbs_maxtime)
                new_node.compute_cost()
                detect_boundary_conflicts(env, new_node)
                if new_node.cost < np.inf:
                    heappush(O,new_node)
                elif verbose:
                    print('Discarding node due to infeasible subproblem')
    return None, node, O
                
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