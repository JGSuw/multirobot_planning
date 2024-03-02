from mapf import *
# from cbs import *
from ecbs import ECBSNode, enhanced_cbs, low_level_solve, focal_astar
import copy
import time
import heapq
import numpy as np
import networkx as nx
from multiprocessing import Pool

def column_lattice_obstacles(h: int, w: int, dy: int, dx: int, obstacle_rows: int, obstacle_cols: int):
    obstacles = []
    for i in range(obstacle_rows):
        for j in range(obstacle_cols):
            col = i*(w+2*dx)+dx
            row = j*(h+2*dy)+dy
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
    def __init__(self, world: GridWorld, region: GridRegion):
        self.world = world
        self.region = region

    def actions(self, v:PathVertex):
        if self.region.contains_node(v.pos):
            for pos in self.world.G.adj[v.pos]:
                u = PathVertex(pos, v.t+1)
                e = PathEdge(v.pos, pos, v.t)
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
        # return self.set_goal.heuristic(p)+self.final_goal.heuristic(p)
        return self.set_goal.heuristic(p)
    
    def satisfied(self, p: tuple):
        return self.set_goal.satisfied(p)

class RCBSNode:
    def __init__(self, 
                partial_paths: dict,
                region_paths: dict, 
                final_goals: dict,
                agent_constraints = None,
                goal_idx = None,
                ):
        self.cost = 0
        self.cbs_nodes = {}
        self.partial_paths = copy.deepcopy(partial_paths)
        self.region_paths = copy.deepcopy(region_paths)
        self.final_goals = copy.deepcopy(final_goals)
        if agent_constraints is None:
            self.agent_constraints = dict((id, {}) for id in region_paths)
        else:
            self.agent_constraints = copy.deepcopy(agent_constraints)
        self.goal_idx = dict((id, 0) for id in region_paths)
        self.conflict_count = 0
        self.conflicts = []
        self.goal_cost = 0
        self.lower_bounds = {}
        self.lower_bound = 0
        
    def apply_constraint(self, id: int, constraint):
        self.agent_constraints[id][constraint]=True

    def compute_cost(self):
        self.cost = 0
        self.goal_cost = sum(len(self.region_paths[id])-1-self.goal_idx[id] for id in self.goal_idx)
        for R in self.partial_paths:
            for id in self.partial_paths[R]:
                path = self.partial_paths[R][id]
                self.cost += len(path)

    def compute_lower_bound(self, env):
        self.lower_bound = sum(lb for lb in self.lower_bounds.values())
        for id in self.region_paths:
            region_path = self.region_paths[id]
            goal_idx = self.goal_idx[id]
            if goal_idx < len(region_path)-1:
                R_start = region_path[goal_idx]
                start_vertex = self.partial_paths[R_start][id][-1]
                for i in range(goal_idx+1, len(region_path)):
                    if i == len(region_path)-1:
                        goal = LocationGoal(self.final_goals[id])
                    else:
                        goal = BoundaryGoal(env, 
                                            region_path[i], 
                                            region_path[i+1], 
                                            self.final_goals[id])
                    R = region_path[i]
                    action_generator = env.action_generators[R]
                    result = focal_astar(
                        lambda v: action_generator.actions(v),
                        {},
                        {},
                        Path([start_vertex]),
                        goal,
                        {},
                        1.00)
                    if result is None:
                        self.cost = np.inf
                        self.lower_bound = np.inf
                        return
                    else:
                        start_vertex = result[0][-1]
                        self.lower_bound += result[1]

    def get_conflicts(self):
        self.conflicts = detect_conflicts(self)
        self.conflict_count = len(self.conflicts)

    def make_solution(self):
        paths = {}
        for id in self.region_paths:
            for R in self.region_paths[id]:
                path = copy.deepcopy(self.partial_paths[R][id])
                if id in paths:
                    paths[id] += path
                else:
                    paths[id] = path
        return MAPFSolution(paths)

    def __lt__(self, other):
        if self.goal_cost < other.goal_cost:
            return True
        return False
    
    def __gt__(self, other):
        if self.goal_cost < other.goal_cost:
            return False 
        return True

def rcbs_init(env: RegionalEnvironment, partial_paths: dict, region_paths: dict, final_goals:dict):
    node = RCBSNode(partial_paths, region_paths, final_goals)
    for R in partial_paths:
        paths = copy.deepcopy(partial_paths[R])
        goals = {}
        for id in paths:
            if len(region_paths[id]) == 1:
                goals[id] = LocationGoal(final_goals[id])
            else:
                R_idx = region_paths[id].index(R)
                R_dest = region_paths[id][R_idx+1]
                goals[id] = BoundaryGoal(env, R, R_dest, final_goals[id])
        node.cbs_nodes[R] = ECBSNode(paths, goals)
    return node

def update_rcbsnode(env: RegionalEnvironment, node: RCBSNode, regions: list, omega:float, cbs_maxtime, verbose):
    for R in regions:
        cbs_node = node.cbs_nodes[R]
        action_generator = env.action_generators[R]
        low_level_solve(action_generator, cbs_node, [id for id in cbs_node.paths], omega)
        result = enhanced_cbs(action_generator, cbs_node, omega, maxtime=cbs_maxtime)
        if result is None:
            if verbose:
                print('Infeasible CBS subproblem')
            node.cost = np.inf
            return node
        node.cbs_nodes[R] = result[0]
        node.lower_bounds[R] = result[1]
        node.partial_paths[R].update(copy.deepcopy(result[0].paths))
    node.get_conflicts()
    node.compute_cost()
    node.compute_lower_bound(env)

def transfer_agent(env: RegionalEnvironment, node: RCBSNode, new_R: tuple, id: int):
    old_goal_idx = node.goal_idx[id]
    new_goal_idx = node.goal_idx[id] = node.region_paths[id].index(new_R)

    new_node = node.cbs_nodes[new_R]
    old_R = node.region_paths[id][old_goal_idx]
    old_node = node.cbs_nodes[old_R]

    # if the new goal is ahead of the old goal,
    # apply path_constraints to the old region.
    if old_goal_idx < new_goal_idx:
        path = node.partial_paths[old_R][id]
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            e = PathEdge(u.pos, v.pos, u.t).compliment()
            old_node.apply_path_constraint(u)
            old_node.apply_path_constraint(v)
            old_node.apply_path_constraint(e)

    # else delete any partial_paths and path_constraints between new_R and old_R
    # on the agent's region_path
    elif old_goal_idx > new_goal_idx:
        for R in node.region_paths[id][new_goal_idx:old_goal_idx]:
            cbs_node = node.cbs_nodes[R]
            path = node.partial_paths[R][id]
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                e = PathEdge(u.pos, v.pos, u.t).compliment()
                if u in cbs_node.path_constraints: del cbs_node.path_constraints[u]
                if v in cbs_node.path_constraints: del cbs_node.path_constraints[v]
                if e in cbs_node.path_constraints: del cbs_node.path_constraints[e]
            if R != new_R:
                del node.partial_paths[R][id]

    del old_node.paths[id]
    del old_node.goals[id]

    # get the starting vertex of the agent in new_R
    if new_R == node.region_paths[id][0]:
        start_vertex = node.partial_paths[new_R][id][0]
    else:
        last_R = node.region_paths[id][node.goal_idx[id]-1]
        start_vertex = node.partial_paths[last_R][id][-1]

    # add the agent data to CBS node of new_R
    new_node = node.cbs_nodes[new_R]
    new_node.paths[id] = Path([start_vertex])
    if new_R == node.region_paths[id][-1]:
        new_node.goals[id] = LocationGoal(node.final_goals[id])
    else:
        next_R = node.region_paths[id][node.goal_idx[id]+1]
        new_node.goals[id] = BoundaryGoal(env, new_R, next_R, node.final_goals[id])
    new_node.agent_constraints[id] = {}

def rcbs(env: RegionalEnvironment, node, omega: float, cbs_maxtime=30., astar_maxtime=10., maxtime=60., verbose=False):
    clock_start = time.time()
    OPEN = [[node.lower_bound, node]]
    FOCAL = [[node, node.conflict_count]]
    while len(OPEN) > 0 or len(FOCAL) > 0:
        if time.time() - clock_start > maxtime:
            if verbose:
                print('rcbs timeout')
            break
        node = None
        while len(FOCAL) > 0:
            # conflict_count, node = heapq.heappop(FOCAL)
            node, conflict_count = heapq.heappop(FOCAL)
            # if node.goal_cost == OPEN[0][0]:
            if node.cost <= omega*OPEN[0][0]:
                break
            node = None

        if node is None:
            lower_bound, node = heapq.heappop(OPEN)
            if verbose:
                print('popped from OPEN')

        elif verbose:
            print('popped from FOCAL')
        
        if verbose:
            print(f'conflict count: {node.conflict_count}')

        if node.conflict_count > 0:
            R1, id1, c1, R2, id2, c2 = node.conflicts[0]
            new_node = copy.deepcopy(node)
            regions_to_update = [R1]
            R_last = new_node.region_paths[id1][new_node.goal_idx[id1]]

            if R1 != R_last:
                regions_to_update.append(R_last)
                transfer_agent(env, new_node, R1, id1)
                if verbose:
                    print(f'transferring {id1} to {R1}')

            new_node.cbs_nodes[R1].apply_agent_constraint(id1,c1)

            if verbose:
                print(f'Applying constraint {c1} to {id1}')
            update_rcbsnode(env, new_node, regions_to_update, cbs_maxtime, astar_maxtime, False)

            if new_node.cost < np.inf:
                if verbose:
                    print('rcbs branched')
                heapq.heappush(OPEN, [new_node.lower_bound, new_node])
                # if new_node.goal_cost == OPEN[0][0]:
                if new_node.cost <= omega * OPEN[0][0]:
                    # heapq.heappush(FOCAL, [new_node.conflict_count, new_node])
                    heapq.heappush(FOCAL, [new_node, new_node.conflict_count])
            elif verbose:
                print('abandoning node')
                del new_node

            regions_to_update = [R2]
            R_last = node.region_paths[id2][node.goal_idx[id2]]

            if R2 != R_last:
                regions_to_update.append(R_last)
                transfer_agent(env, node, R2, id2)
                if verbose:
                    print(f'transferring {id2} to {R2}')

            node.cbs_nodes[R2].apply_agent_constraint(id2,c2)

            if verbose:
                print(f'Applying constraint {c2} to {id2}')

            update_rcbsnode(env, node, regions_to_update, cbs_maxtime, astar_maxtime, False)

            if node.cost < np.inf:
                heapq.heappush(OPEN, [node.lower_bound, node]) 
                if node.cost <= omega * OPEN[0][0]:
                    if verbose:
                        print(f'focal node cost {node.cost}, best lower bound in OPEN {OPEN[0][0]}')
                    heapq.heappush(FOCAL, [node,node.conflict_count])
            elif verbose:
                print('abandoning node')
                del node
        else:
            if all(node.goal_idx[id] == len(node.region_paths[id])-1 for id in node.region_paths):
                if verbose:
                    print('rcbs succeeded')
                return node, OPEN[0][0]
            else:
                if verbose:
                    print('rcbs transferring agents to next region')
                regions_to_update = []
                for id in node.region_paths:
                    goal_idx = node.goal_idx[id]
                    old_R = node.region_paths[id][goal_idx]
                    if old_R not in regions_to_update: regions_to_update.append(old_R)
                    if goal_idx < len(node.region_paths[id])-1:
                        new_R = node.region_paths[id][goal_idx+1]
                        if new_R not in regions_to_update: regions_to_update.append(new_R)
                        transfer_agent(env, node, new_R, id)
                if verbose:
                    gc = max(len(node.region_paths[id])-1-node.goal_idx[id] for id in node.region_paths)
                    print(f'max of remaining trips {gc}, updating node...')
                update_rcbsnode(env, node, regions_to_update, cbs_maxtime, astar_maxtime, False)
                if node.cost < np.inf:
                    heapq.heappush(OPEN, [node.lower_bound, node])
                    # if node.goal_cost == OPEN[0][0]:
                    if node.cost <= omega*OPEN[0][0]:
                        heapq.heappush(FOCAL, [node, node.conflict_count])
                else:
                    if verbose:
                        print('abandoning node')
                    del node
    return None

def detect_conflicts(node: RCBSNode):
    edges = {}
    vertexes = {}
    conflicts = []
    for id in node.region_paths:
        goal_idx = node.goal_idx[id]
        for R in node.region_paths[id][:goal_idx+1]:
            path = node.partial_paths[R][id]
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                e = PathEdge(u.pos, v.pos, u.t)
                if v in vertexes:
                    other = vertexes[v]
                    if other[1] != id:
                        conflicts.append((R,id,e,*other))
                else:
                    vertexes[v] = (R,id,e)
                if e.compliment() in edges:
                    other = edges[e.compliment()]
                    if other[1] != id:
                        conflicts.append((R,id,e,*other))
                else:
                    edges[e] = (R,id,e)
    return conflicts

def random_problem(N_agents: int, env: ColumnLatticeEnvironment, path_cutoff=10):
    # assign start locations to agents
    start_regions = {}
    start_pos = {}
    nodes = list(env.region_graph.nodes)
    for id in range(N_agents):
        start_regions[id] = R = nodes[np.random.choice(len(nodes))]
        sub_env = env.region_graph.nodes[R]['env']
        locs = [p 
                for p in sub_env.G.nodes if p not in start_pos.values() and 
                    all(sub_env.contains_node(u) 
                    for u in env.gridworld.G.adj[p])]
        start_pos[id] = locs[np.random.choice(len(locs))]

    # assign random final goal regions
    final_goal_regions = {}
    final_goals = {}
    shortest_path_lens = dict(nx.shortest_path_length(env.region_graph))
    for id in start_regions:
        R1 = start_regions[id]
        choices = [R2 for R2 in shortest_path_lens[R1] if shortest_path_lens[R1][R2] < path_cutoff]
        final_goal_regions[id] = R2 = choices[np.random.choice(len(choices))]
        sub_env = env.region_graph.nodes[R2]['env']
        locs = [p 
                for p in sub_env.G.nodes if p not in final_goals.values() and
                all(sub_env.contains_node(u)
                    for u in env.gridworld.G.adj[p])]
        final_goals[id] = locs[np.random.choice(len(locs))]

    # assemble trip graph with 1-weight edges initially
    trip_graph = nx.Graph()
    for v1 in env.region_graph.nodes:
        edges = []
        for v2 in env.region_graph.adj[v1]:
            edges.append((v1,v2,0))
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

    partial_paths = dict((R, {}) for R in env.region_graph.nodes)
    for id in region_paths:
        R = region_paths[id][0]
        path = Path([PathVertex(start_pos[id], 0)])
        partial_paths[R][id] = path
    return partial_paths, region_paths, final_goals