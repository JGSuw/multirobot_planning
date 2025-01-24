from mapf import *
import ecbs
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
        self.size = size
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

class RegionalEnvironment(Environment):
    def __init__(self, 
                 gridworld: GridWorld,  # the full world
                 region_graph: nx.Graph
        ):
        self.gridworld = gridworld
        self.region_graph = region_graph
        for R in self.region_graph.nodes:
            region_env = self.region_graph.nodes[R]['env']
            self.region_graph.nodes[R]['boundary'] = region_env.boundary # reference this for convenience
            for pos in region_env.G.nodes:
                self.gridworld.G.nodes[pos]['region'] = R

    def contains_node(self, u: tuple):
        return self.gridworld.contains_node(u)
    
    def contains_edge(self, u: tuple, v: tuple):
        return self.gridworld.contains_edge(u,v)
    
    def dense_matrix(self):
        return self.gridworld.dense_matrix()
    
class RCBSGoal(LocationGoal):
    def __init__(self, loc: tuple, env: RegionalEnvironment, include_regions):
        LocationGoal.__init__(self, loc)
        self.env = env
        self.include_regions = include_regions

    def heuristic(self, loc):
        G = self.env.gridworld.G
        value = LocationGoal.heuristic(self, loc)
        region = G.nodes[loc]['region']
        if region in self.include_regions:
            return value - 1
        else:
            return value
    
class RCBSActionGenerator(ActionGenerator):

    def __init__(self, env: RegionalEnvironment, region_path = {}, constraints = {}):
        self.env = env
        self.constraints = constraints
        self.region_path = region_path
    
    def actions(self, v: PathVertex):
        graph = self.env.gridworld.G
        for pos in graph.adj[v.pos]:
            region = graph.nodes[pos]['region']
            if region in self.region_path:
                u = PathVertex(pos, v.t+1)
                e = PathEdge(v.pos, u.pos, v.t)
                if u in self.constraints:
                    continue
                if e in self.constraints:
                    continue
                if e.compliment() in self.constraints:
                    continue
                yield (u,e)
            
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

        RegionalEnvironment.__init__(self,gridworld, region_graph)

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
                
class RCBSNode:
    def __init__(self, x: dict):
        self.x = x
        self.cbs_nodes = {}
        self.agent_constraints = dict((id, {}) for id in x)
        self.path_constraints = {}
        self.trip_idx = dict((id, 0) for id in x)
        self.region_conflicts = []
        self.conflict_count = 0
        self.cost = 0
        self.goal_cost = 0

    def compute_cost(self):
        self.goal_cost = sum(-self.trip_idx[id] for id in self.trip_idx)
        self.cost = 0
        for r, cbs_node in self.cbs_nodes.items():
            self.cost += cbs_node.cost

    def make_solution(self):
        paths = {}
        for id in self.x:
            sort_by = lambda path: path[0].t
            path_generator = [node.paths[id] for r,node in self.cbs_nodes.items() if id in node.paths]
            partial_paths = sorted(path_generator, key=sort_by)
            paths[id] = partial_paths[0]
            for i in range(1,len(partial_paths)):
                paths[id] += partial_paths[i]
        return MAPFSolution(paths)
    
    def __lt__(self, other):
        if self.goal_cost < other.goal_cost:
            return True
        elif self.goal_cost > other.goal_cost:
            return False
        else:
            return self.cost < other.cost

def detect_boundary_conflicts(node: RCBSNode, action_generators: dict):
    """
    New conflict detection logic...

    For each agent, examine their path, marking all locations they occupy or edges they traverse
    that are part of a region boundary.
    """
    vertexes = {}
    edges = {}
    node.region_conflicts = []
    node.conflict_count = 0
    for id, current_trip_idx in node.trip_idx.items():
        action_generator = action_generators[id]
        region_path = action_generator.region_path
        env = action_generator.env
        # for each trip,
        for trip_idx in range(current_trip_idx+1):
            # get the region and path
            region = next(r for r, idx in region_path.items() if idx == trip_idx)
            cbs_node = node.cbs_nodes[region]
            path = cbs_node.paths[id]
            # get the region boundary from the environment
            boundary = env.region_graph.nodes[region]['boundary']
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

def init_rcbs(x: dict, env: RegionalEnvironment):
    root = RCBSNode(x)
    cbs_nodes = {}
    for r in env.region_graph.nodes:
        agents = [id for id, v in x.items() if env.gridworld.G.nodes[v.pos]['region'] == r]
        cbs_node_x = {}
        for id in agents:
            cbs_node_x[id] = x[id]
        cbs_nodes[r] = ecbs.ECBSNode(cbs_node_x)
    root.cbs_nodes = cbs_nodes
    return root

def update_region(node: RCBSNode, action_generators: dict, final_goals: dict, agents: list, omega: float, r: tuple, cbs_maxtime: float):
    cbs_node = node.cbs_nodes[r]
    # assemble goals and apply constraints to action generator
    goals = {}
    env = action_generators[0].env
    for id in cbs_node.x:
        trip_idx = node.trip_idx[id]
        action_generator = action_generators[id]
        action_generator.constraints = node.path_constraints
        env = action_generator.env
        region_path = action_generator.region_path
        if region_path[r] == len(region_path)-1:
            # final goal
            goals[id] = LocationGoal(final_goals[id])
        else:
            # boundary goal
            next_region = next(r for r, idx in region_path.items() if idx==trip_idx+1)
            goals[id] = BoundaryGoal(action_generator.env, r, next_region, final_goals[id])

    # get occupied vertices from neighboring regions
    # occupied_vertexes = {}
    # traversed_edges = {}
    # for region in env.region_graph.adj[r]:
    #     other_node = node.cbs_nodes[r]
    #     for id, path in other_node.paths.items():
    #         for v in path.vertexes:
    #             try:
    #                 occupied_vertexes[v].append(id)
    #             except:
    #                 occupied_vertexes[v] = [id]
    #         for e in path.generate_edges():
    #             try:
    #                 traversed_edges[e].append(id)
    #             except:
    #                 traversed_edges[e] = [id]

    # cbs_node.occupied_vertexes = occupied_vertexes
    # cbs_node.traversed_edges = traversed_edges

    # solve ECBS subproblem
    ecbs.update_paths(cbs_node, action_generators, goals, agents, omega)
    cbs_node, OPEN, FOCAL = ecbs.enhanced_cbs(node.cbs_nodes[r], action_generators, goals, omega, maxtime=cbs_maxtime)
    if cbs_node.cost < np.inf:
        node.cbs_nodes[r] = cbs_node
        node.compute_cost()
    else:
        # infeasible ECBS subproblem
        node.cost = np.inf

def branch_rcbs(node: RCBSNode,
                action_generators: dict,
                id: int, 
                r: tuple, 
                c: Constraint,
                copy_node=True):
    # allocate new node
    if copy_node:
        new_node = copy.deepcopy(node)
    else:
        new_node = node
    # compare current region of agent to r
    region_path = action_generators[id].region_path
    trip_idx = node.trip_idx[id]
    current_r = next(r for r, idx in region_path.items() if idx==trip_idx)
    # in this case, we have to revert the agent to an earlier trip index
    if current_r != r:
        print('agent must revert!')
        print(f'current region = {current_r} at trip index {node.trip_idx[id]}')
        print(f'branching region = {r} at trip index {region_path[r]}')
        # 1) apply constraint to agent in RCBS node
        new_node.agent_constraints[id][c] = True

        # 2) re-initialize CBS node of current region to exclude the agent
        current_cbs_node = new_node.cbs_nodes[current_r]
        current_cbs_node.x.pop(id)
        # current_cbs_node.constraints.pop(id)

        # 3) re-initialize CBS node of the last region to include the agent
        old_cbs_node = new_node.cbs_nodes[r]
        old_path = old_cbs_node.paths[id]
        old_cbs_node.x[id] = old_path[0]
        old_cbs_node.constraints[id] = new_node.agent_constraints[id]

        # 4) remove path constraints imposed by old_path
        for v in old_path.vertexes[1:]:
            new_node.path_constraints.pop(v)
        for e in old_path.generate_edges():
            new_node.path_constraints.pop(e)

        # 4) get the new trip index
        new_trip_idx = next(idx for region,idx in region_path.items() if region==r)
        new_node.trip_idx[id] = new_trip_idx

        # 5) Remove path constraints imposed by the agent's partial paths from regions inbetween
        for idx in range(new_trip_idx+1, trip_idx):
            region = next(_r for _r, _idx in region_path.items() if _idx == idx)
            cbs_node = new_node.cbs_nodes[region]
            path = cbs_node.paths.pop(id)
            for v in path.vertexes[1:]:
                new_node.path_constraints.pop(v)
            for e in path.generate_edges():
                new_node.path_constraints.pop(e)
    else:
        # apply constraint on agent to current region
        new_node.agent_constraints[id][c] = True
        cbs_node = new_node.cbs_nodes[current_r]
        cbs_node.constraints[id] = new_node.agent_constraints[id]

    return new_node

def advance_agents(node: RCBSNode, action_generators: dict):
    # new_node = copy.deepcopy(node)
    # WARNING!!!

    update_agents = dict((r,[]) for r in node.cbs_nodes)

    # loop over agents to apply path constraints and update trip_idx
    for id, trip_idx in node.trip_idx.items():
        action_generator = action_generators[id]
        region_path = action_generator.region_path
        # applying path constraints from agent's last partial path
        if trip_idx < len(region_path)-1:
            last_r = next(r for r, idx in region_path.items() if idx == trip_idx)
            last_path = node.cbs_nodes[last_r].paths[id]
            for i in range(len(last_path)-1):
                u = last_path[i]
                v = last_path[i+1]
                e = PathEdge(u.pos, v.pos, u.t)
                node.path_constraints[v] = True
                node.path_constraints[e] = True

    # re-initialize CBS nodes
    for id in node.trip_idx:
        # node.trip_idx[id] = min(trip_idx+1, len(region_path)-1)
        action_generator = action_generators[id]
        region_path = action_generator.region_path
        trip_idx = node.trip_idx[id]
        if trip_idx == len(region_path)-1:
            # skip this agent
            continue
        else:
            node.trip_idx[id] = trip_idx = trip_idx + 1
        current_r = next(r for r,idx in region_path.items() if idx == trip_idx)
        update_agents[current_r].append(id)
        current_cbs_node = node.cbs_nodes[current_r]
        last_r = next(r for r,idx in region_path.items() if idx == max(trip_idx-1,0))
        last_cbs_node = node.cbs_nodes[last_r]
        if current_r != last_r and id in last_cbs_node.x:
            # pop agent from last CBS node
            last_cbs_node.x.pop(id)
            last_cbs_node.constraints.pop(id)
            # get last_path to copy start vertex
            last_path = last_cbs_node.paths[id]
            # add agent to current CBS node
            current_cbs_node.x[id] = last_path[-1]
            current_cbs_node.paths[id] = Path([last_path[-1]])
            current_cbs_node.constraints[id] = {}
    return update_agents

def regional_cbs(root: RCBSNode, action_generators: dict, final_goals: dict, omega: float, maxtime=60., cbs_maxtime=30., verbose=False):
    clock_start = time.time()
    for r in root.cbs_nodes:
        agents = list(root.cbs_nodes[r].x.keys())
        update_region(root, action_generators, final_goals, agents, omega, r, cbs_maxtime)
    detect_boundary_conflicts(root, action_generators)

    O = [root]

    while len(O) > 0:

        if time.time() - clock_start > maxtime:
            print('RCBS timeout')
            # return O, F # return the queue for inspection / completing partial solutions
            return None, O

        node = heappop(O)

        if node.conflict_count > 0:
            conflict = node.region_conflicts[0]
            for i, (id, r, c) in enumerate(conflict):
                if verbose:
                    print(f'Branching at region {r} with constraint {c} applied to agent {id}')
                if i == 0 and len(conflict) > 1:
                    new_node = branch_rcbs(node, action_generators, id, r, c, copy_node=True)
                else:
                    new_node = branch_rcbs(node, action_generators, id, r, c, copy_node=False)
                update_region(new_node, action_generators, final_goals, [id], omega, r, cbs_maxtime)
                detect_boundary_conflicts(new_node, action_generators)
                if new_node.cost < np.inf:
                    heappush(O,new_node)
                elif verbose:
                    print('Discarding node due to infeasible subproblem')
        else:
            # if all(node.trip_idx[id] == len(node.region_paths[id])-1 for id in node.trip_idx):
                # if verbose:
                    # print('RCBS successful')
                # return node.make_solution(), node, O
            if all(node.trip_idx[id] == len(action_generators[id].region_path)-1 for id in node.trip_idx):
                if verbose:
                    print('RCBS successfull')
                return node, O
            else:
                if verbose:
                    print(f'# of completed trips {-node.goal_cost}')
                    print('advancing agents...')
                update_agents = advance_agents(node, action_generators)
                for r in node.cbs_nodes:
                    agents = list(node.cbs_nodes[r].x.keys())
                    update_region(node, action_generators, final_goals, update_agents[r], omega, r, cbs_maxtime)
                detect_boundary_conflicts(node, action_generators)
                if node.cost < np.inf:
                    heappush(O,node)
                elif verbose:
                    print('Discarding node due to infeasible subproblem')
    return node, O

def random_problem(N_agents: int, gridworld: GridWorld, rng=np.random.default_rng()):
    # assign start locations to agents
    start_pos = {}
    G = gridworld.G
    nodes = list(G.nodes)
    N_nodes = len(nodes)
    start_pos_idx = rng.choice(N_nodes,size=(N_agents,),replace=False)
    for id in range(N_agents):
        start_pos[id] = nodes[start_pos_idx[id]]

    # assign random final goal regions
    final_pos = {}
    final_pos_idx = rng.choice(N_nodes,size=(N_agents,),replace=False)
    for id in start_pos:
        final_pos[id] = nodes[final_pos_idx[id]]

    return start_pos, final_pos

def make_routing_policy(start_pos, final_pos, env: RegionalEnvironment):
    # assemble trip graph with 1-weight edges initially
    trip_graph = nx.Graph()
    for v1 in env.region_graph.nodes:
        edges = []
        sub_env = env.region_graph.nodes[v1]['env']
        for v2 in env.region_graph.adj[v1]:
            edges.append((v1,v2,10))
        trip_graph.add_weighted_edges_from(edges, weight='c')

    # get start regions and stop regions for agents
    start_regions = {}
    stop_regions = {}
    for id, start in start_pos.items():
        start_regions[id] = env.gridworld.G.nodes[start]['region']
        stop_regions[id] = env.gridworld.G.nodes[final_pos[id]]['region']

    # generate regional paths for agents
    region_paths = {}
    for id in start_regions:
        region_paths[id] = {}
        R1 = start_regions[id]
        R2 = stop_regions[id]
        region_paths[id] = {R1 : 0}
        if R1 == R2:
            continue
        else:
            path = [R for R in nx.shortest_path(trip_graph, R1, R2, weight='c')]
            for i in range(len(path)-1):
                    u = path[i]
                    v = path[i+1]
                    e = (u,v)
                    trip_graph.edges[e]['c']+=1
                    region_paths[id][v] = i+1
    return region_paths