from mapf import *
from cbs import *
import copy
import time
import heapq
import numpy as np
import networkx as nx

def column_lattice_obstacles(h: int, w: int, dy: int, dx: int):
    obstacles = []
    for i in range(1):
        for j in range(1):
            col = i*(w+2*dx)+dx
            row = j*(h+2*dy)+dy
            obstacles+=[(row+k, col+l) for k in range(h) for l in range(w)]
    return obstacles

class GridRegion(Environment):
    def __init__(self, grid_world: GridWorld, location: tuple, size: tuple):
        self.size=size
        self.location = location
        nodes = []
        for node in grid_world.G.nodes:
            if location[0] <= node[0] < location[0]+size[0]:
                if location[1] <= node[1] < location[1]+size[1]:
                    nodes.append(node)

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
                 regions: list          # list of GridRegions corresponding to gridworld
        ):
        self.gridworld = gridworld
        self.region_graph = nx.Graph()
        for R in regions:
            self.region_graph.add_node(R, env=regions[R])
        # add edges to the region graph
        R_nodes = list(self.region_graph.nodes)
        for i, u in enumerate(R_nodes):
            for j, v in enumerate(R_nodes[i+1:], start=i+1):
                Ri = self.region_graph.nodes[u]['env']
                Rj = self.region_graph.nodes[v]['env']
                edges = [(vi,vj)
                         for vi in Ri.G.nodes
                         for vj in Rj.G.nodes
                         if (vi,vj) in gridworld.G.edges]
                if len(edges) > 0 and u != v:
                    self.region_graph.add_edge(u, v, boundary = edges)

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
                 dx: int):      # horizontal free-space around columns
        self.nrows = nrows
        self.ncols = ncols

        # construct the GridWorld
        cell_obstacles = column_lattice_obstacles(column_h, column_w, dy, dx)
        cell_size = (column_h+2*dy, column_w+2*dx)
        obstacles = []
        for i in range(nrows):
            for j in range(ncols):
                loc = (i*cell_size[0], j*cell_size[1])
                for o in cell_obstacles:
                    obstacles.append((o[0]+loc[0], o[1]+loc[1]))
        world_size = (nrows*cell_size[0],ncols*cell_size[1])
        gridworld = GridWorld(world_size, obstacles)

        # construct the GridRegions
        regions = {}
        for i in range(nrows):
            for j in range(ncols):
                loc = (i*cell_size[0], j*cell_size[1])
                regions[(i,j)] = GridRegion(gridworld, loc, cell_size)

        super().__init__(gridworld, regions)

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
        
    def apply_constraint(self, id: int, constraint):
        self.agent_constraints[id][constraint]=True

    def __lt__(self, other):
        # gc1 = sum(-self.goal_idx[id] for id in self.region_paths)
        # gc2 = sum(-other.goal_idx[id] for id in other.region_paths)
        gc1 = max(len(self.region_paths[id])-self.goal_idx[id] for id in self.goal_idx)
        gc2 = max(len(self.region_paths[id])-other.goal_idx[id] for id in other.goal_idx)
        # gc1 = min(-self.goal_idx[id] for id in self.goal_idx)
        # gc2 = min(-other.goal_idx[id] for id in other.goal_idx)
        if gc1 < gc2:
            return True
        elif gc2 < gc1:
            return False
        else:
            return self.cost < other.cost
        # return self.cost < other.cost
    
    def __gt__(self, other):
        return not self.__lt__(other)

def agent_finished(node: RCBSNode, id):
    return node.goal_idx[id] == len(node.goals[id])-1

def update_agent(env: RegionalEnvironment, node: RCBSNode, start_region:tuple, id: int, astar_maxtime):
    R_idx = node.region_paths[id].index(start_region)
    if R_idx > 0:
        R_idx = R_idx - 1
    for i, R in enumerate(node.region_paths[id][R_idx:]):
        if R == node.region_paths[id][-1]:
            goal = LocationGoal(node.final_goals[id])
        else:
            goal = BoundaryGoal(env, R, node.region_paths[id][R_idx+i+1], node.final_goals[id])
        if i == 0:
            path = node.partial_paths[R][id]
        else:
            last_R = node.region_paths[id][R_idx+i-1]
            start_vertex = node.partial_paths[last_R][id][-1]
            path = Path([start_vertex])

        new_path, cost = astar(
            lambda v: env.actions(R, v), 
            path, 
            goal, 
            node.agent_constraints[id],
            maxtime = astar_maxtime)
        if cost == np.inf:
            node.cost = np.inf
            return
        else:
            node.partial_paths[R][id] = new_path
    node.cost = 0
    for R in node.partial_paths:
        for id in node.partial_paths[R]:
            node.cost += len(node.partial_paths[R][id])

def rcbs_init(env, partial_paths: dict, region_paths: dict, final_goals:dict):
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
        action_gen = env.action_generators[R]
        node.cbs_nodes[R] = CBSNode(paths, goals, action_gen)
    return node

def update_rcbsnode(node: RCBSNode, cbs_maxtime, astar_maxtime, verbose):

    # how can I be sure that the agent paths start at the right location
    # in each fucking node?
    node.cost = 0
    for R in node.cbs_nodes:
        cbs_node = node.cbs_nodes[R]
        result = conflict_based_search(cbs_node, maxtime=cbs_maxtime, astar_maxtime=astar_maxtime, verbose=verbose)
        if result is None:
            if verbose:
                print('Infeasible CBS subproblem')
            node.cost = np.inf
            break
        node.cbs_nodes[R] = result
        node.cost += node.cost
        node.partial_paths[R].update(result.paths)

def transfer_agents(env: RegionalEnvironment, node: RCBSNode):
    for id in node.region_paths:
        R_source = node.region_paths[id][node.goal_idx[id]]
        source_node = node.cbs_nodes[R_source]
        if R_source == node.region_paths[id][-1]:
            continue # skip this agent, it is at the final goal
        # make path constraints from this agent's path
        path = node.partial_paths[R_source][id]
        path_constraints = {}
        for i in range(len(path)-1):
            u = path.vertexes[i]
            v = path.vertexes[i+1]
            e = PathEdge(u.pos, v.pos, u.t)
            path_constraints[u] = True
            path_constraints[e] = True
            path_constraints[e.compliment()] = True

        # remove this agent from source_node
        del source_node.paths[id]
        del source_node.goals[id]
        if id in source_node.agent_constraints:
            del source_node.agent_constraints[id]

        # apply path constraints to remaining agents
        for other_id in source_node.paths:
            if other_id in source_node.agent_constraints:
                source_node.agent_constraints[other_id].update(path_constraints)
            else:
                source_node.agent_constraints[other_id] = path_constraints

        node.goal_idx[id] += 1
        R_dest = node.region_paths[id][node.goal_idx[id]]
        # move agent into the new region
        dest_node = node.cbs_nodes[R_dest]
        start_vertex = copy.deepcopy(path.vertexes[-1])
        dest_node.paths[id] = Path([start_vertex])
        if R_dest == node.region_paths[id][-1]:
            dest_node.goals[id] = LocationGoal(node.final_goals[id])
        else:
            next_dest = node.region_paths[id][node.goal_idx[id]+1]
            dest_node.goals[id] = BoundaryGoal(env, R_dest, next_dest, node.final_goals[id])
        dest_node.agent_constraints[id] = {}

def rcbs(env: RegionalEnvironment, queue, cbs_maxtime=30., astar_maxtime=10., maxtime=60., verbose=False):
    clock_start = time.time()
    while len(queue) > 0:
        if time.time() - clock_start > maxtime:
            if verbose:
                print('rcbs timeout')
            break
        node = heapq.heappop(queue)

        conflicts = detect_conflicts(node)

        if len(conflicts) > 0:
            R1, id1, c1, R2, id2, c2 = conflicts[0]
            print(f'R1={R1}, id1={id1}, c1={c1}, R2={R2}, id2={id2}, c2={c2}')
            new_node = copy.deepcopy(node)
            cbs_node = new_node.cbs_nodes[R1]
            cbs_node.apply_constraint(id1,c1)
            update_rcbsnode(new_node, cbs_maxtime, astar_maxtime, False)
            if new_node.cost < np.inf:
                heapq.heappush(queue, new_node)
            elif verbose:
                print('abandoning node')
                del new_node
            cbs_node = node.cbs_nodes[R2]
            cbs_node.apply_constraint(id2,c2)
            update_rcbsnode(node, cbs_maxtime, astar_maxtime, False)
            if node.cost < np.inf:
                heapq.heappush(queue, node)
            elif verbose:
                print('abandoning node')
                del node
        else:
            if all(node.goal_idx[id] == len(node.region_paths[id])-1 for id in node.region_paths):
                if verbose:
                    print('rcbs succeeded')
                paths = {}
                for id in node.region_paths:
                    for R in node.region_paths[id]:
                        partial_path = node.partial_paths[R][id]
                        if id in paths:
                            paths[id] += partial_path
                        else:
                            paths[id] = partial_path
                return MAPFSolution(paths)
            else:
                if verbose:
                    print('rcbs transferring agents to next region')
                transfer_agents(env, node)
                if verbose:
                    print(node.goal_idx)
                update_rcbsnode(node, cbs_maxtime, astar_maxtime, False)
                if node.cost < np.inf:
                    heapq.heappush(queue, node)
                else:
                    if verbose:
                        print("BAD THINGS ARE HAPPENING")
                    if len(queue) == 0:
                        return node
                    else:
                        del node
    return queue

def detect_conflicts(node: RCBSNode):
    edges = {}
    vertexes = {}
    # final_pos = {}
    # final_t = {} 
    conflicts = []
    for id in node.region_paths:
        goal_idx = node.goal_idx[id]
        R_source = node.region_paths[id][goal_idx]
        # for i, R_source in enumerate(node.region_paths[id][goal_idx:], start=goal_idx):
        # R_source = node.region_paths[id][i]
        path = node.partial_paths[R_source][id]
        u = path[0]

        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            e = PathEdge(u.pos, v.pos, u.t)
            
            if u in vertexes:
                other = vertexes[u]
                if other[1] != id:
                    conflicts.append((R_source, id, u, *other))
            else:
                vertexes[u] = (R_source, id, u)

            if v in vertexes:
                other = vertexes[v]
                if other[1] != id:
                    conflicts.append((R_source, id, e, *other))
            else:
                vertexes[v] = (R_source, id, e)

            if e.compliment() in edges:
                other = edges[e.compliment()]
                if other[1] != id:
                    conflicts.append((R_source, id, e, *other))
            else:
                edges[e] = (R_source, id, e)
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
            edges.append((v1,v2,1))
        trip_graph.add_weighted_edges_from(edges, weight='c')

    # generate regional paths for agents
    region_paths = {}
    for id in start_regions:
        R1 = start_regions[id]
        R2 = final_goal_regions[id]
        if R1 == R2:
            region_paths[id] = [R1]
            continue
        paths = list(p for p in nx.all_simple_paths(env.region_graph, R1, R2, cutoff = path_cutoff))
        weights = np.zeros(len(paths))
        for idx, path in enumerate(paths):
            for i in range(len(path)-1):
                u = path[i]
                v = path[i+1]
                e = (u,v)
                weights[idx]-= trip_graph.edges[e]['c']
        pdf = np.exp(weights)/np.sum(np.exp(weights))
        path_idx = np.random.choice(len(paths), p=pdf)
        region_paths[id] = path = paths[path_idx]
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
