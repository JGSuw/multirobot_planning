import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap as ListedColorMap
import matplotlib.animation as animation
import copy
import heapq

# Enum of possible actions
class Action:
    NUM_ACTIONS = 5
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    WAIT = 4

# Represents the world inhabited by robots and obstacles
class Environment:
    FREE = 0
    OBSTACLE = 1
    AGENT = 2
    GOAL = 3
    GOAL_REACHED = 4

    def __init__(self, size, obstacle_pos, agent_pos):
        self.obstacle_pos = obstacle_pos
        self.agents = list(range(len(agent_pos)))
        self.agent_pos = agent_pos
        self.grid = np.zeros(size,dtype=int)
        for (i,j) in obstacle_pos:
            self.grid[i,j] = self.OBSTACLE
        for i, idx in enumerate(agent_pos):
            self.grid[idx] = self.AGENT
    
    
    def get_obstacles(self):
        return np.copy(self.obstacle_pos)
    
    def get_agents(self):
        return np.copy(self.agent_pos)
    
    def update_agent_pos(self, ids, positions):
        for i,j in enumerate(ids):
            self.grid[self.agent_pos[j]] = self.FREE
            self.grid[positions[i]] = self.AGENT
            self.agent_pos[j] = positions[i]

# draw the state of the provided environment
def draw_environment(ax, env, goals, arrows=True, animated=False):
    mat = np.zeros(np.shape(env.grid), dtype=int)
    for loc in env.obstacle_pos:
        mat[loc] = env.OBSTACLE

    for id in env.agents:
        if env.agent_pos[id] == goals[id]:
            mat[goals[id]] = env.GOAL_REACHED
        else:
            mat[env.agent_pos[id]] = env.AGENT
    
    for id, pos in enumerate(goals):
        if mat[pos] != env.AGENT and mat[pos] != env.GOAL_REACHED:
            mat[pos] = env.GOAL

    colors = ["white", "black", "blue", "green", "gold"]
    cmap = ListedColorMap(colors[0:np.max(mat)+1])
    # cmap = ListedColorMap(colors)
    image = ax.imshow(mat, cmap=cmap, animated=True)

    # make arrows pointing agents to their goals
    if arrows:
        for id in env.agents:
            start_y, start_x = env.agent_pos[id] 
            end_y, end_x = goals[id]
            dx = end_x - start_x
            dy = end_y - start_y
            ax.arrow(start_x, start_y, dx, dy, head_width = 0.2, head_length = 0.2, alpha=.5)
    return image

# Generates a random environment with n_obstacles obstacles
def random_problem(size, n_agents: int, n_obstacles: int, seed=None):
    if seed == None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    loc = [(i,j) for i in range(size[0]) for j in range(size[1])]
    # assign agent locations randomly
    agent_idx = rng.choice(len(loc), size=n_agents, replace=False) 
    agent_pos = [tuple(loc[i]) for i in agent_idx]
    # remove agent locations from loc
    for pos in agent_pos:
        loc.remove(pos)
    # assign goal states 
    goal_idx = rng.choice(len(loc)-1, size=n_agents, replace=False)
    goal_pos = [tuple(loc[i]) for i in goal_idx]
    # remove goal positions from loc
    for pos in goal_pos:
        loc.remove(pos)
    # generate obstacles
    obstacle_idx= rng.choice(len(loc)-1, size=n_obstacles, replace=False)
    obstacle_pos = [tuple(loc[i]) for i in obstacle_idx]
    return Environment(size, obstacle_pos, agent_pos), goal_pos

# Represents a vertex in an agent trajectory
class PathVertex:
    def __init__(self, pos: tuple, time: int):
        self.pos = pos
        self.t = time
    
    def __eq__(self, other):
        if type(other) != PathVertex:
            return False
        if self.pos == other.pos and self.t == other.t:
            return True
        else:
            return False
    
    def __hash__(self):
        return (self.pos, self.t).__hash__()

    def __gt__(self, other): # used to break ties in the heapq algorithm
        return self.__hash__() > other.__hash__()
    
    def __str__(self):
        return f't = {self.t}, p = {self.pos}'

# Path represents the trajectory of a single agent 
class Path:
    def __init__(self, vertexes):
        self.vertexes = vertexes

    def insert(self, vertex):
        self.vertexes.append(vertex)

    def __str__(self):
        return ', '.join([f'({v})' for v in self.vertexes])

    def _gt_(self, other):
        return len(self.vertexes) > len(other.vertexes)

    def _lt_(self, other):
        return len(self.vertexes) < len(other.vertexes)
    
    def __getitem__(self, i):
        return self.vertexes[i]
    
    def __len__(self):
        return len(self.vertexes)

# Represents an Edge in a Path
class PathEdge:
    def __init__(self, p1: tuple, p2: tuple, t: int):
        self.p1 = p1
        self.p2 = p2
        self.t = t

    # compliment is used for checking for this edge in hash tables
    def compliment(self):
        return PathEdge(self.p2, self.p1, self.t)

    def __eq__(self, other):
        if type(other) != PathEdge:
            return False
        if self.t != other.t:
            return False 
        if self.p1 != other.p1 and self.p1 != other.p2:
            return False
        if self.p2 != other.p1 and self.p2 != other.p2:
            return False
        return True
    
    def __hash__(self):
        return (self.p1, self.p2, self.t).__hash__()
    
    def __str__(self):
        return f't = {self.t}, p1 = {self.p1}, p2 = {self.p2}'

# Container for constraints to be implemented by a CBS Node 
class ConstraintSet:
    def __init__(self, data = []):
        self._hashmap = {}
        for x in data:
            self._hashmap[x] = True

    def insert(self, x):
        self._hashmap[x] = True

    def __contains__(self, x):
        if type(x) == PathVertex:
            try:
                return self._hashmap[x]
            except KeyError:
                return False
        elif type(x) == PathEdge:
            try:
                return self._hashmap[x]
            except KeyError:
                try: 
                    return self._hashmap[x.compliment()]
                except KeyError:
                    return False
        else:
            return False

# Computes shortest paths for an individual agent, ignoring other agents
def single_agent_astar(env: Environment, id: int, goal: tuple, constraints=None, maxtime = 200):
    if constraints is not None:
        _constraints = copy.copy(constraints) # shallow copy
    else:
        _constraints = {}

    # admissible heuristic function
    h = lambda pos, goal: abs(pos[0]-goal[0]) + abs(pos[1]-goal[1])

    # starting node
    t = 0
    node = PathVertex(env.agent_pos[id], t)

    # priority queue
    queue = []
    queue_finder = {}
    entry = [h(node.pos, goal), node]
    queue_finder[node] = entry
    heapq.heappush(queue, entry)

    # predecessor map, an empty dictionary
    predecessor = {node: None}

    # scores
    g = {node: 0}
    f = {node: h(node.pos, goal)}

    while len(queue) > 0:
        fscore, node = heapq.heappop(queue)
        queue_finder.pop(node)
        pos = node.pos
        t = node.t

        if pos == goal: # we have succeeded
            cost = 0
            v = node
            # reconstruct the path
            vertexes = []
            while predecessor[v] != None:
                cost += 1
                vertexes.append(v)
                v = predecessor[v]
            vertexes.append(v)
            vertexes.reverse()
            return Path(vertexes), cost

        if t > maxtime: # we have failed
            print("A* timeout")
            return None, np.inf

        # generate nodes, checking against constraints.
        new_nodes = []

        # PathVertex resulting from WAIT
        v = PathVertex(pos, t+1)
        if not (v in _constraints):
            new_nodes.append(v)

        # PathVertex resulting from UP
        if pos[1] > 0:
            v = PathVertex((pos[0], pos[1]-1), t+1)
            edge = PathEdge(pos, v.pos, t)
            if (env.grid[v.pos] != env.OBSTACLE):
                if (v not in _constraints) and (edge not in _constraints):
                    new_nodes.append(v)

        # PathVertex resulting from DOWN 
        if pos[1]+1 < np.size(env.grid, 1):
            v = PathVertex((pos[0], pos[1]+1), t+1)
            edge = PathEdge(pos, v.pos, t)
            if (env.grid[v.pos] != env.OBSTACLE):
                if (v not in _constraints) and (edge not in _constraints):
                    new_nodes.append(v)

        # PathVertex resulting from LEFT
        if pos[0] > 0:
            v = PathVertex((pos[0]-1, pos[1]), t+1)
            edge = PathEdge(pos, v.pos, t)
            if (env.grid[v.pos] != env.OBSTACLE):
                if (v not in _constraints) and (edge not in _constraints):
                    new_nodes.append(v)

        # PathVertex resulting from RIGHT
        if pos[0]+1 < np.size(env.grid,0):
            v = PathVertex((pos[0]+1, pos[1]), t+1)
            edge = PathEdge(pos, v.pos, t)
            if (env.grid[v.pos] != env.OBSTACLE):
                if (v not in _constraints) and (edge not in _constraints):
                    new_nodes.append(v)

        # update scores for new nodes
        for v in new_nodes:
            try:
                entry = queue_finder[v]
                if g[node] + 1 < g[v]:
                    predecessor[v] = node
                    # compute scores
                    g_score = g[node] + 1
                    f_score = g_score + h(v.pos, goal)
                    # update the heap
                    entry[0] = fscore
                    heapq.heapify(queue)
                    # update maps
                    g[v] = g_score
                    f[v] = f_score
            except KeyError:
                predecessor[v] = node
                # compute scores
                g_score = g[node]+1
                f_score = g_score + h(v.pos, goal)
                # update the heap
                entry = [f_score, v]
                queue_finder[v] = entry
                heapq.heappush(queue, entry)
                # update maps
                g[v] = g_score
                f[v] = f_score
    return None, np.inf

# Detects vertex and edge conflicts for a list of Path objects
def detect_conflicts(paths):
    M = len(paths)
    constraints = [ConstraintSet() for i in range(len(paths))]
    conflicts = []
    agents_at_goal = []
    if M <= 1:
        return conflicts
    for i, path in enumerate(paths):
        for t in range(len(path)):
            vertex = path[t]
            edge = None
            if t < len(path)-1:
                edge = PathEdge(path[t].pos, path[t+1].pos,t)
            if vertex not in constraints[i]:
                constraints[i].insert(vertex)
            if edge not in constraints[i]:
                constraints[i].insert(edge)
            vertex_tests = [(j != i) and (vertex in constraints[j]) for j in range(M)]
            edge_tests = [(j != i) and (edge in constraints[j]) for j in range(M)]
            for j in range(M):
                if vertex_tests[j]:
                    conflicts.append((i,j,vertex))
                if edge_tests[j]:
                    conflicts.append((i,j,edge))
    return conflicts

# Implements a conflict tree node for conflict based search
class CBSNode:
    def __init__(self):
        self.constraints = {}
        self.paths = []
        self.cost = 0

    def set_paths(self, paths):
        self.paths = copy.deepcopy(paths)
        
    def set_constraints(self, constraints):
        self.constraints = copy.deepcopy(constraints)

    # Creates two child nodes given a pair of agents and a constraint
    def branch(self, agent1, agent2, constraint):
        left_constraints = copy.copy(self.constraints)
        right_constraints = copy.copy(self.constraints) 
        try:
            left_constraints[agent1].insert(constraint)
        except KeyError:
            constraint_set = ConstraintSet()
            constraint_set.insert(constraint)
            left_constraints[agent1] = constraint_set
        try:
            right_constraints[agent2].insert(constraint)
        except KeyError:
            constraint_set = ConstraintSet()
            constraint_set.insert(constraint)
            right_constraints[agent2] = constraint_set
        left_node = CBSNode()
        left_node.set_constraints(left_constraints)
        right_node = CBSNode()
        right_node.set_constraints(right_constraints)
        return left_node, right_node

    def __gt__(self, other):
        return self.cost > other.cost
    
    def __lt__(self, other):
        return self.cost < other.cost

# Represents a MAPF problem
class MAPFProblem:
    def __init__(self, environment, goals):
        if len(goals) != len(environment.agent_pos):
            raise ValueError("Goal states must match number of agents in environment")
        self.n_agents = len(goals)
        self.env = environment
        self.goals = goals

# Represents a MAPF solution
class MAPFSolution:
    def __init__(self, paths):
        self.paths = paths
        self.makespan = max([len(path) for path in paths])
    
    def __str__(self):
        n_agents = len(self.paths)
        mat = np.array([
            [f't = {i}' for i in range(self.makespan)]]
        )
        for i in range(n_agents):
            arr = []
            T = len(self.paths[i])
            for t in range(self.makespan):
                if t < T:
                    arr.append(f'{self.paths[i][t].pos}')
                else:
                    arr.append(None)
            mat = np.vstack((mat, arr)) 
        return np.array_str(mat)

# Low level solving stage of CBS, computes constrained shortest paths
# by invoking single_agent_astar for all agents, given node constraints
def low_level_solve(prob, node):
    M = len(prob.env.agent_pos)
    for i in range(M):
        constraints = None
        try:
            constraints = node.constraints[i]
        except KeyError:
            pass
        # grid_size = np.size(prob.grid)
        path, cost = single_agent_astar(prob.env, i, prob.goals[i], constraints, maxtime = 100)
        node.paths.append(path)
        node.cost += cost
        if node.cost < np.inf:
            continue
        else:
            return

def conflict_based_search(prob):
    M = len(prob.env.agent_pos)
    # compute individual paths for root node
    root = CBSNode()
    low_level_solve(prob, root)
    if root.cost < np.inf:
        pass
    else:
        return None
    # place root node into priority queue
    queue = [root]
    while len(queue) > 0:
        # pop top of queue and check for conflicts
        node = heapq.heappop(queue)
        conflicts = detect_conflicts(node.paths)
        if len(conflicts) > 0:
            # print("Branching")
            i,j,c = conflicts[0]
            left_node, right_node = node.branch(i,j,c)
        else: # if no conflicts, then return solution
            return MAPFSolution(node.paths)
        # invoke A* on the child nodes, and compute their sum of individual costs
        low_level_solve(prob, left_node)
        low_level_solve(prob, right_node)
        # push new nodes onto the queue
        if left_node.cost < np.inf:
            heapq.heappush(queue, left_node)
        if right_node.cost < np.inf:
            heapq.heappush(queue, right_node)
    return None

# Represents a "Meta-Agent", which is a subset of agents in a MAPF problem
class MetaAgent:
    def __init__(self, agent_ids, sort=True):
        if sort:
            self.agent_ids = tuple(sorted(agent_ids))
        else:
            self.agent_ids = tuple(agent_ids)

    def __add__(self, other): # set union, in order assuming the input agents are in order
        agent_ids = []
        i = 0; N = len(self.agent_ids)
        j = 0; M = len(other.agent_ids)
        while i < N or j < M:
            if i == N:
                if other.agent_ids[j] <= agent_ids[-1]:
                    j += 1
                else:
                    agent_ids.append(other.agent_ids[j])
            elif j == M:
                if self.agent_ids[i] <= agent_ids[-1]:
                    i += 1
                else:
                    agent_ids.append(self.agent_ids[i])
            elif self.agent_ids[i] == other.agent_ids[j]:
                agent_ids.append(self.agent_ids[i])
                i += 1
                j += 1
            elif self.agent_ids[i] < other.agent_ids[j]:
                agent_ids.append(self.agent_ids[i])
                i += 1
            else:
                agent_ids.append(other.agent_ids[j])
                j += 1

        return MetaAgent(agent_ids, sort=False)

    def __hash__(self):
        return self.agent_ids.__hash__()

# Creates a MAPF subproblem for a given meta agent
def make_ma_subproblem(prob, meta_agent):
    goals = [prob.goals[i] for i in meta_agent.agent_ids]
    agent_pos = [prob.env.agent_pos[i] for i in meta_agent.agent_ids]
    obstacle_pos = prob.env.obstacle_pos
    size = np.shape(prob.env.grid)
    return MAPFProblem(Environment(size, obstacle_pos, agent_pos), goals)

# Implementation of MA-CBS
def ma_cbs(prob):
    # initial breakdown of problem into single agent tasks
    n_agents = prob.n_agents
    meta_agents = [MetaAgent((id,)) for id in range(n_agents)]
    solutions = dict((meta_agent, None) for meta_agent in meta_agents)
    while True:
        # update paths for meta agents
        for meta_agent in meta_agents:
            if solutions[meta_agent] == None:
                sub_problem = make_ma_subproblem(prob, meta_agent)
                solutions[meta_agent] = conflict_based_search(sub_problem)

        # detect conflicts for meta-agent merging
        merged = {}
        N_meta_agents = len(meta_agents)
        for i in range(N_meta_agents-1):
            meta_agent_i = meta_agents[i]
            for j in range(i+1, N_meta_agents):
                meta_agent_j = meta_agents[j]
                paths = copy.copy(solutions[meta_agent_i].paths)
                paths += copy.copy(solutions[meta_agent_j].paths)
                conflicts = detect_conflicts(paths)
                if len(conflicts) > 0:
                    if meta_agent_i in merged:
                        _a = merged[meta_agent_i]
                    else:
                        _a = meta_agent_i
                    if meta_agent_j in merged:
                        _b = merged[meta_agent_j]
                    else:
                        _b = meta_agent_j
                    new_meta_agent = _a + _b
                    merged[meta_agent_i] = new_meta_agent
                    merged[meta_agent_j] = new_meta_agent
        if len(merged) == 0:
            # construct a mapf solution from the subproblem solutions
            agent_finder = {}
            for meta_agent in meta_agents:
                for j, id in enumerate(meta_agent.agent_ids):
                    agent_finder[id] = (meta_agent, j)
            paths = []
            for id in range(prob.n_agents):
                meta_agent, j = agent_finder[id]
                paths.append(solutions[meta_agent].paths[j])
            return MAPFSolution(paths)
        else:
            # replaces old meta-agents
            for meta_agent in merged.keys():
                new_meta_agent = merged[meta_agent]
                del solutions[meta_agent]
                solutions[new_meta_agent] = None
            meta_agents = list(solutions.keys())

# Class is used to build videos of solutions to MAPF problems.
class MAPFAnimation:
    def __init__(self, prob, solution):
        self.prob = prob
        self.solution = solution
        self.frames = []
        _env = copy.copy(prob.env)
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        for t in range(solution.makespan):
            agent_ids = [i for i in range(prob.n_agents) if len(solution.paths[i]) >= t+1]
            positions = [solution.paths[i][t].pos for i in agent_ids]
            _env.update_agent_pos(agent_ids, positions)
            self.frames.append([draw_environment(ax,_env,prob.goals,arrows=False,animated=True)])
            if t == 0:
                draw_environment(ax,_env, prob.goals, arrows=False, animated=True)
    
    def animate(self):
        return animation.ArtistAnimation(self.fig, self.frames, interval=500, repeat_delay=5000, repeat=True, blit = True)