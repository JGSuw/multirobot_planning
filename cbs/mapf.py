
import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap as ListedColorMap
from matplotlib import colormaps
import matplotlib.animation as animation
import networkx as nx

class Environment():
    FREE = 0
    OBSTACLE = 1
    def __init__(self):
        raise NotImplementedError()
    def contains_node(u):
        raise NotImplementedError()
    def contains_edge(u,v):
        raise NotImplementedError()
    def dense_matrix(self):
        raise NotImplementedError()

class GridWorld(Environment):
    def __init__(self, size, obstacles: list):
        rows, cols = self.size = size
        nodes = [(i, j) for i in range(rows) for j in range(cols)]
        self.obstacles = obstacles
        self.G = nx.Graph()
        for o in obstacles:
            nodes.remove(o)
        self.G.add_nodes_from(nodes)
        for u in self.G.nodes:
            row,col = u
            self.G.add_edge(u,u)
            if row > 0:
                v = (row-1,col)
                if v in self.G.nodes: self.G.add_edge(u,v)
            if row+1 < rows:
                v = (row+1,col)
                if v in self.G.nodes: self.G.add_edge(u,v)
            if col > 0:
                v = (row, col-1)
                if v in self.G.nodes: self.G.add_edge(u,v)
            if col+1 < cols:
                v = (row, col+1)
                if v in self.G.nodes: self.G.add_edge(u,v)

    def contains_node(self, u: tuple):
        return u in self.G.nodes
    
    def contains_edge(self, u: tuple, v: tuple):
        return (u,v) in self.G.edges

    def dense_matrix(self):
        mat = np.zeros(self.size, dtype=int)
        obstacles = self.obstacles
        for o in obstacles:
            mat[*o] = GridWorld.OBSTACLE
        return mat

# draw the state of the provided environment
def draw_environment(ax, env: Environment, agent_pos: dict, goals: dict, arrows=True, animated=False, grid=False):
    tab20b = colormaps.get_cmap('tab20b')
    colors = tab20b(list(range(20)))
    black = np.array([0,0,0,1])
    white = np.array([1, 1, 1, 1])
    colors[0,:] = white
    colors[1,:] = black
    get_color_idx = lambda id: 2+(id % 18)
    mat = env.dense_matrix()
    for id in goals:
        mat[*agent_pos[id]] = get_color_idx(id)
    
    cmap = ListedColorMap(colors)
    image = ax.imshow(mat, cmap=cmap, animated=True)

    if grid:
        # Major ticks
        ax.set_xticks(np.arange(0, mat.shape[1], 1))
        ax.set_yticks(np.arange(0, mat.shape[0], 1))


        # Minor ticks
        ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

        # Remove minor ticks
        ax.tick_params(which='minor', bottom=False, left=False)

    # make arrows pointing agents to their goals
    if arrows:
        for id in goals:
            start_y, start_x = agent_pos[id] 
            end_y, end_x = goals[id]
            dx = end_x - start_x
            dy = end_y - start_y
            ax.arrow(start_x, start_y, dx, dy, head_width = 0.2, head_length = 0.2, alpha=.5)
    return image

class Constraint:
    def __init__(self):
        raise NotImplementedError()
    def __hash__(self):
        raise NotImplementedError()
    def __eq__(self):
        raise NotImplementedError()

class Location(Constraint):
    def __init__(self, pos: tuple):
        self.pos = pos

    def __hash__(self):
        return self.pos.__hash__()

    def __str__(self):
        return self.pos.__str__()

    # equality operator for comparison in constraint sets 
    def __eq__(self, other):
        return self.pos == other.pos

# Represents a vertex in an agent trajectory
class PathVertex(Constraint):
    def __init__(self, pos: tuple, time: int):
        self.pos = copy.copy(pos)
        self.t = time
    
    def __eq__(self, other):
        if type(other) is PathVertex:
            if self.t==other.t and self.pos==other.pos:
                return True
        return False
    
    def __hash__(self):
        return (self.pos, self.t).__hash__()

    def __gt__(self, other): # used to break ties in the heapq algorithm
        return self.__hash__() > other.__hash__()
    
    def __str__(self):
        return f't = {self.t}, p = {self.pos}'
    
# Represents an Edge in a Path
class PathEdge(Constraint):
    def __init__(self, p1: tuple, p2: tuple, t: int):
        self.p1 = copy.deepcopy(p1)
        self.p2 = copy.deepcopy(p2)
        self.t = t

    # compliment is used for checking for this edge in hash tables
    def compliment(self):
        return PathEdge(self.p2, self.p1, self.t)

    def __eq__(self, other):
        if self.t == other.t:
            if self.p1==other.p1 and self.p2==other.p2:
                return True
            # if self.p1==other.p2 and self.p2==other.p1:
            #     return True
        return False
    
    def __hash__(self):
        return (self.p1, self.p2, self.t).__hash__()

    def __str__(self):
        return f't = {self.t}, p1 = {self.p1}, p2 = {self.p2}'

# Path represents the trajectory of a single agent 
class Path:
    def __init__(self, vertexes):
        self.vertexes = copy.deepcopy(vertexes)

    def insert(self, vertex):
        self.vertexes.append(copy.deepcopy(vertex))

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
    
    def __add__(self, other):
        if self.vertexes[0].t < other.vertexes[0].t:
            first = self
            second = other
        else:
            first = other
            second = self
        if first.vertexes[-1] != second.vertexes[0]:
            raise ValueError('Paths must be sequential.')
        new_path = copy.copy(first)
        for i in range(1,len(second)):
            new_path.insert(copy.copy(second.vertexes[i]))
        return new_path

# Container for constraints to be implemented by a CBS Node 
        
class Goal:
    def __init__(self):
        raise NotImplementedError("Goal __init__ not implemented")

    def heuristic(self, loc):
        raise NotImplementedError("Goal heuristic not implemented")
    
    def satisfied(self, loc):
        raise NotImplementedError("Goal satisfied not implemented")

class LocationGoal(Goal):
    def __init__(self, loc):
        self.loc = loc

    def heuristic(self, loc):
        return abs(self.loc[0]-loc[0])+abs(self.loc[1]-loc[1])
    
    def satisfied(self, loc):
        return self.loc == loc
    
    def __str__(self):
        return f'Location Goal: {self.loc}'
    
class SetGoal(Goal):
    def __init__(self, locs):
        self.set = frozenset(locs)

    def dist(self, loc):
        for p in self.set:
            yield abs(p[0]-loc[0]) + abs(p[1]-loc[1])
        
    def heuristic(self, loc):
        return min(self.dist(loc))
    
    def satisfied(self, loc):
        return loc in self.set
    
    def __str__(self):
        return f'Set Goal: {list(s for s in self.set)}'

class VertexGenerator:
    def __init__(self):
        raise NotImplementedError()
    def vertexes(self, v: PathVertex):
        raise NotImplementedError()

class ActionGenerator:
    def __init__(self, vertex_generator: VertexGenerator, constraints: {}):
        self.vertexes = vertex_generator.vertexes
        self.constraints = constraints
    
    def actions(self, v: PathVertex):
        vertexes = self.vertexes(v)
        for u in vertexes:
            edge = PathEdge(v.pos, u.pos, v.t)
            if edge not in self.constraints:
                if u not in self.constraints:
                    yield (u, edge)

    def apply_constraint(self, constraint):
        self.constraints[constraint]= True

class GridWorldActionGenerator(ActionGenerator):
    def __init__(self, env: GridWorld, constraints = {}):
        self.nodes = lambda v: (p for p in env.G.adj[v.pos])
        self.constraints = constraints
    
    def actions(self, v: PathVertex):
        for pos in self.nodes(v):
            u = PathVertex(pos, v.t+1)
            e = PathEdge(v.pos, pos, v.t)
            if u not in self.constraints:
                if e not in self.constraints:
                    yield (u,e)

# Represents a MAPF problem
class MAPFProblem:
    def __init__(self, agent_ids, start_times, start_pos: dict, goals: dict):
        self.agent_ids = agent_ids
        self.start_times = start_times
        self.start_pos = copy.deepcopy(start_pos)
        self.n_agents = len(goals)
        self.goals = copy.deepcopy(goals)

class MAPFSolution:
    def __init__(self, paths):
        self.paths = copy.deepcopy(paths)
        self.t_min = min(
            vertex.t for id in self.paths for vertex in self.paths[id]
        )
        self.makespan = max(self.path_lengths())
        self.t_max = self.t_min + self.makespan
        self.tspan = (self.t_min, self.t_max)
    
    def path_lengths(self):
        return [len(self.paths[id]) for id in self.paths]
    
    def sum_of_path_lengths(self):
        return sum(self.path_lengths())
    
class MAPFAnimation:
    def __init__(self, env: Environment, solution: MAPFSolution):
        self.frames = []
        fig, ax = plt.subplots(figsize=(8,8))
        self.fig = fig
        self.ax = ax
        final_goals = dict((id, solution.paths[id][-1].pos) for id in solution.paths)
        for t in range(solution.makespan):
            agent_pos = {}
            for id in solution.paths:
                path = solution.paths[id]
                if t < len(path):
                    agent_pos[id] = path[t].pos
                else:
                    agent_pos[id] = path[-1].pos
            self.frames.append([draw_environment(ax, env, agent_pos, final_goals, arrows=False, animated=True)])
            if t == 0:
                draw_environment(ax, env, agent_pos, final_goals, arrows=False, animated=True)
    
    def animate(self):
        print(len(self.frames))
        return animation.ArtistAnimation(self.fig, self.frames, interval=500, repeat_delay=5000, repeat=True, blit = True)
