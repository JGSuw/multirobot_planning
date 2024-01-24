import cbs
import math
import numpy as np
import copy
import torch

"""
Generates a set of obstacle positions for the unit cell of a "column lattice" type environment.
"""
class ColumnLatticeCell():
    def __init__(self, W: int, H: int, dx: int, dy: int):
        Cx = int((W-3*dx)/2)
        Cy = int((H-3*dy)/2)
        if Cx <= 0:
            raise Exception("Cx = int((W-4*dx)/2) must be greater than 0")
        if Cy <= 0:
            raise Exception("Cy = int((H-4*dy)/2) must be greater than 0")
        
        self.size = (2*Cx+3*dx+1, 2*Cy+3*dy+1)
        self.obstacle_pos = []
        for i in range(2):
            for j in range(2):
                col = i*(Cx+1*dx)+dx
                row = j*(Cy+1*dy)+dy
                self.obstacle_pos += [(row+k, col+l) for k in range(Cy) for l in range(Cx)]

"""
Creates a random MAPFProblem instance with an environment that encapsulates a single
ColumnLatticeCell.
"""
def gen_column_lattice_problem(cell, rng=np.random.default_rng()):
    W,H = cell.size
    boundary = [(0,j) for j in range(W)]
    boundary += [(i,0) for i in range(H)]
    boundary += [(H-1,j) for j in range(W)]
    boundary += [(i,W-1) for i in range(H)]
    n_agents_max = int(math.sqrt(W**2+H**2)/math.sqrt(2))
    n_agents_min = 4
    n_agents = rng.choice(range(n_agents_min+1, n_agents_max+1))
    n_agents_on_boundary = rng.choice(range(n_agents_min, n_agents))
    
    goal_pos = rng.choice(boundary, n_agents, replace=False)
    goals = [tuple(p) for p in goal_pos]
    agent_pos = []
    free_pos = copy.deepcopy(boundary)
    for i in range(n_agents_on_boundary):
        choices = [p for p in free_pos if p != goals[i]]
        p = tuple(rng.choice(choices))
        free_pos.remove(p)
        agent_pos.append(p)
    free_pos = [(i,j) for i in range(1,H-1) for j in range(1,W-1) if (i,j) not in cell.obstacle_pos]
    other_agents = rng.choice(free_pos, n_agents-n_agents_on_boundary, replace=False)
    agent_pos += [tuple(p) for p in other_agents]
    
    env = cbs.Environment(cell.size, cell.obstacle_pos, agent_pos)
    return cbs.MAPFProblem(env,goals)

def agent_on_boundary(W,H,pos):
    if pos[0] == 0 or pos[1] == 0:
        return True
    if pos[0] == H-1 or pos[1] == W-1:
        return True
    return False

def congestion_features(problem: cbs.MAPFProblem, boundary_agents):
        # solve for a* paths for individual agents
        env = problem.env
        H,W = env.size
        goals = problem.goals

        # construct input maps
        N = int(math.sqrt(W**2+H**2)/math.sqrt(2))
        channels = N+1
        maps = torch.zeros((channels, H, W), dtype=torch.float)
        
        # populate obstacles
        for i, pos in enumerate(env.obstacle_pos):
            maps[0,*pos] = 1

        # populate agent a* paths
        astar_delays = torch.zeros(N, dtype=torch.float)
        
        for i, agent in enumerate(boundary_agents):
            path, cost = cbs.single_agent_astar(env, agent, goals[agent])
            astar_delays[i] = cost
            for t,v in enumerate(path.vertexes):
                maps[1+i, *v.pos] = t+1

        # populate goal location for unknown agents
        offset = 1+len(boundary_agents)
        interior_agents = [i for i in range(len(goals)) if not agent_on_boundary(W,H,env.agent_pos[i])]
        for i, agent in enumerate(interior_agents):
            maps[offset + i, *goals[agent]] = -1

        # output mask for truncating model output vector
        output_mask = torch.zeros(N, dtype=torch.float)
        for i in range(N):
            if i < len(boundary_agents):
                output_mask[i] = 1
        return maps, astar_delays, output_mask

"""
Takes a MAPFProblem and associated MAPFSolution, and returns a tuple 
of tensors for training or inference.

x is a (W,H,2N+1) input tensor of the input problem data
y is a (N,) output tensor of labels (delay times)

The entries of y are masked to only include the delays comptued for 
agents that satisfy agents_on_boundary(agent_pos).
"""
def mapfprob2ffconvtensor(problem: cbs.MAPFProblem, solution: cbs.MAPFSolution):
    H,W = problem.env.size
    N = int(math.sqrt(W**2+H**2)/math.sqrt(2))
    x = torch.zeros((2*N+1,H,W), dtype=torch.float32)
    y = torch.zeros(N, dtype=torch.float32)
    for pos in problem.env.obstacle_pos:
        x[0, *pos] = 1
    for i in range(len(problem.goals)):
        pos = problem.env.agent_pos[i]
        if agent_on_boundary(W,H,pos):
            x[i+1, *pos] = 1
            y[i] = len(solution.paths[i])
        pos = problem.goals[i]
        x[N+i+1, *pos] = 1
    return x,y

from torch.utils.data import Dataset

"""
A torch Dataset of MAPF problems generated by gen_column_lattice_problem and their
corresponding solutions.
"""
class ColumnLatticeDataset(Dataset):
    def __init__(self, data_dir):
        # read the data in
        self.X = []
        self.Y = []
        subfolders = [os.path.join(data_dir, p) for p in os.listdir(data_dir)]
        for subfolder in subfolders:
            files = [os.path.join(subfolder, p) for p in os.listdir(subfolder)]
            for path in files:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    if data['solution'] is not None:
                        self.X.append(data['problem'])
                        self.Y.append(data['solution'])
        self.len = len(self.Y)

    def __len__(self):
        return self.len

    """
    Given index idx, return tensors (x,y) from mapfproblem2ffconvtensor.
    x is a (W,H,2N+1) tensor of the input problem data
    y is a (N,) tensor of labels (delay times)

    The entries of y are masked to only include the delays comptued for 
    agents that satisfy agents_on_boundary(agent_pos).
    """
    def __getitem__(self, idx):
        # return mapfprob2ffconvtensor(self.X[idx], self.Y[idx])
        problem = self.X[idx]
        H,W = problem.env.size
        N = int(math.sqrt(W**2+H**2)/math.sqrt(2))
        solution = self.Y[idx]
        agent_pos = problem.env.agent_pos
        agents_on_boundary = [i for i in range(len(problem.goals)) if agent_on_boundary(W,H,agent_pos[i])]
        features = congestion_features(self.X[idx], agents_on_boundary)
        labels = torch.zeros(N, dtype=torch.float)
        for i, agent in enumerate(agents_on_boundary):
            labels[i] = len(solution.paths[agent])
        return features, labels
    

import multiprocessing
import pickle
import os

"""
Subprocess routine for generating training / test data.
"""
def task(args):
    seed, num_examples = args
    rng = np.random.default_rng(seed=seed)
    cell = ColumnLatticeCell(15,15,2,2)
    for i in range(num_examples):
        prob = gen_column_lattice_problem(cell, rng=rng)
        soln = cbs.conflict_based_search(prob, maxtime=3.)
        if soln is not None:
            data = {'problem': prob, 'solution': soln}
            try:
                os.mkdir('data')
            except:
                pass
            try:
                os.mkdir(f'data/process{seed}')
            except:
                pass
            with open(f'data/process{seed}/data{i}.pickle', 'wb') as file:
                pickle.dump(data, file)

"""
If this file is run at the level of main, it will spawn 4 processes
that execute task(args) to generate 100,000 training/test examples.
"""
if __name__ == "__main__":
    n_cpus = 4
    with multiprocessing.Pool(n_cpus) as p:
        total_examples = 100000
        seeds = [18427, 31072, 53165, 57585]
        # seeds = np.random.randint(0,2**16 - 1, n_cpus)
        p.map(task, [(seed, total_examples//n_cpus) for seed in seeds])