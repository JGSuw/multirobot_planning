import cbs
import math
import numpy as np
import copy

class ColumnLattice():
    def __init__(self, W: int, H: int, dx: int, dy: int):
        Cx = int((W-3*dx)/2)
        Cy = int((H-3*dy)/2)
        if Cx <= 0:
            raise Exception("Cx = int((W-4*dx)/2) must be greater than 0")
        if Cy <= 0:
            raise Exception("Cy = int((H-4*dy)/2) must be greater than 0")
        
        self.size = (2*Cx+3*dx, 2*Cy+3*dy)
        self.obstacle_pos = []
        for i in range(2):
            for j in range(2):
                col = i*(Cx+1*dx)+dx
                row = j*(Cy+1*dy)+dy
                self.obstacle_pos += [(row+k, col+l) for k in range(Cy) for l in range(Cx)]

def gen_column_lattice_problem(column_lattice, rng=np.random.default_rng()):
    W,H = column_lattice.size
    boundary = [(0,j) for j in range(W)]
    boundary += [(i,0) for i in range(H)]
    boundary += [(H-1,j) for j in range(W)]
    boundary += [(i,W-1) for i in range(H)]
    n_agents_max = int(math.sqrt(W**2+H**2)/math.sqrt(2))
    n_agents_min = 4
    n_agents = rng.choice(range(n_agents_min, n_agents_max+1))
    
    goal_pos = rng.choice(boundary, n_agents, replace=False)
    goals = [tuple(p) for p in goal_pos]
    agent_pos = []
    free_pos = copy.deepcopy(boundary)
    for i in range(n_agents_min):
        choices = [p for p in free_pos if p != goals[i]]
        p = tuple(rng.choice(choices))
        free_pos.remove(p)
        agent_pos.append(p)
    free_pos = [(i,j) for i in range(1,H-1) for j in range(1,W-1) if (i,j) not in column_lattice.obstacle_pos]
    other_agents = rng.choice(free_pos, n_agents-n_agents_min, replace=False)
    agent_pos += [tuple(p) for p in other_agents]
    
    env = cbs.Environment(column_lattice.size, column_lattice.obstacle_pos, agent_pos)
    return cbs.MAPFProblem(env,goals)

import multiprocessing
import pickle
import os

def task(args):
    seed, num_examples = args
    rng = np.random.default_rng(seed=seed)
    column_lattice = ColumnLattice(15,15,2,2)
    for i in range(num_examples):
        prob = gen_column_lattice_problem(column_lattice, rng=rng)
        soln = cbs.conflict_based_search(prob, maxtime=1.)
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

if __name__ == "__main__":
    n_cpus = 4
    with multiprocessing.Pool(n_cpus) as p:
        total_examples = 10000
        # seeds = np.random.randint(0,2**16 - 1, n_cpus)
        seeds = [18427, 31072, 53165]
        p.map(task, [(seed, total_examples//n_cpus) for seed in seeds])