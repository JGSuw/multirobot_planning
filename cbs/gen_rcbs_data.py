from mapf import *
from rcbs import *
import numpy as np
import networkx as nx
import pickle
from multiprocessing import Pool
import os
import time
from datetime import date
import functools

def solve_problem(env: RegionalEnvironment, 
                  start_vertex: dict, 
                  region_paths: dict, 
                  final_goals: dict,
                  omega: float):
    root = init_rcbs(env, start_vertex, final_goals, region_paths)
    result = regional_cbs(root, env, omega, maxtime=30.)
    if type(result) == RCBSNode:
        return result.make_solution()
    else:
        return None

def main(N_problems: int,   # number of problems to solve
        dir: str,       # output directory
        nrows: int,         # number of rows of subregions
        ncols: int,         # number of columns of subregions
        colh: int,          # height of obstacle columns
        colw: int,          # width of obstacle columns
        dy: int,            # vertical free-space around obstacles
        dx: int,            # horizontal free-space around obstacles
        N_agents: int,      # number of agents
        path_cutoff: int,   # upper limit on length of regional paths
        omega: float,       # suboptimality factor (>= 1.0)
        seed):          # seed value for random number generation

    proc_dir = os.path.join(dir, f'{seed}')
    os.makedirs(proc_dir)
    env = ColumnLatticeEnvironment(nrows, ncols, colh, colw, dy, dx, 2, 2)
    save_environment(env, proc_dir)
    rng = np.random.default_rng(seed=seed)
    for i in range(N_problems):
        problem_data = random_problem(N_agents, env, path_cutoff, rng=rng)
        start_vertex = problem_data[0]
        final_goals = problem_data[1]
        region_paths = problem_data[2]
        save_problem(start_vertex, region_paths, final_goals, os.path.join(proc_dir, f'problem_{i}.pickle'))
        clock_start = time.time()
        solution = solve_problem(env, start_vertex, region_paths, final_goals, omega)
        if solution is not None:
            print(f'Worker {seed} solved MAPF problem {i+1}/{N_problems} in {time.time()-clock_start} seconds.')
            save_solution(solution, os.path.join(proc_dir, f'solution_{i}.pickle'))
        else:
            print(f'Worker {seed} failed to solve MAPF problem {i+1}')
    return None

def save_environment(env: RegionalEnvironment, dir: os.path):
    grid_nodes = [p for p in env.gridworld.G.nodes]
    grid_adj = dict((p,env.gridworld.G.adj[p]) for p in grid_nodes)
    grid_obstacles = env.gridworld.obstacles
    region_nodes = [R for R in env.region_graph.nodes]
    region_adj = dict((R, env.region_graph.adj[R]) for R in region_nodes)
    boundaries = {}
    for R1 in region_nodes:
        for R2 in env.region_graph.adj[R1]:
            boundary = [e for e in env.region_graph.edges[(R1,R2)]['boundary']]
            boundaries[(R1,R2)] = boundary
    env_data = {
        'grid_nodes': grid_nodes,
        'grid_adj': grid_adj,
        'grid_obstacles': grid_obstacles,
        'region_nodes': region_nodes,
        'region_adj': region_adj,
        'boundaries': boundaries
    }
    with open(os.path.join(dir, 'env.pickle'), 'wb') as f:
        pickle.dump(env_data, f)

def save_problem(start_vertex: dict, 
                    region_paths: dict, 
                    final_goals: dict, 
                    file_path: os.path):
    problem_data = {
        'start_pos' : {id : start_vertex[id].pos for id in start_vertex},
        'region_paths' : region_paths,
        'final_goals' : final_goals
    }
    with open(file_path, 'wb') as f:
        pickle.dump(problem_data,f)

def save_solution(solution: MAPFSolution, file_path: os.path):
    vertexes = {}
    edges = {}
    paths = {}
    for id in solution.paths:
        path = solution.paths[id]
        u = path[0]
        paths[id] = [(u.pos, u.t)]
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            paths[id].append((v.pos, v.t))
            e = PathEdge(u.pos, v.pos, u.t)
            if u.pos in vertexes:
                vertexes[u.pos].append((id, u.t))
            else:
                vertexes[u.pos] = [(id, u.t)]
            if (e.p1, e.p2) in edges:
                edges[(e.p1, e.p2)].append((id, u.t))
            else:
                edges[(e.p1, e.p2)] = [(id, u.t)]
    data = {
        'vertexes' : vertexes,
        'edges' : edges,
        'paths' : paths 
    }
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":

    N_cpus = 8
    N_problems = 45
    N_problems_total = N_cpus*N_problems

    now = time.time()
    datestring = date.fromtimestamp(now)
    dir = str(f'{datestring}_rcbs_output')

    nrows = ncols = 5
    path_cutoff = 10
    colh = colw = 4
    dy = dx = 2
    N_agents = 120
    omega = 1.05
    seeds = list(s for s in np.random.randint(0,int(1e5), size=N_cpus))
    with Pool(N_cpus) as p:
        args = (N_problems, 
                dir, 
                nrows, 
                ncols, 
                colh, 
                colw, 
                dy, 
                dx, 
                N_agents, 
                path_cutoff, 
                omega)
        task = functools.partial(main, *args)
        p.map(task, seeds)