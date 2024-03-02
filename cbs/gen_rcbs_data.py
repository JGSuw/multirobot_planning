from mapf import *
from regional_mapf import *
import numpy as np
import networkx as nx
import pickle
from multiprocessing import Pool
import os
import time
from datetime import date
import functools

def random_problem(N_agents: int, 
                   env: ColumnLatticeEnvironment, 
                   path_cutoff: int,
                   rng = np.random.default_rng()):
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

def solve_problem(env: RegionalEnvironment, 
                  partial_paths: dict, 
                  region_paths: dict, 
                  final_goals: dict,
                  omega: float):
    node = rcbs_init(env, partial_paths, region_paths, final_goals)
    update_rcbsnode(env, node, [R for R in partial_paths], omega, 30., False)
    result = rcbs(env, node, omega, maxtime=60., cbs_maxtime=30.)
    if result is not None:
        node = result[0]
        paths = {}
        for id in node.region_paths:
            region_path = node.region_paths[id]
            for R in region_path:
                path = node.partial_paths[R][id]
                if id not in paths:
                    paths[id] = path
                else:
                    paths[id] += path
        return MAPFSolution(paths)
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
    env = ColumnLatticeEnvironment(nrows, ncols, colh, colw, dy, dx, 1, 1)
    save_environment(env, proc_dir)
    rng = np.random.default_rng(seed=seed)
    for i in range(N_problems):
        problem_data = random_problem(N_agents, env, path_cutoff, rng=rng)
        partial_paths = problem_data[0]
        region_paths = problem_data[1]
        final_goals = problem_data[2]
        save_problem(partial_paths, region_paths, final_goals, os.path.join(proc_dir, f'problem_{i}.pickle'))
        solution = solve_problem(env, partial_paths, region_paths, final_goals, omega)
        if solution is not None:
            print(f'Worker {seed} solved MAPF problem {i+1}/{N_problems}')
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

def save_problem(partial_paths: dict, 
                    region_paths: dict, 
                    final_goals: dict, 
                    file_path: os.path):
    problem_data = {
        'partial_paths' : partial_paths,
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
    N_cpu_hrs = 1
    # N_problems = N_cpu_hrs*N_cpus*60 # number of problems per worker process
    N_problems = N_cpu_hrs*60
    N_problems_total = N_cpus*N_problems

    now = time.time()
    datestring = date.fromtimestamp(now)
    dir = str(f'{datestring}_rcbs_output')

    nrows = ncols = path_cutoff = 8
    colh = colw = 8
    dy = dx = 2
    N_agents = 100
    omega = 1.05
    seeds = list(range(N_cpus))
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