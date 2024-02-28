from mapf import *
import heapq 
import copy
import time

class CBSNode:
    def __init__(self, 
                 paths: dict,
                 goals: dict, 
                 action_generator: ActionGenerator,
                 agent_constraints = {},
                 cost_offset = 0):
        self.cost_offset = cost_offset
        self.goals = goals
        self.paths = paths
        self.agent_constraints = agent_constraints
        self.agent_constraint_count = 0
        self.action_generator= action_generator 
        self.agent_constraints = {}
        self.cost = 0.

    def apply_constraint(self, id: int, constraint):
        if id not in self.agent_constraints:
            self.agent_constraints[id] = {}
        self.agent_constraints[id][constraint] = True

    def __gt__(self, other):
        return self.cost - self.cost_offset > other.cost - other.cost_offset
    
    def __lt__(self, other):
        return self.cost - self.cost_offset < other.cost - other.cost_offset

def astar(
    action_gen,
    path: Path,
    goal: Goal, 
    agent_constraints, 
    maxtime = 1):
    clock_start = time.time()

    # admissible heuristic function
    h = lambda loc: goal.heuristic(loc)

    # scores
    g = {}
    f = {}
    predecessor = {}
    
    # priority queue
    queue = []
    queue_finder = {}

    v = path[0]
    if v in agent_constraints:
        print('A* infeasibility')
        return None, np.inf
    predecessor[v] = None
    g_score = 0
    f_score = g_score + h(v.pos)
    g[v] = g_score
    f[v] = f_score 
    entry = (f_score, v)
    queue_finder[v] = entry
    heapq.heappush(queue, entry)

    v_last = v 
    for i in range(1,len(path)):
        v_next = path[i]
        edge = next((e for (u,e) in action_gen(v_last) if u == v_next), None)
        if edge is None:
            break # end of warmstart
        if v_next in agent_constraints or edge in agent_constraints:
            break # end of warmstart
        else: 
            predecessor[v_next] = v_last
            g_score = g[v_last]+1
            f_score = g_score + h(v_next.pos)
            g[v_next] = g_score
            f[v_next] = f_score
            entry = (f_score, v_next)
            queue_finder[v_next] = entry
            heapq.heappush(queue, entry)
        v_last = v_next
            
    while len(queue) > 0:
        fscore, v_last = heapq.heappop(queue)
        queue_finder.pop(v_last)
        if goal.satisfied(v_last.pos): 
            v = v_last
            # reconstruct the path
            vertexes = []
            while predecessor[v] != None:
                vertexes.append(v)
                v = predecessor[v]
            vertexes.append(v)
            vertexes.reverse()
            path = Path(vertexes)
            return Path(vertexes), len(path)

        if (time.time()-clock_start) > maxtime: # we have failed
            print('A* timeout')
            return None, np.inf
        
        # get new nodes
        new_nodes = []
        for (v,e) in action_gen(v_last):
            if v in agent_constraints or e in agent_constraints:
                continue # skip this vertex
            new_nodes.append(v)

        # update scores for new nodes
        for v in new_nodes:
            if v in queue_finder:
                entry = queue_finder[v]
                if g[v_last] + 1 < g[v]:
                    predecessor[v] = v_last
                    # compute scores
                    g_score = g[v_last] + 1
                    f_score = g_score + h(v.pos)
                    # update the heap
                    entry = (fscore, v)
                    queue_finder[v] = entry
                    heapq.heapify(queue)
                    # update maps
                    g[v] = g_score
                    f[v] = f_score
            else:
                predecessor[v] = v_last
                # compute scores
                g_score = g[v_last]+1
                f_score = g_score + h(v.pos)
                # update the heap
                entry = (f_score, v)
                queue_finder[v] = entry
                heapq.heappush(queue, entry)
                # update maps
                g[v] = g_score
                f[v] = f_score
    # del queue
    return None, np.inf

def low_level_solve(node: CBSNode, agents_to_update, astar_maxtime):
    for id in agents_to_update:
        if id in node.agent_constraints:
            agent_constraints = node.agent_constraints[id]
        else:
            agent_constraints = {}
        path, cost = astar(
            node.action_generator.actions,
            node.paths[id],
            node.goals[id], 
            agent_constraints,
            maxtime = astar_maxtime)
        if cost < np.inf:
            node.paths[id] = path
            continue
        else:
            node.cost = np.inf
            return
    node.cost = sum(len(node.paths[id]) for id in node.paths)

def detect_conflicts(paths: dict):
    vertexes = {}
    edges = {}
    final_pos = dict((paths[id][-1].pos, id) for id in paths)
    final_t = dict((id, paths[id][-1].t) for id in paths)
    for id in paths:
        path = paths[id]
        for i in range(len(path)-1):
            v1 = path[i]
            v2 = path[i+1]
            e = PathEdge(v1.pos, v2.pos, v1.t)
            
            # Check for vertex constraints
            if v1 in vertexes:
                other = vertexes[v1]
                if other[0] != id:
                    return [(id,v1,*other)]

            if v2.pos in final_pos:
                other_id = final_pos[v2.pos]
                if id != other_id:
                    if v2.t >= final_t[other_id]:
                        return [(id, e, None, None)]
            
            if v2 in vertexes:
                other = vertexes[v2]
                if id != other[0]:
                    return[(id, e, *other)]

            # case 3
            if e.compliment() in edges:
                other = edges[e.compliment()]
                if other[0] != id:
                    return [(id,e,*other)]

            # update dictionaries
            else:
                vertexes[v1]=(id,v1)
                edges[e]=(id,e)

    return []

def conflict_based_search(
        start_node: CBSNode,
        maxtime = 30.,
        astar_maxtime = 5.,
        verbose = False
    ):
    agents_to_update = [id for id in start_node.goals]
    low_level_solve(start_node, agents_to_update, astar_maxtime)
    if start_node.cost < np.inf:
        queue = [start_node]
    else:
        if verbose:
            print('infeasible cbs problem')
        return start_node
    clock_start = time.time()
    while len(queue) > 0:
        node = heapq.heappop(queue)
        conflicts = detect_conflicts(node.paths)
        if len(conflicts) > 0:
            id1, c1, id2, c2 = conflicts[0]
            if verbose:
                print(f'conflict between {id1} at {c1} and {id2} at {c2}')
            if id2 is not None:
                new_node = copy.deepcopy(node)
                new_node.apply_constraint(id2, c2)
                low_level_solve(new_node, [id2], astar_maxtime)
                if new_node.cost < np.inf:
                    heapq.heappush(queue, new_node)
            node.apply_constraint(id1, c1)
            low_level_solve(node, [id1], astar_maxtime)
            if node.cost < np.inf:
                heapq.heappush(queue, node)
        else:
            if verbose:
                print('cbs problem solved')
            return node
        if (time.time() - clock_start) > maxtime:
            if verbose:
                print('cbs timeout.')
            heapq.heappush(queue, node)
            return None
    if verbose: 
        print('Infeasible problem')
    return None
