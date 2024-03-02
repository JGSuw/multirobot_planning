from mapf import *
import heapq 
import copy
import time

class CBSNode:
    def __init__(self, 
                 paths: dict,
                 goals: dict, 
                #  action_generator: ActionGenerator,
                 cost_offset = 0):
        self.cost_offset = cost_offset
        self.goals = goals
        self.paths = paths
        self.agent_constraint_count = 0
        # self.action_generator= action_generator 
        self.agent_constraints = {}
        self.path_constraints = {}
        self.cost = 0.

    def apply_agent_constraint(self, id: int, constraint):
        if id not in self.agent_constraints:
            self.agent_constraints[id] = {}
        self.agent_constraints[id][constraint] = True

    def apply_path_constraint(self, constraint):
        self.path_constraints[constraint] = True

    def __gt__(self, other):
        return self.cost - self.cost_offset > other.cost - other.cost_offset
    
    def __lt__(self, other):
        return self.cost - self.cost_offset < other.cost - other.cost_offset

def astar(
    action_gen: ActionGenerator.actions,
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
    g_score = 1
    f_score = g_score + h(v.pos)
    g[v] = g_score
    f[v] = f_score 
    entry = [f_score, v]
    queue_finder[v] = entry
    heapq.heappush(queue, entry)

    # v_last = v 
    # for i in range(1,len(path)):
    #     v_next = path[i]
    #     e = PathEdge(v_last.pos, v_next.pos, v_last.t)
    #     if v_next in agent_constraints or e in agent_constraints or e.compliment() in agent_constraints:
    #         break # end of warmstart
    #     else: 
    #         predecessor[v_next] = v_last
    #         g_score = g[v_last]+1
    #         f_score = g_score + h(v_next.pos)
    #         g[v_next] = g_score
    #         f[v_next] = f_score
    #         entry = [f_score, v_next]
    #         queue_finder[v_next] = entry
    #         heapq.heappush(queue, entry)
    #     v_last = v_next
            
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
            if v in agent_constraints or e in agent_constraints or e.compliment() in agent_constraints:
                continue # skip this vertex
            new_nodes.append(v)

        # update scores for new nodes
        for v in new_nodes:
            if v in g:
                if g[v_last] + 1 < g[v]:
                    predecessor[v] = v_last
                    # compute scores
                    g_score = g[v_last] + 1
                    f_score = g_score + h(v.pos)
                    # update the heap
                    if entry in queue_finder:
                        entry = queue_finder[v]
                        entry[0] = fscore
                        heapq.heapify(queue)
                    else:
                        entry = [f_score, v]
                        heapq.heappush(queue, entry)
                    # update maps
                    g[v] = g_score
                    f[v] = f_score
            else:
                predecessor[v] = v_last
                # compute scores
                g_score = g[v_last]+1
                f_score = g_score + h(v.pos)
                # update the heap
                entry = [f_score, v]
                queue_finder[v] = entry
                heapq.heappush(queue, entry)
                # update maps
                g[v] = g_score
                f[v] = f_score
    # del queue
    return None, np.inf

def low_level_solve(action_generator: ActionGenerator.actions, node: CBSNode, agents_to_update, astar_maxtime):
    for id in agents_to_update:
        constraints = copy.deepcopy(node.path_constraints)
        if id in node.agent_constraints:
            constraints.update(node.agent_constraints[id])
        path, cost = astar(
            action_generator,
            node.paths[id],
            node.goals[id], 
            constraints,
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
    conflicts = []
    for id in paths:
        path = paths[id]
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            e = PathEdge(u.pos, v.pos, u.t)

            if v in vertexes:
                other = vertexes[v]
                if other[0] != id:
                    conflicts.append((id,e,*other))
            else:
                vertexes[v] = (id,e)

            if v.pos in final_pos:
                other_id = final_pos[v.pos]
                if id != other_id:
                    if v.t >= final_t[other_id]:
                        conflicts.append((id, e, None, None))

            if e.compliment() in edges:
                other = edges[e.compliment()]
                if other[0] != id:
                    conflicts.append((id,e,*other))
            else:
                edges[e] = (id,e)

    return conflicts

def conflict_based_search(
        action_generator: ActionGenerator.actions,
        start_node: CBSNode,
        maxtime = 30.,
        astar_maxtime = 5.,
        verbose = False
    ):
    agents_to_update = [id for id in start_node.goals]
    low_level_solve(action_generator, start_node, agents_to_update, astar_maxtime)
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
                new_node.apply_agent_constraint(id2, c2)
                low_level_solve(action_generator, new_node, [id2], astar_maxtime)
                if new_node.cost < np.inf:
                    heapq.heappush(queue, new_node)
            node.apply_agent_constraint(id1, c1)
            low_level_solve(action_generator, node, [id1], astar_maxtime)
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
