from mapf import *
from networkx import Graph

"""
Specififes a graph partition as a dictionary of labels for the 
partition elements mapping to the set of vertices contained
by that element.
"""
class VertexPartition(dict):
    def __init__(self, labels: list, subsets: list[frozenset]):
        dict.__init__(self)
        if len(labels) != len(subsets):
            raise ValueError('len(labels) does not equal len(subsets)')
        all_vertices = set()
        for i, subset in enumerate(subsets):
            if len(all_vertices.intersection(subset)) != 0:
                raise ValueError('subsets do not define a partition')
            all_vertices = all_vertices.union(subset)
            self[labels[i]] = subset

"""
Produces a vertex partition of the input gridworld into homogeneous
regions with the specified shape.
"""
def partition_gridworld(env: GridWorld, region_shape: tuple):
    if (env.size[0] % region_shape[0]) != 0:
        raise ValueError('env.size[0] not divisible by region_shape[0]')
    if (env.size[1] % region_shape[1]) != 0:
        raise ValueError('env.size[1] not divisible by region_shape[1]')
    labels = []
    subsets = []
    k_rows = int(env.size[0] / region_shape[0])
    k_cols = int(env.size[1] / region_shape[1])
    for i in range(k_rows):
        start_row = i*region_shape[0]
        stop_row = (i+1)*region_shape[0]
        for j in range(k_cols):
            start_col = j*region_shape[1]
            stop_col = (j+1)*region_shape[1]
            labels.append((i,j))
            subsets.append(
                frozenset(
                    (row,col) 
                    for row in range(start_row, stop_row) 
                    for col in range(start_col, stop_col)
                )
            )
    return VertexPartition(labels, subsets)

"""
Constructs a graph on which a routing game may be defined.

@param env: the GridWorld
@param partition: a VertexPartition of the Gridworld env

@detail: 

This routing network is "simple" in the sense that it has
the fewest number of classes under the following edge relation:

Two edges (e,f) are equivalent implies that there exists
unique regions (r,p) of the partition such that (e,f) lies
in the intersection of edges cut by (r,p).
"""
class SimpleRoutingNet(Graph):
    def __init__(self, env: GridWorld, partition: VertexPartition):
        Graph.__init__(self)
        self.partition = partition

        # loop over partitions to get their boundaries
        boundary_edges = {}
        for label, vertices in partition.items():
            B = []
            for u in vertices:
                for v in env.G.neighbors(u):
                    if v not in vertices:
                        B.append(frozenset((u,v)))
            boundary_edges[label] = frozenset(B)
        
        # loop over boundaries to get edge classes,
        # adding nodes to the routing net
        all_labels = list(boundary_edges.keys())
        for i in range(len(all_labels)-1):
            U = boundary_edges[all_labels[i]]
            for j in range(i+1,len(all_labels)):
                V = boundary_edges[all_labels[j]]
                cut_edges = U.intersection(V)
                if len(cut_edges) > 0:
                    self.add_node(frozenset((all_labels[i], all_labels[j])), cut_edges = cut_edges)

        # loop over routing net nodes, adding edges to any edge classes covering
        # the same region
        node_list = list(self.nodes.keys())
        for i in range(len(node_list)-1):
            u = node_list[i]
            for j in range(i+1, len(node_list)):
                v = node_list[j]
                if len(u.intersection(v)) != 0:
                    self.add_edge(u,v,flow=0,cost=0)

class CongestionModel():
    def __init__(self):
        raise NotImplementedError

    def compute_congestion(edge_flows: dict):
        raise NotImplementedError()
    
class AffineCongestion(CongestionModel):
    def __init__(self, a: dict, b: dict):
        self.a = a
        self.b = b

    def compute_congestion(self, flow_state):
        congestion = {}
        for edge in flow_state:
            congestion[edge] = self.a[edge]*flow_state[edge] + self.b[edge]
        return congestion

class RoutingGame():
    def __init__(   self, 
                    net: SimpleRoutingNet, 
                    congestion_model: CongestionModel ,
                    demands: dict
    ):
        self.net = net
        self.congestion_model = congestion_model
        self.demands = demands
        self.flow_state = {}
        self.edge_costs = {}
        self.reset_state()

    def update_edge_costs(self):
        congestion = self.congestion_model.compute_congestion(self.flow_state)
        self.edge_costs.update(congestion)

    def total_system_cost(self):
        return sum(self.edge_costs[edge] * flow 
                  for edge, flow in self.flow_state.items())
    
    def reset_state(self):
        for edge in self.net.edges:
            self.flow_state[frozenset(edge)] = 0
            self.edge_costs[frozenset(edge)] = 0
        self.update_edge_costs()

def make_demands(net: SimpleRoutingNet, prob: MAPFProblem):
    demands = {}
    manhattan_dist = lambda u,v: abs(u[0]-v[0]) + abs(u[1])-abs(v[1])
    for agent_id, start_pos in prob.start_pos.items():
        goal = prob.goals[agent_id]
        start_region = next(label 
                      for label in net.partition 
                      if start_pos in net.partition[label]
        )
        goal_region = next(label
                            for label in net.partition
                            if goal in net.partition[label]
        )
        if start_region == goal_region:
            continue # this agent does not appear in the problem
        start_dist_lb = np.inf
        start_class = None

        for edge_class in net.nodes:
            if start_region in edge_class:
                cut_edges = net.nodes[edge_class]['cut_edges']
                for edge in cut_edges:
                    for edge_pos in edge:
                        start_to_edge = manhattan_dist(start_pos, edge_pos)
                        if start_to_edge < start_dist_lb:
                            start_dist_lb = start_to_edge
                            start_class = edge_class

        goal_dist_lb = np.inf
        goal_class = None
        for edge_class in net.nodes:
            if goal_region in edge_class:
                cut_edges = net.nodes[edge_class]['cut_edges']
                for edge in cut_edges:
                    for edge_pos in edge:
                        start_to_edge = manhattan_dist(start_pos, edge_pos)
                        edge_to_goal = manhattan_dist(edge_pos, goal)
                        if start_to_edge + edge_to_goal < goal_dist_lb:
                            goal_dist_lb = start_to_edge + edge_to_goal
                            goal_class = edge_class

        od_pair = (start_class, goal_class)
        if od_pair in demands:
            demands[od_pair] += 1
        else:
            demands[od_pair] = 1

    return demands



