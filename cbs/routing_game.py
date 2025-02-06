from mapf import *
from networkx import Graph, DiGraph
import math

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
            labels.append(i*k_cols+j)
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
class SimpleRoutingNet(DiGraph):
    def __init__(self, env: GridWorld, partition: VertexPartition):
        DiGraph.__init__(self)
        self.partition = partition
        self.partition_edges = {}
        # loop over partitions to get their boundaries
        boundary_edges = {}
        for label, vertices in partition.items():
            B = []
            self.partition_edges[label] = [] # need this later
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
                grid_edges = U.intersection(V)
                if len(grid_edges) > 0:
                    grid_vertices = set()
                    for grid_edge in grid_edges:
                        grid_vertices = grid_vertices.union(grid_edge)
                    self.add_node(
                        frozenset((all_labels[i], all_labels[j])), 
                        grid_edges = grid_edges,
                        grid_vertices = frozenset(grid_vertices))

        # loop over routing net nodes, adding edges to any edge classes covering
        # the same region
        node_list = list(self.nodes.keys())
        for i in range(len(node_list)-1):
            u = node_list[i]
            u_vertices = self.nodes[u]['grid_vertices']
            for j in range(i+1, len(node_list)):
                v = node_list[j]
                v_vertices = self.nodes[v]['grid_vertices']
                region = tuple(u.intersection(v))
                if len(region) == 1: # no shared boundary
                    dx = np.average([
                        [p[0]-q[0], p[1]-q[1]]
                        for p in u_vertices for q in v_vertices
                    ], axis=0)
                    # dx = dx / np.linalg.norm(dx)

                    # what if we wanted probabilities of going UP,DOWN,LEFT,RIGHT?
                    RIGHT = max(0,dx[0])
                    LEFT = max(0,-dx[0])
                    UP = max(0, dx[1])
                    DOWN = max(0,-dx[1])
                    p = np.array([RIGHT,LEFT,UP,DOWN])
                    p = p / np.sum(p)
                    p_compliment = np.array([p[1],p[0],p[3],p[2]])
                    self.partition_edges[region[0]] += [(u,v),(v,u)]
                    self.add_edge(u,v,flow=0,cost=0,dx=dx,p=p)
                    self.add_edge(v,u,flow=0,cost=0,dx=-dx,p=p_compliment)

class CongestionModel():
    def __init__(self):
        raise NotImplementedError()

    def compute_congestion(edge_flows: dict):
        raise NotImplementedError()
    
class SimpleCongestion(CongestionModel):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def compute_congestion(self, net: SimpleRoutingNet):
        for edge, attrs in net.edges.items():
            other_edge = (edge[1],edge[0])
            other_attrs = net.edges[other_edge] 
            # compute total flow through the region
            if attrs['flow'] > 0:
                linear = -self.a*other_attrs['flow']
                attrs['cost'] = linear + self.b
            else:
                attrs['cost'] = self.b

class BoltzmannCongestion(CongestionModel):
    def __init__(self, offsets: dict, p_min=.01):
        self.offsets = offsets
        self.p_min = p_min

    def compute_congestion(self, net: SimpleRoutingNet):
        # loop over regions
        for region in net.partition:
            # get the graph edges crossing this region
            region_edges = net.partition_edges[region]
            total_flow = 0
            Cov = np.zeros((2,2))
            
            # compute flow vectors, product of scalar flows with edge directions
            N = len(region_edges)
            flows = np.array([net.edges[e]['flow'] for e in region_edges])
            total_flow = sum(flows)
            if total_flow <= 1 or np.sum(flows > 0) == 1:
                for e in region_edges: 
                    attrs = net.edges[e]
                    attrs['cost'] = self.offsets[e]
                continue # skip this region since it has no flow

            weights = np.array([net.edges[e]['flow']/total_flow for e in region_edges])
            flow_vecs = np.zeros((N,2))
            avg_flow_vec = np.zeros((2,))
            p_dir = np.zeros(4)
            for i, e in enumerate(region_edges):
                attrs = net.edges[e]
                attrs['cost'] = self.offsets[e]
                flow_vecs[i,:] = attrs['dx']
                avg_flow_vec += weights[i]*flow_vecs[i,:]
                p_dir += flows[i]*attrs['p']
            diffs = flow_vecs-avg_flow_vec
            p_dir = p_dir / total_flow

            # compute weighted flow_vector covariance
            Cov[0,0] = np.sum(weights * diffs[:,0]**2)
            Cov[1,1] = np.sum(weights * diffs[:,1]**2)
            Cov[0,1] = Cov[1,0] = np.sum(weights * diffs[:,0]*diffs[:,1])
            Cov = Cov / (1-np.sum(weights**2))

            # fuckery

            # loop over edges, computing probability vectors
            N = len(net.partition[region])
            entropy = np.sum([-p*math.log(p) for p in p_dir if p > 0])
            print('\n')
            # print(f'Total flow: {total_flow}')
            # print(f'Average Flow vector: {avg_flow_vec}')
            # print(f'Entropy: {entropy}')
            # print(f'Covariance determinant: {det}')
            for i, e in enumerate(region_edges):
                # print(f'Edge: {e}')
                # print(f'Flow: {flows[i]}')
                # print(f'Direction: {flow_vecs[i]}')
                # print(f'Weight: {weights[i]}')
                # an energy partition function
                if flows[i] <= 0:
                    continue
                foo = flows[i]*flow_vecs[i]
                bar = avg_flow_vec

                E0 = np.dot((foo+bar),Cov@(foo+bar))/np.dot((foo-bar),Cov@(foo-bar))
                # print(f'E0: {E0}')
                j = np.arange(0, (total_flow-flows[i])//np.sqrt(N))
                Z = np.exp(-j*E0/entropy)
                p = Z / np.sum(Z)
                net.edges[e]['cost'] += np.sum(j*p)


class RoutingGame():
    def __init__(   self, 
                    net: SimpleRoutingNet, 
                    congestion_model: CongestionModel ,
                    demands: dict
    ):
        self.net = net
        self.congestion_model = congestion_model
        self.demands = demands
        self.initialize_state()

    def update_edge_costs(self):
        self.congestion_model.compute_congestion(self.net)

    def total_system_cost(self):
        total = 0
        for edge, attrs in self.net.edges.items():
            total += attrs['cost']*attrs['flow']
        return total
    
    def initialize_state(self):
        for edge,attrs in self.net.edges.items():
            attrs['flow'] = 0
            attrs['cost'] = 0
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
                grid_vertices = net.nodes[edge_class]['grid_vertices']
                for vertex in grid_vertices:
                    start_to_vert = manhattan_dist(start_pos, vertex)
                    if start_to_vert < start_dist_lb:
                        start_dist_lb = start_to_vert 
                        start_class = edge_class

        goal_dist_lb = np.inf
        goal_class = None
        for edge_class in net.nodes:
            if goal_region in edge_class:
                grid_vertices = net.nodes[edge_class]['grid_vertices']
                for vertex in grid_vertices:
                    start_to_vert = manhattan_dist(start_pos, vertex)
                    vert_to_goal = manhattan_dist(vertex, goal)
                    if start_to_vert + vert_to_goal < goal_dist_lb:
                        goal_dist_lb = start_to_vert + vert_to_goal
                        goal_class = edge_class

        od_pair = (start_class, goal_class)
        if od_pair in demands:
            demands[od_pair] += 1
        else:
            demands[od_pair] = 1

    return demands

