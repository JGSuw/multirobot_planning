# Intro
The scope of this document is to describe as concisely as possible the mathematical definition of the multirobot path planning problem. We begin with a single robot case, formulated to leverage network algorithms (djikstra).

## Single Robot Planning
A single robot, moving from an initial position and orientation, to a final position and orientation, in an $ n \times x $ grid-world with obstacles. The position and orientation of the robot are represented by elements of the set


$ V := \{1, \cdots, n\}^2 \times O $ where $ O $ are the *orientations* $ \{N,S,E,W\} $.

In addition, there is an action set $A := O \cup \{ \emptyset \}$ where $d \in O$ represents motion in the direction $d$ and $\emptyset$ is the null-action (no motion of the robot).

Given an action $a \in A$, the action *acts* upon vertices $(i,j,o) \mapsto a(i,j,o) := \dots \in \{0, \cdots, n+1\}^2 \times O$

Next, we have a obstacle map $X: \{1,\cdots, n\}^2 \to \{0,1\}$ where $X(i,j) = 0$ means there is no obstacle at location $(i,j)$ and vis versa.

Finally, we have a cost-mapping that represents to the cost to move between verticies in $V$, written as

$$ C: V \times A \mapsto \mathbb{R} $$

where $\forall v \in V, \forall a \in A : C(v,a) > 0$ and

$$a(v) \not \in V \lor X(a(v)) = 1 \implies C(v,a) = \infty $$

hence $C(v,a)$ is infinite if taking the action $a$ at vertex $v$ is impossible (or otherwise results in some collision).

### Remark on the computtional complexity of Dijkstra's Algorithm.

The above problem, when represented as a graph, has $|V|$ vertices and $\mathcal{O}(|V|)$ edges. Using Dijkstra's algorithm to compute the shortest paths gives a worst-case asymptotic performance of $\mathcal{O}\left(|E| + |V|\log |V|\right) = \mathcal{O} (|V| \log |V|)$, which expands to 

$$ \mathcal{O} (n^2 \log (n) )$$