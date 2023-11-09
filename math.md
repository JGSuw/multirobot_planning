# Intro
The scope of this document is to describe as concisely as possible the mathematical definition of the multirobot path planning problem. We begin with a single robot case, formulated to leverage network algorithms (djikstra).

## Single Robot Planning - States, Actions, Transition Function
A single robot, moving from an initial position and orientation, to a final position and orientation, in an $ n \times n $ grid-world with obstacles. The position and orientation of the robot are represented by elements of the set

{% raw $}
$$ X := \{1,\dots,n\}^2 \times \left \{0,\cfrac{\pi}{2},\pi,\cfrac{3\pi}{2}\right \} $$
{% endraw $}

where $\theta = 0$ corresponds to the North facing direction and $\theta = \cfrac{\pi}{2}$ is West facing. In addition, we define the *closure* of $X$, denoted $\bar{X}$ as 

{% raw $}
$$ \bar{X} := \{0,\dots,n+1\}^2 \times \left \{ 0,\cfrac{\pi}{2},\pi,\cfrac{3\pi}{2} \right \} $$
{% endraw $}

so as to include the boundary of the grid world. From some $x \in X$ the robot can take actions from the *action space*, defined by the ordered set $$A := \{\text{wait}, \text{left}, \text{right}, \text{forward}\} \approx \{0,1,2,3\}$$.

Given some $x \in X$ and $a \in A$, the robot may transition to a new state $x'$ given by the map

$$ f: X \times A \mapsto \bar{X} $$
$$ (x,a) \mapsto f(x,a) = x' $$

this transition function is so defined:

$$
f((i,j,\psi),a) := \begin{cases}
 (i,j,\psi), & \text{for } a = 0 \\
 (i,j,\psi-.5\pi), & \text{for } a = 1 \\
 (i,j,\psi+.5\pi), & \text{for } a = 2 \\
 (i-\sin\psi, j-\cos\psi, \psi), & \text{for } a = 3
\end{cases}
$$

## Single Robot Planning - Constraints, Costs, and Graph Representation

In a given state only some actions are permissible, due to the presence of obstacles or the boundary of the space. This is expressed by restricting the post action state $x'$ from a given state $x$ via the mapping 

$$ S: X \to \mathbb{P}(X) $$

$$ x \mapsto S(x) $$

so that at a state $x$, the action $a$ is permissible if and only if $f(x,a) \in S(x)$. 

In the case for a single robot in a static environment, the mapping $S$ can be represented by a matrix $S_{ij}$ of ones and zeros, with $1$ corresponding to obstacle, and $0$ corresponding to free space. In this way, $x \mapsto S(x) := \{ (i,j,\psi) \in X | S_{ij} = 0\}$.

Taking an action $a$ at a state $x$ results in a positive cost

$$C: X \times A \to \mathbb{R} $$
$$ (x,a) \mapsto C(x,a) $$

which is congruent with the permissible actions in the following sense:

$$ \forall (x,a) \in X \times A: f(x,a) \not \in S(x) \implies C(x,a) = \infty$$

Similarly to $S$, we can represent the mapping $C$ as a tensor indexed as $C_{ij}^a$ where $((i,j,\psi),a) \mapsto C_{ij}^a$.

Using these maps, we can construct a graph $(V,E)$ suitable for using Dijkstra's algorithm to compute shortest paths. First, we construct $V$ to satisfy the formula

$$ x' \in V \iff \exists x \in X \text{ s.t. } x' \in S(x) $$

For the edges $E$, we require 

$$ (x,f(x,a),C(x,a)) \in E \iff f(x,a) \in S(x) $$

This graph has $\mathcal{O}(|E|) = \mathcal{O}(n^2)$ and $\mathcal{O}(|V|) = \mathcal{O}(n^2)$. Hence the worst case asymptotic time complexity of Dijkstra's shortest path algorithm is $\mathcal{O}(n^2(1+2\log(n))$

## Multirobot Path Planning - States, Actions, Transition Function

The multirobot case is constructed similarly to the single robot case, but has exponentially worse complexity. Given $m$ robots, the joint statespace is the product set $X^m$, and the joint action space $A^m$, so that at some epoch, a robot $j \in \{1,\dots,m\}$ may take action $a_j \in A$ at state $x_j \in X$.

The joint transition function is denoted

$$ F: X^m \times A^m $$
$$ F(x,a) = x' \iff x'_i = f(x_i,a_i) \hspace{2mm} \forall i \in \{1,\dots,m\}$$

this map operates on compound states $x \in X^m$ and joint actions $a \in A^m$.

## Multirobot Path Planning - Constraints, Costs, and Graph Representation

The permissible actions for the multirobot case are implicitly defined by extending the mapping $S$ to the multirobot case in the following way:

$$ S: X^m \to \mathbb{P}(X^m)$$
$$(x) \mapsto S(x) := \{x' \in X^m | S_{p(x'_i)} = 0 \text{ and } p(x'_i) \neq p(x'_j) \text{, for all } j \neq i \}$$

here, $p:X \to \{1,\dots,n\}^2$ is the position map, which acts as $p((i,j,\psi)) \mapsto (i,j)$.

As for the cost, it is now a map $C: X^m \mapsto \mathbb{R}^m$, and can be constructed from the single robot cost tensor $C_{ij}^a$ modified to be congruent with $S$ so that impermissible actions result in infinite costs i.e. given $x' = f(x,a)$ satisfies the formulas

$$S_{p(x'_i)} = 1 \implies C_i(x,a) = \infty $$

$$p(x'_i) = p(x'_j) \implies C_i(x,a) = C_j(x,a) = \infty$$

Constructing a graph to represent path planning for multiple robots is more complicated than the single robot case, but proceeds in the same way. We begin by constructing the vertex set $V$ to satisfy the formula

$$ x' \in V \iff \exists x \in X^m \text{ such that } x' \in S(x) $$

Then, the edges $E$ are constructed to satisfy the formula

$$ (x, F(x,a), C(x,a)) \in E \iff F(x,a) \in S(x) $$

To elucidate the complexity of running Dijkstra's algorithm on the multirobot case, we consider some conservative upper bounds on $|E|$ and $|V|$. To begin with, we have $|V| = | \bigcup_{x \in X^m} S(x) | < n^{2m}$. For the edges $|E|$ we first observe that $|A| = 4$. Given a vertex $x \in V$, for robot $i$ there are at most $|A|$ "adjacent" states. It then follows that there are at most $4^m$ edges leaving a state $x \in X^m$. Hence, $|E|$ is bounded above by $4^m |V| < 4^m n^{2m}$. 

In summary, $|V|$ < $n^{2m}$ and $|E| < 4^m n^{2m}$, giving an upper bound on the worst-case complexity of Dijkstra's algorithm to be $4^m n^{2m}(1+2m \log n)$ which is really really bad.