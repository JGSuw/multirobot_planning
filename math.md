# Intro
The scope of this document is to describe as concisely as possible the mathematical definition of the multirobot path planning problem. We begin with a single robot case, formulated to leverage network algorithms (djikstra).

## Single Robot Planning - States, Actions, Transition Function
A single robot, moving from an initial position and orientation, to a final position and orientation, in an $ n \times n $ grid-world with obstacles. The position and orientation of the robot are represented by elements of the set

$$ X := \{1,\dots,n\}^2 \times \left\{0,\cfrac{\pi}{2},\pi,\cfrac{3\pi}{2}\right\} $$

where $\theta = 0$ corresponds to the North facing direction and $\theta = \cfrac{\pi}{2}$ is West facing. In addition, we define the *closure* of $X$, denoted $\bar{X}$ as 

$$ \bar{X} := \{0,\dots,n+1\}^2 \times \left\{0,\cfrac{\pi}{2},\pi,\cfrac{3\pi}{2}\right\} $$

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

so that at a state $x$, the action $a$ is permissible if and only if $f(x,a) \in S(x)$. Taking an action $a$ at a state $x$ results in a positive cost

$$C: X \times A \to \mathbb{R} $$
$$ (x,a) \mapsto C(x,a) $$

which is congruent with the permissible actions in the following sense:

$$ \forall (x,a) \in X \times A: f(x,a) \not \in S(x) \implies C(x,a) = \infty$$

Using these maps, we can construct a graph $(X,E)$ suitable for using Dijkstra's algorithm to compute shortest paths:

$$ \forall (x,a) \in X \times A: (x,f(x,a),C(x,a)) \in E \iff f(x,a) \in S(x) $$
