# Improving eigenpairs of automated multilevel substructuring with subspace iterations

Jiacong Yin <sup>a,b</sup>, Heinrich Voss <sup>b</sup>, Pu Chen <sup>a,</sup>⇑ 

<sup>a</sup> State Key Laboratoryfor Turbulence and Complex Systems and Department ofMechanics and Aerospace Engineering, College ofEngineering, Peking University, 100871 Beijing, China <sup>b</sup> Institute of Numerical Simulation, Hamburg University of Technology, D-21071 Hamburg, Germany 

## a r t i c l e i n f o

Article history: Received 24 November 2011 Accepted 8 January 2013 

Keywords: Eigenvalue Eigenvector AMLS Subspace iteration 

## a b s t r a c t

This paper improves the eigenpair approximations obtained from the automated multilevel substructuring (AMLS) method by subspace iterations. Two variants of AMLS hybrid Subspace Iteration Method (AMLS-SIMa and AMLS-SIMb) are proposed. AMLS-SIMa is a derivative of the basic subspace iteration by utilizing the AMLS approximations as initial vectors. AMLS-SIMb further takes advantage of the AMLS transformed block diagonal stiffness matrix to avoid factorization of the original stiffness matrix. Numerical experiments show that: (a) the error of AMLS approximate eigenpairs can be significantly reduced with just a few iteration steps; (b) AMLS-SIMb is more efficient than AMLS-SIMa with less execution time. - 2013 Elsevier Ltd. All rights reserved. 

## 1. Introduction

The automated multilevel substructuring (AMLS) method, proposed by Bennighof and his colleagues [1–3], is an efficient condensation method to determine hundreds to thousands of approximate eigenmodes and frequency responses for large and complex structures. Compared with the classical block Lanczos method [4,5], AMLS reduces required computational resources. It was reported [3] that AMLS can extract 824 approximate eigenvalues of an eigenvalue problem with 290-million degrees of freedom on a workstation and requires only half execution time in comparison with the solution by Lanczos method on a supercomputer. Hence, AMLS has been applied to a few engineering areas in recent years, including the frequency response analysis in the automobile industry [6], ship vibrations [7] and electromagnetic problems [8,9]. The standard AMLS only addresses linear eigenvalue problems. Elssel and Voss firstly extended it for gyroscopic [10] and some other nonlinear eigenvalue problems [11,12]. Alternative AMLS schemes for strong and weak fluid–solid interaction problems were researched by Stammberger and Voss as well [13]. 

Nevertheless, the accuracy of AMLS is not as good as classical eigensolution methods since AMLS is only one-shot projection method without any iterative improvement. Bekas and Saad pointed out that AMLS is actually the first order approximation of a corresponding rational eigenvalue problem [14]. The error of AMLS depends on the selected cutoff frequency x that truncates eigenmodes of each substructure. Generally speaking, the higher the cutoff frequency is set, the more accurate are the computed results. Whereas, the choice of $\omega _ { \mathrm { c u t o f f } }$ is heuristic. It is selected according to experiences of analysts. Choosing $\omega _ { \mathrm { c u t o f f } }$ improperly may yield inexact eigenapproximations. 

In 2005, Yang et al. [15] derived a rough error bound of AMLS to evaluate the accuracy. Later a better a priori bound was given by Elssel and Voss [16]. To improve AMLS, the shift acceleration technique was researched by Elssel [11] and Ko et al. [17]. Liao et al. suggested a moment-matching mode selection strategy [18] to catch important eigenmodes of substructures. However, in the sense of structural dynamics, these improvements are not satisfactory because the lower precision of extracted eigenvector approximations does not meet the requirement of differences of eigenvector components in modal strain and stress computations. So that applications of AMLS are mainly limited to the areas of eigenvalue solution and frequency response analysis, in which the accuracy of eigenvectors is of lower importance. 

In order to extend AMLS to be wider applicable in structural dynamics and to decrease the accuracy-dependence on cutoff frequency, this study chooses a different way to improve AMLS, i.e. employing subspace iterations to refine the approximate eigenpairs computed by AMLS. The Subspace Iteration Method (SIM) was developed and so named by Bathe [19]. It is a generalized inverse power method to compute several lowest eigenvalues for large sparse systems. In the past decades, SIM has integrated many useful techniques, including shifts [20,21], Sturm’s sequence check and convergence criteria [22], optimizing the number of iteration vectors [23,24], etc. A recent review of the development of SIM was presented in [25]. 

In this paper we consider two alternatives for improving the eigenvalue and eigenvector approximations obtained from AMLS by SIM. The first one, AMLS-SIMa is a derivative of the Basic Subspace Iteration Method (BSIM), which takes advantage of AMLS only by assigning initial subspace as the AMLS eigenvector approximations. In particular, it requires a triangular factorization of the stiffness matrix. AMLS-SIMb employs the fact, that one essential ingredient of AMLS (besides the reduction of the dimension by modal condensation) consists of transforming the stiffness matrix into a block diagonal form. Utilizing this transformation reduces the cost of the subspace iteration substantially. 

The paper is organized as follows. Section 2 briefly summarizes the main phases of AMLS, namely sub-structuring based on the sparsity of the systems matrices, block Gaussian elimination, modal condensation of substructures, solution of the final condensed system and extraction of eigenvector approximations. Section 3 discusses the proposed two AMLS-SIM variants in detail. Their performance will be evaluated via four engineering problems with different scales in Section 4. Concluding remarks in Section 5 summarize the main ideas. 

## 2. The standard AMLS

The purpose of AMLS for linear (generalized) eigenvalue problems is to find $n _ { p }$ eigenpairs of 

$$
K x = \lambda M x,\tag{1}
$$

in a frequency range of interest. Usually (1) arises in a finite element model and the stiffness matrix $K \in \mathbb { R } ^ { n \times n }$ as well as the mass matrix M $\in \mathbb { R } ^ { n \times n }$ are symmetric positive definite. 

## 2.1. Partition and AMLS tree

Similarly as in the component mode synthesis (CMS) [26,27] a structure is partitioned into a small number of substructures based on the sparsity pattern of the system matrices, but more generally than in CMS these substructures are recursively substructured on a number of levels yielding a tree topology for the substructures. 

We stress the fact that substructuring does not necessarily mean that it is obtained by a domain decomposition on a real structure, but it is understood in a purely algebraic sense. Although substructuring by hand may yield smaller interfaces than the ones obtained by automatic partitioners [16], the dissection of the matrices can be derived by applying a graph partitioner like CHA-CO [28] or METIS [29] to the undirected graph corresponding to the nonzero pattern of the matrices under consideration. These partitioners have been used to construct fill-reducing orderings of sparse matrices for efficient factorization but have shown to be a beneficial basis of substructuring methods as well. 

For convenience to review AMLS, this paper takes a binary partition with only two levels of substructures as an example, but clearly all derived formulae and notations can be generalized to non-binary partitions with more levels of substructures for implementation. 

In that specific two-level binary partition [2], the original stiffness matrix K is reordered as 

$$
P ^ {T} K P = \left( \begin{array}{c c c c c c c} K _ {1 1} & & K _ {1 3} & & & & K _ {1 7} \\ & K _ {2 2} & K _ {2 3} & & & & K _ {2 7} \\ K _ {3 1} & K _ {3 2} & K _ {3 3} & & & & K _ {3 7} \\ & & & K _ {4 4} & & K _ {4 6} & K _ {4 7} \\ & & & & K _ {5 5} & K _ {5 6} & K _ {5 7} \\ & & & K _ {6 4} & K _ {6 5} & K _ {6 6} & K _ {6 7} \\ K _ {7 1} & K _ {7 2} & K _ {7 3} & K _ {7 4} & K _ {7 5} & K _ {7 6} & K _ {7 7} \end{array} \right),\tag{2}
$$

where the permutation matrix P represents the reordering operation of the partition, $K _ { i j } \in \mathbb R ^ { n _ { i } \times n _ { j } }$ and $K _ { j i } = K _ { i j } ^ { T } , ~ ( i , j = 1 , \ldots , 7 )$ . The permuted mass matrix ${ \bf \dot { \cal P } } ^ { T } M { \bf \Phi } { \cal P }$ possesses the same block structure as (2). Usually, the reordering of (K,M) is implemented implicitly. So we do not distinguish between $( P ^ { T } K P , P ^ { T } M P )$ and (K,M) in the following sections of this paper. 

The AMLS partition tree is an important clue for the management of substructures. According to the terminology of computer science, a tree is a data structure that simulates a hierarchical tree structure. As an example, the tree corresponding to (2) is given in Fig. 1, where each circle labeled by a number $s ( s = 1 , . . . , 7 )$ is a treenode that corresponds to a substructure of $( K _ { s s } , M _ { s s } ) .$ . The set of all treenodes on level l is denoted by S<sub>l</sub>, e.g. in Fig. 1, $S _ { l = 2 } = \{ 1 , 2 , 4 , 5 \} , \ S _ { l = 1 } = \{ 3 , 6 \}$ and $S _ { l = 0 } = \{ 7 \}$ . Each treenode s in a tree has zero or more children, which are below it in the tree and their sets are denoted by CðsÞ. A treenode s that has at least one child j is called the child’s parent, denoted by $s = \mathcal { P } ( j )$ . For example, $3 = \mathcal { P } ( 1 )$ and $3 \in \mathcal { C } ( 7 )$ . Treenode 7 is the parent of treenode 1’s parent, so 7 is called an ancestor of 1, denoted by $7 \in \mathcal { A } ( 1 )$ We define that PðsÞ is also an ancestor of treenode s, i.e. ${ \mathcal { P } } ( s ) \in { \mathcal { A } } ( s )$ . Finally, we call the treenode that does not have any ancestor the root of the tree; while those treenodes which do not have any children are called the leaves of the tree. In Fig. 1, the root is treenode 7 and the set of leaves equals to $S _ { l = 2 } .$ 

One of the most important operations on a tree is traversal, moving through all the treenodes of the tree, visiting each one in turn [30]. The creation of an AMLS partition tree follows the sequence of preorder traversal of the tree, i.e. visiting a treenode s always before its children, $\mathcal { C } ( s )$ . Whereas in postorder traversal, each treenode is visited after its children. For Fig. 1, sequences of preorder and postorder traversals are {7,3,1,2,6,4,5} and {1, 2, 3, 4, 5, 6, 7}, respectively. 

## 2.2. The block Gaussian elimination

After the partition, AMLS starts a block Gaussian elimination procedure to transform K into a block diagonal matrix Ke, i.e. 

$$
\widetilde {K} = U ^ {T} K U = \operatorname{diag} \left(K _ {1 1}, K _ {2 2}, \widetilde {K} _ {3 3}, K _ {4 4}, K _ {5 5}, \widetilde {K} _ {6 6}, \widetilde {K} _ {7 7}\right),\tag{3}
$$

in which U is the congruence transformation matrix. The corresponding transformed mass matrix Me is 

$$
\widetilde {M} = U ^ {T} M U = \left( \begin{array}{c c c c c c c} M _ {1 1} & & \widetilde {M} _ {1 3} & & & & \widetilde {M} _ {1 7} \\ & M _ {2 2} & \widetilde {M} _ {2 3} & & & & \widetilde {M} _ {2 7} \\ \widetilde {M} _ {3 1} & \widetilde {M} _ {3 2} & \widetilde {M} _ {3 3} & & & & \widetilde {M} _ {3 7} \\ & & & M _ {4 4} & & \widetilde {M} _ {4 6} & \widetilde {M} _ {4 7} \\ & & & & M _ {5 5} & \widetilde {M} _ {5 6} & \widetilde {M} _ {5 7} \\ & & & \widetilde {M} _ {6 4} & \widetilde {M} _ {6 5} & \widetilde {M} _ {6 6} & \widetilde {M} _ {6 7} \\ \widetilde {M} _ {7 1} & \widetilde {M} _ {7 2} & \widetilde {M} _ {7 3} & \widetilde {M} _ {7 4} & \widetilde {M} _ {7 5} & \widetilde {M} _ {7 6} & \widetilde {M} _ {7 7} \end{array} \right).\tag{4}
$$

In our implementation, the block Gaussian elimination procedure performs on substructures with the sequence of postorder traversal of the corresponding AMLS partition tree. The block Gaussian elimination on each substructure (treenode) s produces a factor, $U _ { s } ,$ in the form of 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/21fdf6ac8925b8a1bd678d1a2f96b5d79da0f1bf8ebcbac3598fdabaf5c28ff2.jpg)



Fig. 1. The two-level AMLS partition tree.


$$
U _ {s} = \left( \begin{array}{c c c c c} I & & & & \\ & \ddots & \Psi_ {s i} & & \Psi_ {s 7} \\ & & I & & \\ & & & \ddots & \\ & & & & I \end{array} \right), \quad \forall s \in \mathcal {S} _ {l = 2, 1}, \quad i = \mathcal {A} (s).\tag{5}
$$

Thus, we have 

$$
U = \prod_ {s = 1} ^ {6} U _ {s}.\tag{6}
$$

The non-diagonal blocks of $U _ { s } ,$ i.e. $\Psi _ { s i } ,$ are known as constraint modes [2]. Using a superscript [s] to denote that the elimination on substructures $\{ 1 , . . . , s \}$ has been completed and letting 

$$
\left(\widetilde {K} _ {i j} ^ {[ 0 ]}, \widetilde {M} _ {i j} ^ {[ 0 ]}\right) = (K _ {i j}, M _ {i j}),
$$

then the sub-matrices of products $( U _ { 1 } \cdot \cdot \cdot U _ { s } ) ^ { T } K ( U _ { 1 } \cdot \cdot \cdot U _ { s } ) , ( U _ { 1 } \cdot \cdot \cdot U _ { s } ) .$ $^ T M ( U _ { 1 } \cdots U _ { s } )$ and the expression of $\Psi _ { s i }$ can be presented as follows: 

(a) If s is a leaf, i.e. $s \in { \mathcal { S } } _ { l = 2 } , \ \forall i , j \in { \mathcal { A } } ( s )$ , then 

$$
\Psi_ {s i} = - K _ {s s} ^ {- 1} K _ {s i},\tag{7a}
$$

$$
\widetilde {K} _ {s i} ^ {[ s ]} = 0,\tag{7b}
$$

$$
\widetilde {M} _ {s i} ^ {[ s ]} = M _ {s i} + M _ {s s} \Psi_ {s i},\tag{7c}
$$

$$
\widetilde {K} _ {i j} ^ {[ s ]} = \widetilde {K} _ {i j} ^ {[ s - 1 ]} + \Psi_ {s i} ^ {T} K _ {s j},\tag{7d}
$$

$$
\widetilde {M} _ {i j} ^ {[ s ]} = \widetilde {M} _ {i j} ^ {[ s - 1 ]} + \Psi_ {s i} ^ {T} M _ {s j} + M _ {s i} ^ {T} \Psi_ {s j} + \Psi_ {s i} ^ {T} M _ {s s} \Psi_ {s j}.\tag{7e}
$$

(b) If s is not a leaf, i.e. $s \in { \mathcal { S } } _ { l = 1 } , \forall i , j \in { \mathcal { A } } ( s )$ , then 

$$
\Psi_ {s i} = - \left(\widetilde {K} _ {s s} ^ {[ s - 1 ]}\right) ^ {- 1} \widetilde {K} _ {s i} ^ {[ s - 1 ]},\tag{8a}
$$

$$
\widetilde {K} _ {s i} ^ {[ s ]} = 0,\tag{8b}
$$

$$
\widetilde {M} _ {s i} ^ {[ s ]} = \widetilde {M} _ {s i} ^ {[ s - 1 ]} + \widetilde {M} _ {s s} ^ {[ s - 1 ]} \Psi_ {s i},\tag{8c}
$$

$$
\widetilde {K} _ {i j} ^ {[ s ]} = \widetilde {K} _ {i j} ^ {[ s - 1 ]} + \Psi_ {s i} ^ {T} \widetilde {K} _ {s j} ^ {[ s - 1 ]},\tag{8d}
$$

$$
\widetilde {M} _ {i j} ^ {[ s ]} = \widetilde {M} _ {i j} ^ {[ s - 1 ]} + \Psi_ {s i} ^ {T} \widetilde {M} _ {s j} ^ {[ s - 1 ]} + \left(\widetilde {M} _ {s i} ^ {[ s - 1 ]}\right) ^ {T} \Psi_ {s j} + \Psi_ {s i} ^ {T} \widetilde {M} _ {s s} ^ {[ s - 1 ]} \Psi_ {s j}.\tag{8e}
$$

Obviously, the transformed $( \widetilde { K } _ { s s } , \widetilde { M } _ { s s } ) , ( s \in { \cal S } _ { l = 1 , 0 } )$ , in (3) and (4) is explicitly determined when the eliminations on substructures $\{ 1 , \ldots , ( s - 1 ) \}$ have been carried out, 

$$
(\widetilde {K} _ {s s}, \widetilde {M} _ {s s}) = \left(\widetilde {K} _ {s s} ^ {[ s - 1 ]}, \widetilde {M} _ {s s} ^ {[ s - 1 ]}\right), \quad \forall s \in \mathcal {S} _ {l = 1, 0}.\tag{9}
$$

However, the non-diagonal blocks, $\widetilde { M } _ { s i } , ( s \in S _ { l = 1 , 0 } , i \in \mathcal { A } ( s ) )$ , are determined by 

$$
\widetilde {M} _ {s i} = \widetilde {M} _ {s i} ^ {[ i ]}, \quad \forall i \in \mathcal {A} (i)\tag{10}
$$

in an implicit way that will be further described in the next subsection. 

## 2.3. Substructure condensation

Recall that AMLS is essentially a condensation method. It contains two steps: 

(a) extracting $n _ { m _ { s } }$ eigensolutions, $\begin{array} { r } { n _ { m _ { s } } \ll n _ { s } , } \end{array}$ , ofeach substructure s 

$$
K _ {s s} Z _ {s} = M _ {s s} Z _ {s} \Lambda_ {s}, \quad \forall s \in \mathcal {S} _ {l = 2}\tag{11a}
$$

$$
\widetilde {K} _ {s s} Z _ {s} = \widetilde {M} _ {s s} Z _ {s} \Lambda_ {s}, \quad \forall s \in \mathcal {S} _ {l = 1, 0}\tag{11b}
$$

lower than a given cutoff frequency $\omega _ { \mathrm { c u t o f f } } / c u t o f \mathrm { j }$ eigenvalue k<sub>cutoff</sub>: 

$$
\lambda_ {s, i} <   \omega_ {\text { cutoff }} ^ {2} \equiv \lambda_ {\text { cutoff }}, (i = 1, \dots , n _ {m _ {s}}).\tag{12}
$$

Here $Z _ { s } = [ z _ { s , 1 } , \ldots , z _ { s , n _ { m _ { s } } } ] \in \mathbb { R } ^ { n _ { s } \times n _ { m _ { s } } }$ and $\Lambda _ { s } = \mathrm { d i a g } [ \lambda _ { s , 1 } , \ldots , \lambda _ { s , n _ { m _ { s } } } ] .$ (b) using the global projection matrix $\boldsymbol { Z } \in \mathbb { R } ^ { n \times n _ { m } }$ $\begin{array} { r } { ( n _ { m } = \sum n _ { m _ { s } } , \ n _ { m } \ll n ) , } \end{array}$ 

$$
Z = \operatorname{diag} (Z _ {1}, Z _ {2}, Z _ {3}, Z _ {4}, Z _ {5}, Z _ {6}, Z _ {7}),\tag{13}
$$

to project $( \widetilde K , \widetilde M )$ onto a reduced subspace. The final reduced matrices $K _ { c } \in \mathbb { R } ^ { n _ { m } \times n _ { m } }$ and $M _ { c } \in \mathbb { R } ^ { n _ { m } \times n _ { m } }$ are 

$$
K _ {c} = Z ^ {T} \widetilde {K} Z = \operatorname{diag} \left(\Lambda_ {1}, \Lambda_ {2}, \Lambda_ {3}, \Lambda_ {4}, \Lambda_ {5}, \Lambda_ {6}, \Lambda_ {7}\right)
$$

and 

ð14Þ 

$$
M _ {c} = Z ^ {T} \widetilde {M} Z = \left( \begin{array}{c c c c c c c} I & & m _ {1 3} & & & & m _ {1 7} \\ & I & m _ {2 3} & & & & m _ {2 7} \\ m _ {3 1} & m _ {3 2} & I & & & & m _ {3 7} \\ & & & I & & m _ {4 6} & m _ {4 7} \\ & & & & I & m _ {5 6} & m _ {5 7} \\ & & & m _ {6 4} & m _ {6 5} & I & m _ {6 7} \\ m _ {7 1} & m _ {7 2} & m _ {7 3} & m _ {7 4} & m _ {7 5} & m _ {7 6} & I \end{array} \right),\tag{15}
$$

respectively, in which $m _ { j s } = m _ { s i } ^ { T } , \ m _ { j s } = Z _ { i } ^ { T } \widetilde { M } _ { j s } Z _ { s }$ 

In our implementation, the block Gaussian elimination procedure and these condensation operations are performed in an interleaving way: as soon as a sub-matrix pencil of $( K _ { s s } , M _ { s s } )$ or $( \widetilde { K } _ { s s } , \widetilde { M } _ { s s } )$ has been formed, the eigensolution of $( \Lambda _ { s } , Z _ { s } )$ and the calculation of $m _ { j s }$ will be started immediately. In addition, an intermediate matrix $\tilde { \mu } _ { j s }$ is used to obtain $m _ { j s } \mathrm { : }$ (a) if s is a leaf, then compute 

$$
\tilde {\mu} _ {s i} ^ {[ s ]} = Z _ {s} ^ {T} (M _ {s i} + M _ {s s} \Psi_ {s i}), \quad \forall i \in \mathcal {A} (s).\tag{16}
$$

(b) if s is not a leaf, then compute 

$$
\tilde {\mu} _ {s i} ^ {[ s ]} = Z _ {s} ^ {T} \left(\widetilde {M} _ {s i} ^ {[ s - 1 ]} + \widetilde {M} _ {s s} ^ {[ s - 1 ]} \Psi_ {s i}\right) \quad \forall i \in \mathcal {A} (s),\tag{17a}
$$

$$
m _ {j s} = \tilde {\mu} _ {j s} ^ {[ s - 1 ]} Z _ {s} \quad \forall j \in \mathcal {C} (s),\tag{17b}
$$

$$
\tilde {\mu} _ {j i} ^ {[ s ]} = \tilde {\mu} _ {j s} ^ {[ s - 1 ]} \Psi_ {s i} + \tilde {\mu} _ {j i} ^ {[ s - 1 ]} \quad \forall j \in \mathcal {C} (s), \quad \forall i \in \mathcal {A} (s).\tag{17c}
$$

Comparing (17a) and (17c) with (8c), we find that updating of $\tilde { M } _ { s i }$ is in fact implicitly performed in the form $\widetilde { \mu } _ { s i } = Z _ { s } ^ { T } \widetilde { M } _ { s i } ,$ instead of $\widetilde { M } _ { s i } .$ . That is because $\widetilde { M } _ { s i }$ is only used to compute $m _ { s i } = Z _ { s } ^ { T } \widetilde { M } _ { s i } Z _ { i }$ in the standard AMLS approach and apparently, since $n _ { m _ { s } }$  n much less operations and storage are required for $\tilde { \mu } _ { s i } \in \mathbb { R } ^ { n _ { m _ { s } } \times n _ { i } }$ than for $\widetilde { M } _ { s i } \in \mathbb { R } ^ { n _ { s } \times n _ { i } }$ 

## 2.4. Final condensed system

## Algorithm 1. The standard AMLS algorithm

Algorithm 1. The standard AMLS algorithm
8: do pre-condensation by (16)
9: mark s has been addressed and reset s to be its parent $\mathcal{P}(s)$ 10: else
11: partition s into several substructures $\mathcal{C}(s)$ 12: reset s to be an unaddressed substructure of $\mathcal{C}(s)$ 13: end if
14: else
15: if all substructures in $\mathcal{C}(s)$ have been addressed then,
16: solve constraint modes $\Psi_{si}$ by (8a), $\forall i \in \mathcal{A}(s)$ 17: solve the sub-eigenproblem (11b) to obtain $(\Lambda_s, Z_s)$ 18: do block Gaussian eliminations by (8d) and (8e)
19: do pre-condensation by (17a)
20: compute $m_{js}$ by (17b), $\forall j \in \mathcal{C}(s)$ 21: update $\tilde{\mu}_{ji}^{[s]}$ by (17c), $\forall j \in \mathcal{C}(s)$ , $\forall i \in \mathcal{A}(s)$ 22: assemble the wanted $(K_c, M_c)$ by $\Lambda_j$ , $\Lambda_s$ and $m_{js}$ , $\forall j \in \mathcal{C}(s)$ 23: mark s has been addressed and reset s to be its parent $\mathcal{P}(s)$ 24: else
25: reset s to be an unaddressed substructure of $\mathcal{C}(s)$ 26: end if
27: end if
28: end while
29: Solve the final system (18) to obtain eigenvalues $\Lambda_c$ 

When all partitions, block Gaussian eliminations and the condensation operations have been completed, we obtain the final condensed eigenproblem by AMLS, i.e. 

$$
K _ {c} X _ {c} = M _ {c} X _ {c} \Lambda_ {c}.\tag{18}
$$

Its solution will provide an approximation, $( \Lambda _ { c } , V ) ,$ , of the eigenpairs of the original problem (1), (K,X). Here the matrix 

$$
V = U Z X _ {c}\tag{19}
$$

contains AMLS approximate eigenvectors. 

We summarize the standard AMLS in Algorithm 1, in which the substructures are partitioned on the preorder traversal of the AMLS tree, while the block Gaussian eliminations and condensations are performed on the postorder traversal of the tree. Hence, during a standard AMLS procedure, each treenode will be visited twice except the leaves. The leaves are visited only once since no further partition is needed at all. So the standard AMLS algorithm without forming eigenvectors needs only one pre-postorder mixed traversal, i.e. {7, 3, 1, 2, 3, 6, 4, 5, 6, 7} for Fig. 1. 

## 2.5. AMLS eigenvector approximations

The standard AMLS excludes the calculation of eigenvector approximations. In this subsection we discuss two strategies for computing AMLS eigenvector approximations with different sequences of calculation in (19). 

The first one, a preliminary strategy, is to compute the projection matrix 

$$
T = U Z\tag{20}
$$

in advance, and then use T to calculate $V = T X _ { c }$ . Here T could also be explained as the global transformation matrix that transforms and projects (K,M) from the original space onto the AMLS condensed subspace to get $( K _ { c } , M _ { c } )$ . Kaplan gave the algorithm of forming T in detail [2]. After the computation of T, both eigenvectors and dynamical responses in the AMLS condensed subspace can be recovered back to the original space. 

However, assembling T is relatively expensive. According to [2] its cost ist similar to the one computing M . Therefore, a modified strategy is proposed to compute V. Note that $U \in \mathbb { R } ^ { n \times n }$ $Z \in \mathbb { R } ^ { n \times n _ { m } } , X _ { c } \in \mathbb { R } ^ { n _ { m } \times n _ { p } }$ and $n \gg n _ { m } > n _ { p } ,$ hence it is more efficient to compute 

$$
\widetilde {V} = Z X _ {c},\tag{21}
$$

and then to transform $\widetilde { V }$ back to the original space from right to left 

$$
V = U \widetilde {V} = U _ {1} \dots U _ {s} \widetilde {V}.\tag{22}
$$

To implement these two strategies, the calculated $Z _ { s } , \ \Psi _ { s i } ( \forall s \in S _ { l = 0 , 1 , 2 }$ ; 8i 2 AðsÞÞ and X are required to be stored during the execution of the standard AMLS. The performance of the modified strategy, which is also our default choice to form AMLS eigenvector approximations, will be shown by numerical experiments in Section 4.1. 

## 3. AMLS with Subspace Iterations

As mentioned in the Introduction, the accuracy of the computed eigenpairs $( \Lambda _ { c } , V )$ of AMLS depends on the cutoff frequency $\omega _ { \mathrm { c u t o f f } } .$ However, there is not a general principle to determine $\omega _ { \mathrm { c u t o f f } } .$ In addition, the standard AMLS is a one-shot projection method without any iterative improvement. So that the accuracy of AMLS is not satisfactory in the sense of structural dynamics, especially for the strain and stress computation. In order to extend AMLS to be wider applicable in structural dynamics and to make AMLS being less dependent on cutoff frequency, this paper employs subspace iterations to improve the approximate eigenpairs obtained by AMLS. 

## 3.1. Subspace iteration methods

Algorithm 2. The basic subspace iteration method (BSIM)
1: determine the dimension of subspace $n_{q} = \min(2n_{p}, n_{p} + 8)$ (23) $(n_{p} \text{ is the number of required eigenpairs}), \text{the error tolerance } \epsilon_{\text{tol}} \text{ and the maximum iteration step } n_{k}$ 2: do factorization $K = LDL^{T}$ 3: determine the initial iteration vectors $Q^{(0)}$ 4: for $k = 1, 2, \ldots, n_{k}$ , do
5: $\overline{P}^{(k)} = K^{-1}Q^{(k-1)}$ 6: $K_{c}^{(k)} = (\overline{P}^{(k)})^{T}Q^{(k-1)}$ 7: $\overline{Q}^{(k)} = M\overline{P}^{(k)}$ 8: $M_{c}^{(k)} = (\overline{P}^{(k)})^{T}\overline{Q}^{(k)}$ 9: $K_{c}^{(k)}X_{c}^{(k)} = M_{c}^{(k)}X_{c}^{(k)}\Lambda_{c}^{(k)}$ 10: $Q^{(k)} = \overline{Q}^{(k)}X_{c}^{(k)}$ 11: check convergence: if $\left|1 - \frac{\lambda_{c,i}^{(k)}}{(x_{c,i}^{(k)})^{T}x_{c,i}^{(k)}}\right|^{1/2} \leqslant \epsilon_{\text{tol}}$ (24)
holds for all required eigenpairs $(i = 1, \ldots, n_{p})$ , then stop iterations
12: end for 

Algorithm 3. An improved subspace iteration method (ISIM)

1: determine $n_{q}$ , the dimension of subspace
2: initialize iteration vector $Q^{(0)} \in R^{N \times n_{q}}$ 3: determine the number of maximum iteration step $n_{k}$ 4: determine the shift $\sigma$ 5: do factorization $K - \sigma M = LDL^{T}$ 6: do Sturm's sequence check
7: for $k = 1, 2, \ldots, n_{k}$ 8: M-orthogonlize $Q^{(k)}$ 9: compute $Q^{(k)} = (LDL^{T})^{-1} M Q^{(k-1)}$ 10: compute $K_{c}^{(k)} = (Q^{(k)})^{T} KQ^{(k)}$ , $M_{c}^{(k)} = (Q^{(k)})^{T} MQ^{(k)}$ 11: solve $K_{c}^{(k)} X_{c}^{(k)} = M_{c}^{(k)} X_{c}^{(k)} \Lambda_{c}^{(k)}$ 12: compute the eigenvectors $V^{(k)} = Q^{(k)} X_{c}^{(k)}$ 13: remove converged eigenvectors from $V^{(k)}$ , take the rest of vectors and some random vectors as the new iteration vectors $Q^{(k)}$ 14: M-orthogonlize $Q^{(k)}$ with respect to the already converged eigenvectors
15: end for
16: go back to Step 1, if not all wanted eigenpairs have been convergenced 

As one of classical eigensolution methods, subspace iteration method (SIM) has been developed and widely used for solving large generalized eigenvalue problems in structural dynamics. SIM was firstly proposed and so named by Bathe [19]. Basic Subspace Iteration Method (BSIM) (Algorithm 2) [22] is the original and simplest version, in which the number $n _ { q }$ of initial vectors is suggested to be $n _ { q } = \operatorname* { m i n } { \left( 2 n _ { p } , n _ { p } + 8 \right) }$ if $n _ { p }$ eigenpairs are desired. This suggestion was given in early publications [22,31] based on the experience that typically a small number of eigenpairs was wanted and the computers were much less powerful. Recently, Bathe revisited BSIM and optimized the number of initial iteration vectors based on a quite general model for the distribution of the eigenvalues [24], in which a new rule, 

$$
n _ {q} = \max (2 n _ {p}, n _ {p} + 8),\tag{25}
$$

instead of $n _ { q } = \operatorname* { m i n } \left( 2 n _ { p } , n _ { p } + 8 \right)$ , was derived to accelerate the rate of convergence of SIM. 

Besides optimizing the number of iteration vectors, many other acceleration techniques of SIM were also reported, such as restarts, aggressive shifts, Sturm’s sequence check and convergence criterion [20,25,32]. A typical improved subspace iteration method (ISIM) that integrates these techniques is given in Algorithm 3. 

This study focuses on the improvement of AMLS with subspace iterations. Hence, we consider mainly the basic iteration 

$$
Q ^ {(k + 1)} = K ^ {- 1} M Q ^ {(k)},\tag{26}
$$

excluding shifts and Sturm’s sequence check at present. Based on this declaration, two variants of AMLS hybrid Subspace Iteration Method, AMLS-SIMa and AMLS-SIMb, will be discussed. 

## 3.2. AMLS-SIMa: independent of AMLS

The first variant, AMLS-SIMa (Algorithm 4), is a derivative of BSIM, which takes advantage of AMLS only by choosing the AMLS approximate eigenvectors as initial subspace. In particular, it requires a triangular factorization of the original stiffness matrix $K = L D L ^ { T }$ . Obviously, the AMLS eigenvector approximations are the only inherited data from AMLS. So AMLS-SIMa is independent of AMLS during the subspace iteration phase. 

Nevertheless, AMLS is designed for computing hundreds or even thousands of eigenvalues, much larger than decades of eigenpairs solved by classical SIMs themselves. This situation may yield two special characteristics that differ from classical SIMs: 

(a) According to the rate of convergence of SIM for the ith eigenpairs proved by Bathe [33], 

$$
\lambda_ {i} / \lambda_ {n _ {q} + 1}, \quad i = 1, \dots , n _ {q},\tag{27}
$$

the convergence of eigenpairs for AMLS-SIM should be faster than classical SIMs at the lower end of spectrum, because AMLS provides more trial vectors for iterations. In addition, the initial subspace spanned by AMLS approximate eigenvectors is close to the exact subspace. Therefore, it is reasonable to improve eigenpairs of AMLS by SIM within a few iteration steps. 

(b) The cost of matrix–vector multiplications, inner products of vectors and linear equation solutions in a single iteration step will be much higher than classical SIMs. Thus, in the default options, we do not perform Rayleigh–Ritz analysis in each iteration step, i.e. neither form nor solve the reduced problem $\left( K _ { c } ^ { ( k ) } , M _ { c } ^ { ( k ) } \right)$ until all iterations are completed. This treatment is actually one kind of the multiple inverse iteration techniques [34–37], which is workable because the solution of $\left( K _ { c } ^ { ( k ) } , M _ { c } ^ { ( k ) } \right)$ is needed only for convergence criterion and has no influence on the rate of convergence. A disadvantage of performing Rayleigh–Ritz analysis after all iterations is that the approach may be computationally unstable if $n _ { k } > 3 .$ . Fortunately, since AMLS already produces a good initial approximation to the exact subspace, and the AMLS approximate eigenvectors are already M-orthogonlized, a very small number of iteration steps $n _ { k }$ can improve the AMLS approximate eigenpairs with acceptable accuracy before loss of stability. On the other hand, if the computational stability is required to be guaranteed, Rayleigh–Ritz analysis can also be activated in every iteration step in both AMLS-SIMa and AMLS-SIMb. 

Algorithm 4. AMLS-SIMa: The subspace iterations independent of AMLS

Require AMLS approximate eigenvectors V
1: initialize the iteration vector $Q^{(0)} = V$ 2: determine the maximum iteration number $n_{k}$ 3: do factorization $K = LDL^{T}$ 4: for $k = 1, 2, \ldots, n_{k}$ do
5: compute $R = M Q^{(k-1)}$ 6: solve $Q^{(k)}: K Q^{(k)} = R$ by the factorization of K
7: end for
8: reduce stiffness matrix by $K_{c}^{(n_{k})} = R^{T} Q^{(n_{k})}$ 9: reduce mass matrix by $M_{c}^{(n_{k})} = (Q^{(n_{k})})^{T} MQ^{(n_{k})}$ 10: solve $K_{c}^{(n_{k})} X_{c}^{(n_{k})} = M_{c}^{(n_{k})} X_{c}^{(n_{k})} \Lambda_{c}^{(n_{k})}$ to obtain improved eigenvalues $\Lambda_{c}^{(n_{k})}$ 11: compute improved eigenvectors $V^{(n_{k})} = Q^{(n_{k})} X_{c}^{(n_{k})}$ 

## 3.3. AMLS-SIMb: cooperated with AMLS

Except for the AMLS eigenvector approximations as initial trial vectors, AMLS-SIMa does not utilize any other AMLS data for subspace iterations. It requires to factorize the original stiffness matrix $K = L D L ^ { T }$ 

AMLS-SIMb, as shown in Algorithm 5, takes advantage of the transformed block diagonal stiffness matrix Ke, instead of the original K, for subspace iterations, i.e. 

$Q^{(k+1)} = U\widetilde{K}^{-1}U^{T}MQ^{(k)}.$ (28)

where $\widetilde{K} = U^{T}KU$ and U is the AMLS transformation matrix. (28) involves two transformations that are not contained in AMLS-SIMa, i.e. the forward transformation $\tilde{\nu} = U^{T}\nu$ and the backward transformation

w = U $\tilde{w}$ .

(Here v, $\tilde{v}$ , w and $\tilde{w}$ represent vectors.) Although these transformations require extra computation, our numerical results in Section 4 will show that AMLS-SIMb is more efficient than AMLS-SIMa with less execution time. 

Algorithm 5. AMLS-SIMb: The subspace iterations cooperated with AMLS

Require: transformed eigenvectors $\widetilde{V}$ , the transformed stiffness matrix $\widetilde{K}$ and the transformation matrix U from AMLS

1: initialize the iteration vector $\widetilde{Q}^{(0)} = \widetilde{V}$ 2: determine the maximum iteration number $n_{k}$ 3: for $k = 1, 2, \ldots, n_{k}$ do

4: transform backward $Q^{(k-1)} = U\widetilde{Q}^{(k-1)}$ 5: compute $R = M\widetilde{Q}^{(k-1)}$ 6: transform forward $\widetilde{R} = U^{T}R$ 7: solve $\widetilde{Q}^{(k)} : \widetilde{K}\widetilde{Q}^{(k)} = \widetilde{R}$ 8: end for

9: reduce stiffness matrix by $K_{c}^{(n_{k})} = \widetilde{R}^{T}\widetilde{Q}^{(n_{k})}$ 10: transform backward $Q^{(n_{k})} = U\widetilde{Q}^{(n_{k})}$ 11: reduce mass matrix by $M_{c}^{(n_{k})} = (Q^{(n_{k})})^{T}MQ^{(n_{k})}$ 12: solve $K_{c}^{(n_{k})}X_{c}^{(n_{k})} = M_{c}^{(n_{k})}X_{c}^{(n_{k})}\Lambda_{c}^{(n_{k})}$ to obtain improved eigenvalues $\Lambda_{c}^{(n_{k})}$ 13: compute improved eigenvectors $V^{(n_{k})} = Q^{(n_{k})}X_{c}^{(n_{k})}$ 

Here we emphasize two reasons for excluding the transformed matrix Me and the corresponding subspace iteration step 

$$
\widetilde {Q} ^ {(k + 1)} = \widetilde {K} ^ {- 1} \widetilde {M} \widetilde {Q} ^ {(k)}\tag{29}
$$

in AMLS-SIMb: 

(a) the original mass matrix M is sparse, whereas the matrix $\widetilde { M }$ contains dense blocks $\tilde { M _ { i j } }$ which require a lot of extra storage, and therefore M<sup>~</sup> is never assembled explicitly; (b) the transformation matrix U is needed to compute the final eigenvector approximations due to (19), and therefore has to be stored anyhow in the course of the AMLS method. 

## 3.4. Implementation of AMLS-SIMb

AMLS-SIMb utilizes the transformed stiffness matrix $\widetilde { K }$ and the transformation matrix U of AMLS. both of which are stored in dense blocks, $\widetilde { K } _ { s s }$ and $\Psi _ { s i } ,$ corresponding to the treenode index s $( i \in { \mathcal { A } } ( s ) )$ of the AMLS partition tree. Therefore, like the standard AMLS, the implementation of Algorithm 5 must be performed on several traversals of the AMLS tree. 

First of all, remember that the block Gauss eliminations, $\widetilde { K } = U ^ { T } K U ,$ , are performed on a postorder traversal of the AMLS tree, i.e. $\{ 1 , 2 , 3 , 4 , 5 , 6 , 7 \}$ for Fig. 1. Thus, the backward transformation $Q ^ { ( k - 1 ) } = { \cal U } \widetilde Q ^ { ( k - 1 ) }$ (Step 4 of Algorithm 5) could be executed by 

$$
(\widetilde {Q} ^ {(k - 1)}) ^ {[ s - 1 ]} = U _ {s} (\widetilde {Q} ^ {(k - 1)}) ^ {[ s ]}\tag{30}
$$

on an preoder traversal, i.e. {7,6, 5,4,3, 2,1} or {7,3,1, 2,6, 4,5} for Fig. 1. For convenience, we ignore the superscript [s] in (30) to simplify the notation of $( \stackrel { \sim } { Q } ^ { ( k - 1 ) } ) ^ { [ s - 1 ] }$ and $( \widetilde Q ^ { ( k - 1 ) } ) ^ { [ s ] }$ as $\dot { Q } ^ { ( k - 1 ) }$ and $\widetilde { Q } ^ { ( k - 1 ) } ,$ respectively. Recall the block structure of $U _ { s }$ in (5), then (30) can be rewritten as 

$$
Q _ {s} ^ {(k - 1)} = \widetilde {Q} _ {s} ^ {(k - 1)} + \sum_ {i} \Psi_ {s i} \widetilde {Q} _ {i} ^ {(k - 1)}, \quad Q _ {i} ^ {(k - 1)} = \widetilde {Q} _ {i} ^ {(k - 1)}, \quad \forall i \in \mathcal {A} (s),\tag{31}
$$

where $Q _ { i } ^ { ( k - 1 ) }$ denotes the ith block of $Q ^ { ( k - 1 ) } ,$ i.e. 

$$
Q ^ {(k - 1)} = \left(\left(Q _ {1} ^ {(k - 1)}\right) ^ {T}, \left(Q _ {2} ^ {(k - 1)}\right) ^ {T}, \dots , \left(Q _ {7} ^ {(k - 1)}\right) ^ {T}\right) ^ {T}.
$$

Similarly, the forward transformation $\widetilde { R } = U ^ { T } R$ (Step 6 of Algorithm 5) can be carried out by 

$$
\widetilde {R} _ {s} = R _ {s}, \quad \widetilde {R} _ {i} = R _ {i} + \Psi_ {s i} ^ {T} R _ {s}, \quad \forall i \in \mathcal {A} (s),\tag{32}
$$

on a postorder traversal of the AMLS tree. The traversal implementation of the iteration body of AMLS-SIMb (Step 4 to Step 7 of Algorithm 5) are summarized in Algorithm $6 ,$ as well as the computation of $K _ { c }$ (Step 9 of Algorithm 5). Here $K _ { c }$ is formed on the last iteration step in order to take advantage of the block diagonal property of ${ \widetilde { K } } .$ Then, Step 10 and 11 of Algorithm 5 can be executed by an extra traversal of the AMLS partition tree. 

Algorithm 6. The traversal implementation of the iteration body for AMLS-SIMb
1: set current addressing substructure s to be the root of AMLS tree.
2: initialize R = 0 and $K_{c} = 0$ 3: while not all substructures have been addressed,
4: if This is the first time to visit s, then
5: transform backward $Q^{(k-1)} = U_{s}\widetilde{Q}^{(k-1)}$ by (31)
6: compute the contributions of s for $R = MQ^{(k-1)}$ by $R_{s} = R_{s} + M_{ss}Q_{s}^{(k-1)} + \sum_{i} M_{si}Q_{i}^{(k-1)}, \quad \forall i \in \mathcal{A}(s),$ (33a) $R_{i} = R_{i} + M_{si}^{T}Q_{s}^{(k-1)}, \quad \forall i \in \mathcal{A}(s).$ (33b)
7: end if
8: if s is a leaf or all substructures of $C(s)$ have been addressed,
9: transform forward $\widetilde{R} = U_{s}^{T}R$ by (32)
10: solve $\widetilde{Q}_{s}^{(k)} : \widetilde{K}_{ss}\widetilde{Q}_{s}^{(k)} = R_{s}$ 11: if $k = n_{k}$ , then
12: compute $K_{c} = K_{c} + \widetilde{R}_{s}\widetilde{Q}_{s}^{(k)}$ 13: end if
14: mark s has been addressed and reset s to be its parent $\mathcal{P}(s)$ 15: else
16: reset s to be an unaddressed substructure of $C(s)$ 17: end if
18: end while 

## 4. Numerical experiments

In order to evaluate the performance of AMLS-SIMa and AMLS-SIMb, four engineering problems with different scales are selected for numerical experiments. Table 1 gives a brief description of these problems. 

The two variants, AMLS-SIMa and AMLS-SIMb, were coded based on the serial version of Elssel’s AMLS implementation [11]. 


Table 1



Four test problems and their descriptions (n: the number of degrees of freedom; $n _ { z } ( K ) { : }$ the number of non-zeros of the stiffness matrix; $n _ { z } ( M ) \colon$ the number of non-zeros of the mass matrix)


<table><tr><td>Problems</td><td>Description</td><td>n</td><td><eq>n_z(K)</eq></td><td><eq>n_z(M)</eq></td></tr><tr><td>bcsstk24</td><td>Sports arena [39]</td><td>3562</td><td>159,910</td><td>3562</td></tr><tr><td>wtaoc01k</td><td>Blade of a 50 KW wind turbine [40]</td><td>23,328</td><td>2,220,068</td><td>814,694</td></tr><tr><td>wtnh01k</td><td>Blade of a 1.5 MW wind turbine [41]</td><td>117,990</td><td>11,243,248</td><td>5,590,256</td></tr><tr><td>bs6_k</td><td>Structure of connected beams [11]</td><td>517,161</td><td>20,727,171</td><td>6,909,057</td></tr></table>


Table 2



Required storage for useful matrices of four problems.


<table><tr><td>Problem</td><td>K</td><td>M</td><td><eq>\widetilde{K} = {U}^{T}{KU}</eq></td><td>U</td></tr><tr><td>bcsstk24</td><td>1.0 MB</td><td>0.3 MB</td><td>1.7 MB</td><td>3.5 MB</td></tr><tr><td>wtaoc01k</td><td>12.8 MB</td><td>4.8 MB</td><td>23.3 MB</td><td>57.9 MB</td></tr><tr><td>wtnh01k</td><td>64.9 MB</td><td>32.5 MB</td><td>126.4 MB</td><td>374.2 MB</td></tr><tr><td>bs6_k</td><td>121.1 MB</td><td>42.0 MB</td><td>393.3 MB</td><td>1454.8 MB</td></tr></table>


Table 3



Execution time for computing AMLS eigenvector approximations.


<table><tr><td>Problem</td><td><eq>\lambda_{\text{cutoff}}</eq> (rad<eq>^2</eq>/s<eq>^2</eq>)</td><td><eq>n_q</eq></td><td><eq>t_{\text{AMLS}}</eq> (s)</td><td><eq>t_{\text{eigvec}}</eq> (s)<eq>^a</eq></td><td><eq>t_{\text{eigvec}}</eq> (s)<eq>^b</eq></td></tr><tr><td>bcsstk24</td><td><eq>1 \times 10^3</eq></td><td>201</td><td>0.78</td><td>0.21</td><td>0.05</td></tr><tr><td>wtaoc01k</td><td><eq>1 \times 10^7</eq></td><td>144</td><td>11.66</td><td>4.63</td><td>0.46</td></tr><tr><td>wtnh01k</td><td><eq>1 \times 10^6</eq></td><td>350</td><td>100.21</td><td>196.78</td><td>7.60</td></tr><tr><td>bs6_k</td><td><eq>3 \times 10^{10}</eq></td><td>300</td><td>308.39</td><td>Out of memory</td><td>33.08</td></tr></table>


<sup>a</sup> Execution time by the preliminary strategy. 



<sup>b</sup> Execution time by the modified strategy. 


It employs the METIS [29] and the Intel Math Kernel Library v10.3 optimized LAPACK to achieve high performance of computation. Table 2 lists the matrices and their disk storages required for AMLS and subspace iterations. 

## 4.1. Computation of AMLS eigenvector approximations

Sections 4.1,4.2,4.3,4.4 report on numerical experiments performed on a 64-bit Linux platform with an Intel Pentium D CPU (3.46 GHz, 2 cores) and 7.7 GB memory. 

First of all, we list the execution time required for the standard AMLS and the computation of AMLS eigenvector approximations of the four problems in Table 3. Both of the two strategies mentioned in Section 2.5 are implemented. Although these two strategies calculate exactly the same eigenvector approximations according to basic linear algebra knowledge, Table 3 shows that the modified strategy presented in (21) and (22) performs more efficiently along with the scale of problems increases. So we set the modified strategy as the default option in our code to compute the AMLS eigenvector approximations for the further subspace iterations. 

## 4.2. Modal error of eigenpairs

Before discussing any improvement of subspace iterations, let us review the error of eigenpairs that are produced by the standard AMLS algorithm. The relative error between eigenvalues of AMLS and eigenvalues of some classical algorithms or codes, e.g. eigs of Matlab, has been reported in [3], etc. Generally speaking, if a proper cutoff frequency $\omega _ { \mathrm { c u t o f f } }$ is selected, the relative error is expected to be in the range of $1 0 ^ { - 4 }$ to $1 0 ^ { - 2 }$ for all computed eigenvalues. 


(a): bcsstk24


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/434343b61ded1db283b2d50b90b239bbf09a584ab7317f7b7d635ff52542f973.jpg)



(b): wtaoc01k


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/cc90eb918fff6e6dc2bd93771c98aa7d534df5941026df68bbb8b88146271d7e.jpg)



(c): wtnh01k


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/fb13dfe6c46ffa1242ba0dcd54e24bf989ed2e612be166d29166521d75444956.jpg)



(d): bs6_k


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/d1cc12ba819a0d68c5f18e335c70fca4c25a4efdfed4f6b7844558c6e974407a.jpg)



Fig. 2. Modal error of AMLS eigenpair approximations for four problems (Unit of $\lambda _ { \mathrm { c u t o f f } } \colon \mathrm { r a d } ^ { 2 } / s ^ { 2 } ) .$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/5dc32f23ec3bd5caefb6826eb1e39d3d9c52913e39937c9b2a941853ed748c32.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/20713c3b6ebbc6ba9efadd6931a9b12483df4a6078bc676cb1fca9a9cf780341.jpg)



(c): wtnh01k $( \lambda _ { _ \mathrm { C u t o f f } } = 1 { \ : } 6 6 )$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/1c06748a17bf2971a8fa823678518eb0e5dbd5221aa734d0091db8cdd5e574b7.jpg)



(d): bs6_k $( \lambda _ { \mathrm { c u t o f f } } = 3 \mathsf { e } 1 0 )$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/06d1dc298f87973a3e17f23495943dca530ea70f6aaac5dd2fbf58c78efdd374.jpg)



Fig. 3. Modal error of improved eigenpairs by AMLS-SIM for four problems (Unit of $\lambda _ { \mathrm { c u t o f f } } \mathrm { : }$ ra $^ 2 / s ^ { 2 } ) .$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/f492f60603b84f8895f1397642a6015b070dc0e045b8a10db461c7df50b28624.jpg)


$$
\lambda_ {\mathrm{cutoff}} = 1. 0 \times 1 0 ^ {4} (\mathrm{rad/s}) ^ {2}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/cc4d9949dbd655d5861e944d0853ed2fcf33c50da0301457e4b440749df24f4d.jpg)



Fig. 4. Relative error of eigenvalues (Problem: bcsstk24).


The error of AMLS eigenvector approximations is rarely studied in the literature. In this paper, one kind of residual errors suggested by Bathe [22], or called as modal error, is used to evaluate the error of each eigenpair $\scriptstyle ( \lambda , x )$ of the eigenvalue problem (1), i.e. 

$$
\epsilon = \frac {\| K x - \lambda M x \| _ {2}}{\| \lambda M x \| _ {2}}.\tag{34}
$$

Fig. 2 provides the modal error of AMLS approximate eigenpairs. In Fig. 2, increasing the $\lambda _ { \mathrm { c u t o f f } }$ 10 times, the modal error is found to be only slightly reduced. It implies that the accuracy of AMLS approximate eigenvectors is not sensitive to the cutoff frequency in terms of the modal error. In addition, not a single eigenpair has a modal error less than $1 0 ^ { 0 } .$ But in structural dynamics, the modal error is required to be as low as $1 0 ^ { - 3 } .$ . Otherwise, no accurate strain or stress can be derived. For many classical eigensolution methods, e.g. the block Lanczos algorithm [5], the modal error can achieve $1 0 ^ { - 5 }$ or lower. So that there should exist a big potential to improve AMLS, especially for eigenvector approximations. 

## 4.3. Performance of AMLS with subspace iterations

For the variants of AMLS-SIM, the standard AMLS could be seen as a pre-conditioner that provides a good initial subspace. Then in the further iteration refinements, both AMLS-SIMa and AMLS-SIMb calculate improved eigenpairs with the same precision. Fig. 3 demonstrates that even one or two iteration steps can significantly improve the accuracy of eigenpairs, especially at the lower end of the spectrum. This phenomenon can be explained by $( 2 7 ) ,$ the rate of convergence of SIM. 

Besides eigenvectors, AMLS approximate eigenvalues are also enhanced considerably as illustrated in Fig. 4. Here we use instead of the exact eigenvalues of the respective problems as reference values the eigenvalue approximations obtained with the Matlab function eigs. Fig. 4 shows that AMLS-SIM with a higher cutoff frequency always delivers more accurate eigenvalues. But it should be mentioned that AMLS-SIM relaxes the selection of cutoff frequency by extra subspace iteration steps. 


Execution time comparison of AMLS-SIMa and AMLS-SIMb with a single subspace iteration step.


<table><tr><td>Problem</td><td><eq>n_q</eq></td><td><eq>t_{\text{SIMa}}^{\text{Factorization}}</eq>(s)</td><td><eq>t_{\text{SIMa}}</eq>(s)</td><td><eq>t_{\text{SIMb}}</eq>(s)</td><td>Saved <eq>time^a</eq>(%)</td></tr><tr><td>bcsstk24</td><td>201</td><td>0.56</td><td>0.75</td><td>0.53</td><td>29</td></tr><tr><td>wtaoc01k</td><td>144</td><td>1.28</td><td>7.70</td><td>4.35</td><td>44</td></tr><tr><td>wtnh01k</td><td>350</td><td>7.99</td><td>167.03</td><td>69.55</td><td>58</td></tr><tr><td>bs6_k</td><td>300</td><td>27.17</td><td>385.71</td><td>210.07</td><td>46</td></tr></table>


<sup>a</sup> This column estimates how much execution time could be saved by AMLS-SIMb. The percentage is obtained by $( t _ { \mathrm { S I M a } } - t _ { \mathrm { S I M b } } ) / t _ { \mathrm { S I M a } }$ 



Execution time comparison of AMLS-SIMa, AMLS-SIMb, BSIM without shifts (Algo rithm 2) and ISIM with shifts (Algorithm 3) for Problem bcsstk24 (n = 3562) with a converged tolerance $\epsilon _ { t o l } = 1 0 ^ { - 3 }$


<table><tr><td rowspan="2">Methods</td><td colspan="2"><eq>n_p = 50</eq></td><td colspan="2"><eq>n_p = 100</eq></td><td colspan="2"><eq>n_p = 200</eq></td></tr><tr><td>t</td><td><eq>n_k(n_{LDL^T})</eq></td><td>t</td><td><eq>n_k(n_{LDL^T})</eq></td><td>t</td><td><eq>n_k(n_{LDL^T})</eq></td></tr><tr><td>BSIMaa</td><td>5.41 s</td><td>38(1)</td><td>22.20 s</td><td>79(1)</td><td>88.08 s</td><td>127(1)</td></tr><tr><td>BSIMbb</td><td>4.67 s</td><td>14(1)</td><td>13.01 s.</td><td>16(1)</td><td>27.53 s</td><td>12(1)</td></tr><tr><td>ISIM</td><td>6.31 s</td><td>61(17)</td><td>13.72 s</td><td>114(32)</td><td>30.54 s</td><td>201(56)</td></tr><tr><td>AMLS-SIMac</td><td>3.64 s</td><td>3(1)</td><td>4.79 s</td><td>4(1)</td><td>11.33 s</td><td>10(1)</td></tr><tr><td>AMLS-SIMbc</td><td>3.18 s</td><td>3(1)</td><td>3.91 s</td><td>4(1)</td><td>9.42 s</td><td>10(1)</td></tr></table>


<sup>a</sup> The number of iteration vectors in BSIMa is determined by $n _ { q } = \operatorname* { m i n } { ( 2 n _ { p } , n _ { p } + 8 ) } .$ <sup>b</sup> The number of iteration vectors in BSIMb is determined by $n _ { q } =$ max $( 2 n _ { p } , n _ { p } + 8 )$ 



<sup>c</sup> Including the execution time of AMLS (t ) and the computation of eigenvector approximations $( t _ { \mathrm { e i g v e c } } )$ 


## 4.4. Comparison of AMLS-SIMa and AMLS-SIMb

The execution time of AMLS-SIMa $( t _ { \mathrm { S I M a } } )$ and AMLS-SIMb $\left( t _ { \mathsf { S I M b } } \right)$ with only one iteration step are listed in Table 4. Here the factorization (Step 3 of Algorithm 4) and solution (Step 6 of Algorithm 4) of the original stiffness matrix in AMLS-SIMa are implemented by the Intel Math Kernel Library v10.3 optimized PARDISO package. Since the factorization is performed only once, its execution $t _ { S I M a }$ 

Table 4 shows that AMLS-SIMb significantly reduces the execution time of subspace iterations compared to AMLS-SIMa. The percentage of reduced time is estimated by $( t _ { \mathrm { S I M a } } - t _ { \mathrm { S I M b } } ) / t _ { \mathrm { S I M a } }$ in the last column of Table 4. Recall that degrees of freedom of these problems increase from bcsstk24 to bs6_k (Table 1). So in general, AMLS-SIMb saves more execution time compared to AMLS-SIMa for larger problems. In Table 4, the only exception of this observation appears between wtnh01k and bs6_k. This could be explained by the sparsity of finite element matrices. In fact, there are 95.3 non-zeros per row of K and 47.4 non-zeros per row of M for Problem wtnh01k on average. $( 9 5 . 3 = n _ { z } ( K ) / n = 1 1 , 2 4 3 , 2 4 8 / \uparrow$ 117,990, $4 7 . 4 = n _ { z } ( M ) / n = 5 , 5 9 0 , 2 5 6 / 1 1 7 , 9 9 0 .$ . Please see Table 1.) But for Problem bs6_k, these two numbers are 40.0 and 13.4, respectively. It implies that AMLS-SIMb performs even better than AMLS-SIMa for those structures with denser matrices. 

## 4.5. Comparison with current subspace iteration methods

The computations reported on in this subsection were performed on a Linux platform with an Intel Core2 Duo CPU (T7100@1.79 GHz, 2 cores) and 2.5 GB memory. 

In order to compare the two AMLS-SIM variants with BSIM and ISIM, we slightly modified AMLS-SIMa and AMLS-SIMb by activating the Rayleigh–Ritz analysis of $K _ { c } ^ { ( k ) } X _ { c } ^ { ( k ) } = M _ { c } ^ { ( k ) } X _ { c } ^ { ( k ) } \Lambda _ { c } ^ { ( k ) }$ and the convergence criterion (24) in each iteration step. For BSIM without shifts (Algorithm 2), the former rule $n _ { q } = \operatorname* { m i n } ( 2 n _ { p } , n _ { p } + 8 )$ [22] and the new one $n _ { q } = \mathrm { m a x } ( 2 n _ { p } , n _ { p } + 8 ) [ 2 4 ]$ are both adopted. To achieve high performance, the Intel Math Kernel Library v10.3 optimized PARDISO serves as the sparse solver in the implementation of BSIM. Another fast direct sparse solver developed by ourselves with a cell-sparse storage scheme and the loop-unrolling technique [38] is integrated in the program of ISIM with shifts (Algorithm 3) [32,25]. 

Table 5 lists the execution time, numbers of iteration steps (n ) and numbers of $L D L ^ { T }$ factorizations $( n _ { L D L ^ { T } } )$ required by each subspace iteration methods for Problem bcsstk24 with $\epsilon _ { t o l } = 1 0 ^ { - 3 }$ Table 5 shows that: (a) BSIMa with $n _ { q } =$ min $( 2 n _ { p } , n _ { p } + 8 )$ costs the most execution time; (b) BSIMb with $n _ { q } = \mathrm { m a x } \left( 2 n _ { p } , n _ { p } + 8 \right)$ significantly accelerates the rate of convergence in comparison with BSI-Ma since about twice as much initial vectors are employed in BSIMb; (c) ISIM is more efficient than BSIMa not only because of shifts, but also due to the much smaller subspace that always removes converged eigenvectors and adds new random vectors; (d) the proposed AMLS-SIMa and AMLS-SIMb are faster than BSIMa, BSIMb and ISIM, especially if hundreds of eigenpairs are required. 

Fig. 5 demonstrates that the accuracy of AMLS-SIM variants is almost the same with BSIM at the lower end of spectrum; while from mode 75 to 200, the error of AMLS-SIM variants rises up (but still under the tolerance $\epsilon _ { t o l } = 1 0 ^ { - 3 } ) .$ . This is the price that one has to pay for the cost reduction. 


(a): Relative errors of eigenvalues


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/6cf95abb9117fa36c1bb48cfb57846d33d9309398469caa238ddfb82d848cb59.jpg)


## (b): Modal errors of eigenpairs

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/60faf471-b454-49e4-8e81-60b9c1de0a10/17028ee339526c63577fa3d5e9697d84733e11c0cd2afe22ceb43caf4ddca26c.jpg)



Fig. 5. Relative error of eigenvalues and modal error by BSIM, ISIM and AMLS-SIM with $\epsilon _ { t o l } = 1 0 ^ { - 3 }$ (Problem: bcsstk24).


## 5. Conclusions

In this paper, the standard AMLS algorithm for linear eigenvalue problems is reviewed, including discussions of two eigenvector computational strategies. Based on the modified eigenvector computation strategy, two variants of the AMLS hybrid Subspace Iteration Method (AMLS-SIMa and AMLS-SIMb) are presented to improve the eigenpair approximations of the standard AMLS. AMLS-SIMa is independent of AMLS, while AMLS-SIMb takes advantage of the transformed block diagonal stiffness matrix to reduce the cost of subspace iterations. 

Our numerical experiments with different scales indicate that even one or two subspace iterations can significantly reduce the modal error of the AMLS eigenpair approximations at the lower end of spectrum. The comparison of AMLS-SIMa and AMLS-SIMb demonstrate that AMLS-SIMb is faster than AMLS-SIMa in the sense of execution time, especially for larger and more complicated problems. Moreover, AMLS-SIMa and AMLS-SIMb could also be seen as improvements for subspace iteration methods because AMLS can serve as a pre-conditioner to accelerate the rate of convergence. 

Finally, we conclude that AMLS-SIMb not only successfully improves AMLS eigenpair approximations, but also makes AMLS being less dependent on cutoff frequency by subspace iterations. These results may extend AMLS into a wider application area in structural dynamics. 

In future, it should be possible to develop a more robust AMLS hybrid Subspace Iteration Method that integrates Sturm’s sequence check and shifts on the basis of AMLS-SIMb. Furthermore, since AMLS-SIMb is implemented by traversals on the partition tree similarly with the standard AMLS algorithm, we believe that AMLS-SIMb could also be efficiently parallelized for larger problems. 

## Acknowledgments

This work was done in Hamburg, Germany when the first author was visiting the Institute of Numerical Simulation, Hamburg University of Technology. The first author would like to thank Dr. Kolja Elssel, Dr. Markus Stammberger and Dr. Duy Nam Le for their constructive directions of coding AMLS and many other useful discussions, as well as the host Institute of Numerical Simulation. 

## References



[1] Bennighof JK, Kim CK. An adaptive multi-level substructuring method for efficient modeling of complex structures In: Proceedings of the AIAA 33rd SDM conference, Dallas, Texas, USA, 1992, p. 1631–1639 





[2] Kaplan MF, Implementation of automated multilevel substructuring for frequency response analysis of structures, Ph.D. thesis, The University of Texas at Austin, 2001. 





[3] BennighofJK, Lehoucq RB. An automated multilevel substructuring method for eigenspace computation in linear elastodynamics. SIAM J Sci Comput 2004:25:2084-106. http://dx.doi.org/10.1137/S1064827502400650 





[4] Lanczos C. An iteration method for the solution of the eigenvalue problem of linear differential and integral operators. J Res Nat Bureau Stand 1950:45:222-82 





[5] Grimes RG, Lewis JG, Simon HD. A shifted block lanczos algorithm for solving sparse symmetric generalized eigenproblems. SIAM J Matrix Anal Appl 1994;15(1):228–72. http://dx.doi.org/10.1137/S0895479888151111. 





[6] Kropp A, Heiserer D. Efficient broadband vibro-acoustic analysis of passenger car bodies using an fe-based component mode synthesis approach. J Comput Acoust 2003:11:139-57. http://dx.doi.org/10.1142/S0218396X03001870 





[7] Le DN. Extending deterministic vibration analysis of ships into the medium frequency range, Ph.D. thesis, Hamburg University of Technology, 2009. 





[8] Yang C, Gao W, Bai Z, Li XS, Lee LQ, Husbands P, et al. Algebraic sub-structuring for electromagnetic applications. Lect Notes Comput Sci 2006:3732:364–73 bttp://dx doi org/10.1007/11558958,43 





[9] Rachowicz W, Zdunek A. Automated multi-level substructuring (amls) for electromagnetics. Comput Methods Appl Mech Eng 2009;198(13- 14):1224–34. http://dx.doi.org/10.1016/j.cma.2008.10.011. 





[10] Elssel K, Voss H. Reducing huge gyroscopic eigenproblems by automated multi-level substructuring. Arch Appl Mech 2006;76:171–9. 





[11] Elssel K. Automated multilevel substructuring for nonlinear eigenvalue problems, Ph.D. thesis, Hamburg, University of Technology, 2006. 





[12] Elssel K, Voss H. Reducing sparse nonlinear eigenproblems by automated multi-level substructuring. Adv Eng Softw 2008;39:828–38. http://dx.doi.org/ 10.1016/j.advengsoft.2007.05.004. 





[13] Stammberger M, Voss H. Automated multi-level sub-structuring for fluid–solid interaction problems. Numer Linear Algebra Appl 2011;18(3):411–27. http:// dx.doi.org/10.1002/nla.734. 





[14] Bekas C, Saad Y. Computation of smallest eigenvalues using spectral schur complements. SIAM J Sci Comput 2005;27:458–81. http://dx.doi.org/10.1137/ 040603528. 





[15] Yang C, Gao W, Bai Z, Li XS, Lee LQ, Husbands P, et al. An algebraic substructuring method for large-scale eigenvalue calculation. SIAM J Sci Comput 2005;27:873–92. http://dx.doi.org/10.1137/040613767. 





[16] Elssel K, Voss H. An a priori bound for automated multilevel substructuring. SIAM J Matrix Anal Appl 2006;28:386–97. http://dx.doi.org/10.1137/ 040616097. 





[17] Ko JH, Jung SN, Byun D, Bai Z. An algebraic substructuring using multiple shifts for eigenvalue computations. J Mech Sci Technol 2008;22:440–9. http:// dx.doi.org/10.1007/s12206-007-1046-1. 





[18] Liao BS, Bai Z, Gao W. The important modes of subsystems: a momentmatching approach. Int J Numer Methods Eng 2007;70(13):1581–97. http:// dx.doi.org/10.1002/nme.1940 





[19] Bathe KJ, Wilson EL. Large eigenvalue problems in dynamic analysis. J Eng Mech Div 1972;98(6):1471–85. 





[20] Bathe KJ, Ramaswamy S. An accelerated subspace iteration method. Comput Methods Appl Mech Eng 1980;23(3):313–31. http://dx.doi.org/10.1016/0045- 7825(80)90012-2 





[21] Wilson EL, Itoh T. An eigensolution strategy for large systems. Comput Struct 1983;16(1-4):259–65. http://dx.doi.org/10.1016/0045-7949(83)90166-9. 





[22] Bathe KJ. Finite element procedure. Prentice-Hall, Inc.; 1996. Ch. The Subspace Iteration Method, pp. 954–978. 





[23] Sedeh RS. Bathe M. Bathe KI. The subspace iteration method in protein normal mode analysis. J Comput Chem 2010;31(1):66–74. http://dx.doi.org/10.1002/ jcc.21250 





[24] Bathe KJ. The subspace iteration method – Revisited, Comput Struct. http:// dx.doi.org/10.1016/j.compstruc.2012.06.002, in press. 





[25] Zhao QC, Chen P, Peng WB, Gong YC, Yuan MW. Accelerated subspace iteration with aggressive shift. Comput Struct 2007;85(19–20):1562–78. http:// dx.doi.org/10.1016/j.compstruc.2006.11.033. 





[26] Hurty WC. Vibrations of structural systems by component-mode synthesis. J Eng Mech Div ASCE 1960;86:51–69. 





[27] Craig RR, Bampton MCC. Coupling of substructures for dynamic analysis. AIAA J 1968;6:1313–9. 





[28] Hendrickson B, Leland R. The chaco user’s guide: Version 2.0, Tech. rep., Sandia National Labs, Tech. Report SAND94-2692, Albuquerque, NM, June 1995. 





[29] Karypis G, Kumar V. Metis – a software package for partitioning unstructured graphs, partitioning meshes, and computing fill-reducing orderings of sparse matrices. version 4.0, Tech. rep., University of Minnesota, Minneapolis, 1998. 





[30] Kruse RL, Ryba AJ. Data structures and program design in C++. Pearson Education: 1999 





[31] Bathe KJ, Wilson EL. Solution methods for eigenvalue problems in structural mechanics. Int J Numer Methods Eng 1973;6:213–26. 





[32] Gong Y, Zhou H, Chen P, Yuan M. Comparison of subspace iteration, iterative Ritz vector method and iterative Lanczos method. J Vibr Eng 2005;18(2):227–32 [in Chinese]. 





[33] Bathe KJ. Formulations and computational algorithms in finite element analysis. Cambridge, Massachusetts, UK: MIT Press; 1977. Ch. Convergence of Subspace Iterations, pp. 575–598. 





[34] Lam YC, Bertolini AF. Acceleration of the subspace iteration method by selective repeated inverse iteration. Finite Elem Anal Des 1994:18:309–17. 





[35] Bertolini AF, Lam YC. Accelerated reduction of subspace upper bound by multiple inverse iteration. Comput Syst Eng 1995;6(1):67–72. 





[36] Qian Y, Dhatt G. An accelerated subspace iteration method for generalized eigenproblems. Comput Struct 1995:54:1127-34. 





[37] Wang XL, Zhou J. An accelerated subspace iteration method for generalized eigenproblems. Comput Struct 1999;71:293–301. 





[38] Chen P, Zheng D, Sun S, Yuan M. High performance sparse static solver in finite element analyses with loop-unrolling, Ady Eng Softw 2003:34:203-15. 





[39] Duff IS, Grimes R, Lewis J. The user’s guide for the harwell-boeing sparse matrix collection (release i). Tech. rep., CERFACS. Technical Report TR/PA/92/ 86, Lyon, France, 1992. 





[40] McKittrick LR, Cairns DS, Mandell J, Combs DC, Rabern D, Van Luchene RD. Analysis of a composite blade design for the aoc 15/50 wind turbine using a finite element model, Tech. rep., Sandia National Laboratories, SAND2001- 1441, Albuquerque, NM, USA, 2001. 





[41] Aerodynamics design of nh1500 wind turbine blade and its performance verifications, Tech. rep., Nanjing University of Aeronautics and Astronauticse, Nanjing, Jiangsu, China, 2010. 

