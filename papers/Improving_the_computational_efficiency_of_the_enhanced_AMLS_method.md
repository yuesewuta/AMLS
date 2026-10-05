# Improving the computational efficiency of the enhanced AMLS method

Cheolgyu Hyun <sup>a</sup>, Seung-Hwan Boo <sup>b</sup>, Phill-Seung Lee <sup>a,</sup>⇑ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/f13b0df14763720c099975c5b946037aa39cc2ac3ba9bfbf8410397470d9764a.jpg)


<sup>a</sup> Department of Mechanical Engineering, Korea Advanced Institute of Science and Technology, 291 Daehak-ro, Yuseong-gu, Daejeon 34141, Republic of Korea <sup>b</sup> Division of Naval Architecture and Ocean Systems Engineering, Korea Maritime and Ocean University, 727 Taejong-ro, Yeongdo-gu, Busan 49112, Republic of Korea 

## a r t i c l e i n f o

Article history: Received 13 July 2019 Accepted 10 November 2019 

Keywords: Structural dynamics Finite element method Model reduction Component mode synthesis AMLS method Eigenvalue problem 

## a b s t r a c t

The objective of this study is to improve the computational efficiency of the enhanced automated multilevel substructuring (EAMLS) method (Kim et al., 2015) for dealing with large finite element models in structural dynamics. In the EAMLS method, the compensation procedure of the residual mode effect is a bottleneck resulting in a large computational cost. The original EAMLS method indeed cannot be used for finite element models with more than one million degrees of freedom in personal computers. An improvement in the computational efficiency is achieved by an interface subspace reduction and a new compensation procedure derived by residual flexibility matrices for only bottom-level substructures. Through this approach with submatrix level computations, the computation time and memory requirements are significantly reduced. Several numerical examples are presented to show the effectiveness of the proposed method. 

 2019 Elsevier Ltd. All rights reserved. 

## 1. Introduction

In structural dynamics, finite element (FE) analyses have been used for decades and are indispensable in current engineering practice. However, despite enormous improvement in computer performance, considerable computation time is required for the dynamic analysis of FE models with more than millions of degrees of freedom (DOFs). Because computational efficiency is of great interest to deal with large FE models as well as reliability of numerical solutions, significant efforts have been made to reduce the computation time and computer memory requirements. Component mode synthesis (CMS) [1–13] has been widely used to reduce FE models in various engineering fields, such as parametric analysis [14–16], experimental modal analysis [17,18], damage detection [19], and protein dynamics [20]. 

In particular, the automated multilevel substructuring (AMLS) method [21–24], a multilevel CMS method, is an efficient model reduction scheme for large FE models. In the AMLS method, global (non-reduced) mass and stiffness matrices of a FE model are automatically reordered using the nested dissection algorithm and divided into many submatrices (substructures) [25–30]. Then, through a recursive transformation procedure, a reduced model is constructed. The procedure only retains the dominant substructural normal modes and substructural constraint modes and truncates the residual substructural normal modes. Finally, a reduced system is solved, and approximate global responses are computed through a back transformation. The AMLS method has been adopted in commercial FE software and applied to numerous practical engineering problems owing to its effectiveness [31–34]. 

Recently, studies have been conducted to improve the solution accuracy of the AMLS method [1,35–37] and to estimate errors in the AMLS method [38,39]. The enhanced AMLS (EAMLS) method [1] was developed to compensate for the residual substructural normal modes, while such modes are simply truncated in the original AMLS method. The EAMLS method produces a more accurate reduced model than that of the AMLS method. However, unlike the AMLS method, the EAMLS method hardly handles FE models with large DOFs (more than one million DOFs). This is a critical drawback of the EAMLS method. 

In the EAMLS method, the compensation procedure of the residual mode effect requires multiplying some sparse matrices by the multilevel constraint mode matrix that generally contains a number of dense block matrices. Furthermore, the multilevel constraint mode matrix with the same dimensions as the global matrix should be calculated explicitly, while the calculation is done implicitly in the AMLS method [1,22]. This compensation procedure with all residual flexibility matrices produces the fully populated reduced matrices and significantly increases the computational cost even if the procedure is performed in a submatrix level computation. For the above reasons, the EAMLS method is only available for FE models with tens of thousands of DOFs in personal computers. 


(c)



(a)


In this study, we propose a novel algorithm to improve the computational efficiency of the original EAMLS method. First, a new pattern of partitioned matrices is proposed. Then, the recursive transformation procedure is conducted, and an interface subspace reduction is subsequently performed. To compensate for the residual mode effect, a residual mode correction is employed for the mass matrix using the residual flexibility matrices for only the bottom-level substructures. This approach does not need to calculate the multilevel constraint mode matrix explicitly and prevents the reduced matrices from being fully populated. All the calculations are performed in a submatrix level without global matrix operations. The proposed methodology substantially reduces the bottleneck of the original EAMLS method while minimizing loss of accuracy. It can speed up the reduction procedure with less memory usage. 

In the following sections, the new EAMLS method is derived. Its submatrix formulation is provided for an efficient computation, and the performance of the new method is presented through several numerical examples. 

## 2. New enhanced AMLS method

In this section, we derive the new formulation of the enhanced AMLS (EAMLS) method. The key procedures are the recursive transformation of the substructures, interface subspace reduction, and residual mode correction. These procedures focus on improving the computational performance of the original EAMLS method [1]. 

## 2.1. Recursive transformation of the substructures

In the AMLS and original EAMLS methods [1,21–24], the global structure is partitioned into several substructures, as shown in Fig. 1(a). This process is referred to as multilevel substructuring, and its hierarchy is represented by the tree diagram, see Fig. 1 (b). From this tree diagram, the bottom- and higher-level substructures are defined, and the sets of ancestors and descendants corresponding to the ith substructure, A and $D _ { i } ,$ are defined [22]. For example, in Fig. 1(b), A is {7}, and $D _ { 3 }$ is {1, 2}. The other sets can be also defined in the same way. Based on this tree diagram, the block matrix pattern for the AMLS and original EAMLS methods is described, as shown in Fig. 1(c). 

In the new EAMLS method, the recursive transformation procedure of the AMLS method is modified. Multilevel substructuring and numbering are performed as shown in Fig. 2(a). The substructure numbers for all bottom-level substructures are designated, and then, the substructure numbers for the rest of the substructures (i.e., higher-level substructures) are designated. Consequently, the tree diagram and block matrix pattern are defined as shown in Fig. 2(b) and (c), respectively. 

After substructuring into n substructures, the generalized eigenvalue problem of the global FE model is defined as 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/9d368328d3dbf55a878ff9c7b7d0ef7c04a5fc8ceaad60a387f997b929a44f54.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/91e880651a4e15553d3dbac50e41bc345e8397dd961fa594d61fcee2ccf23936.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/661739ccef9de0c47f81a907ad1c1e3a83a1106ee2f56657494d41e87017afc6.jpg)



Fig. 1. Multilevel substructuring for the AMLS and original EAMLS methods: (a) partitioned structure, (b) tree diagram, and (c) block matrix pattern for the reordered matrix


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/658e66f3d7e8a53af31cb84aa5edd2d5cc4a7c52bdd9004549916ef086acd2d5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/dbf32963c7edfc313164d4b4d29d1d2fce8d9f5f2f073f0c058e469b8e8ef002.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/2afcea917af41f5b923511b6cd6c51f573f7c90cf59a128e55193129b5d599de.jpg)



(c)



Fig. 2. Multilevel substructuring for the new EAMLS method: (a) partitioned structure, (b) tree diagram, and (c) block matrix pattern for the reordered matrix.


$$
\begin{array}{l} \mathbf {K} \varphi = \omega^ {2} \mathbf {M} \varphi \\ \text {with} \\ \mathbf {M} = \left[ \begin{array}{c c c c c} \mathbf {M} _ {1, 1} & & & & \\ & \ddots & & & \\ & & \mathbf {M} _ {i, i} & \mathbf {M} _ {i, j} & \\ & \text {sym.} & & \ddots & \\ & & & & \mathbf {M} _ {n, n} \end{array} \right], \\ \mathbf {K} = \left[ \begin{array}{c c c c c} \mathbf {K} _ {1, 1} & & & & \\ & \ddots & & \\ & & \mathbf {K} _ {i, i} & \mathbf {K} _ {i, j} & \\ & \text {sym.} & & \ddots & \\ & & & & \mathbf {K} _ {n, n} \end{array} \right] \text {for i = 1, 2, ..., n, ∀j∈A_{i}}, \end{array}\tag{1}
$$

where M and K are the global mass and stiffness matrices of $N \times N \left( N \right)$ is the number of DOFs in the global matrix), respectively; $\varphi$ is the global mode corresponding to the natural frequency x. The diagonal matrices $\mathbf { M } _ { i , i }$ and ${ \bf K } _ { i , i }$ denote the mass and stiffness matrices of the ith substructure, respectively. The off-diagonal matrices $\mathbf { M } _ { i , j }$ and ${ \bf K } _ { i , j }$ denote the mass and stiffness matrices of the ith substructure coupled with the jth substructure, respectively, where the jth substructure denotes the ancestors for the ith substructure, see Fig. 2(b) and $( \mathsf { c } ) .$ The off-diagonal matrices $\mathbf { M } _ { i , j }$ and $\mathbf { K } _ { i , j }$ are the zero matrices when the jth substructure is not the ancestor of the ith substructure. 

By performing the recursive transformation procedure, the ith transformed mass and stiffness matrices are obtained as follows: 

$$
\begin{array}{l} \mathbf {M} ^ {(i)} = \Big (\mathbf {T} ^ {(1)} \mathbf {T} ^ {(2)} \dots \mathbf {T} ^ {(i)} \Big) ^ {T} \mathbf {M} \Big (\mathbf {T} ^ {(1)} \mathbf {T} ^ {(2)} \dots \mathbf {T} ^ {(i)} \Big), \\ \mathbf {K} ^ {(i)} = \Big (\mathbf {T} ^ {(1)} \mathbf {T} ^ {(2)} \dots \mathbf {T} ^ {(i)} \Big) ^ {T} \mathbf {K} \Big (\mathbf {T} ^ {(1)} \mathbf {T} ^ {(2)} \dots \mathbf {T} ^ {(i)} \Big) \text {for} i = 1, 2, \dots , n, \end{array}\tag{2}
$$

in which $\mathbf { M } ^ { ( i ) }$ and $\mathbf { K } ^ { ( i ) } \left( i < n \right)$ ) are the incompletely transformed mass and stiffness matrices, respectively, $\mathbf { M } ^ { ( n ) }$ and $\mathbf { K } ^ { ( n ) }$ are the completely transformed mass and stiffness matrices, respectively, and $\mathbf { T } ^ { ( i ) }$ is the recursive transformation matrix for the ith transformation. 

In Eq. (2), $\mathbf { T } ^ { ( i ) }$ is defined by 

$$
\mathbf {T} ^ {(i)} = \left[ \begin{array}{c c c c c c c} \mathbf {I} _ {1} & & & & & \\ & \ddots & & & & \\ & & \mathbf {I} _ {i - 1} & & & \\ & & & \boldsymbol {\Phi} _ {i} & & \boldsymbol {\Psi} _ {i, j} \\ & & & & \mathbf {I} _ {i + 1} & & \\ & & & & & \ddots & \\ & & & & & & \mathbf {I} _ {n} \end{array} \right] \text {for} i = 1, 2, \dots , n, \forall j \in A _ {i},\tag{3}
$$

where $\Phi _ { i }$ is the substructural normal mode matrix for the ith substructure, $\Psi _ { i , j }$ is the substructural constraint mode matrix of the ith substructure coupled with the jth substructure, and I denotes the identity matrices. The substructural constraint mode matrix $\Psi _ { i , j }$ is the zero matrix when the jth substructure is not the ancestor for the ith substructure. 

The substructural normal mode matrix $\Phi _ { i }$ can be computed by the following substructural eigenvalue problem: 

$$
\mathbf {K} _ {i, i} ^ {(i - 1)} \boldsymbol {\Phi} _ {i} = \mathbf {M} _ {i, i} ^ {(i - 1)} \boldsymbol {\Phi} _ {i} \boldsymbol {\Lambda} _ {i} \quad \text {   for   } i = 1, 2, \dots , n,\tag{4}
$$

in which $\mathbf { M } _ { i , i } ^ { ( i - 1 ) }$ and $\mathbf { K } _ { i , i } ^ { ( i - 1 ) }$ are the substructural mass and stiffness matrices corresponding to the ith substructure in the (i-1)th transformed mass and stiffness matrices, $\mathbf { M } ^ { ( i - 1 ) }$ and $\mathbf { K } ^ { ( i - 1 ) }$ <sup>Þ</sup>, respectively. 

The substructural eigenvalue matrix $\Lambda _ { i }$ is a diagonal matrix consisting of eigenvalues, squares of the natural frequencies, for the ith substructure. Here, $\mathbf { M } ^ { ( 0 ) }$ and $\mathbf { K } ^ { ( 0 ) }$ are identical to M and K in Eq. (1), respectively. 

The substructural constraint mode matrix $\Psi _ { i , j }$ is defined as 

$$
\boldsymbol {\Psi} _ {i, j} = - \left(\mathbf {K} _ {i, i} ^ {(i - 1)}\right) ^ {- 1} \mathbf {K} _ {i, j} ^ {(i - 1)} \quad \text { for } i = 1, 2, \dots , n - 1, \forall j \in A _ {i},\tag{5}
$$

where $\mathbf { K } _ { i , j } ^ { ( i - 1 ) }$ is the off-diagonal matrix in $\mathbf { K } ^ { ( i - 1 ) }$ , which corresponds to the ith substructure coupled with the jth substructure. 

The substructural normal mode and eigenvalue matrices in Eq. (4), U<sub>i</sub> and $\Lambda _ { i } ,$ , are decomposed by a given cutoff frequency for the substructures as follows: 

$$
\boldsymbol {\Phi} _ {i} = \left[ \begin{array}{c c} \boldsymbol {\Phi} _ {i} ^ {d} & \boldsymbol {\Phi} _ {i} ^ {r} \end{array} \right],   \boldsymbol {\Lambda} _ {i} = \left[ \begin{array}{c c} \boldsymbol {\Lambda} _ {i} ^ {d} & \boldsymbol {0} \\ \boldsymbol {0} & \boldsymbol {\Lambda} _ {i} ^ {r} \end{array} \right],\tag{6}
$$

in which $\Phi _ { i } ^ { d }$ and $\Phi _ { i } ^ { r }$ are the dominant and residual substructural normal mode matrices of $\mathbf { M } _ { i , i } ^ { ( i - 1 ) }$ and $\mathbf { K } _ { i , i } ^ { ( i - 1 ) }$ , respectively, and $\boldsymbol { \Lambda } _ { i } ^ { d }$ and ${ \boldsymbol { \Lambda } } _ { i } ^ { r }$ are the corresponding eigenvalue matrices. Note that all diagonal entries in $\boldsymbol { \Lambda } _ { i } ^ { d }$ are smaller than the square of the given cutoff frequency. 

In Eq. (2), the complete transformation matrix T is defined as 

$$
\mathbf {T} = \mathbf {T} ^ {(1)} \mathbf {T} ^ {(2)} \dots \mathbf {T} ^ {(n)} = \prod_ {i = 1} ^ {n} \mathbf {T} ^ {(i)},\tag{7}
$$

and it can be decomposed into two parts as follows: [1,22] 

$$
\begin{array}{l} \mathbf {T} = \hat {\boldsymbol {\Psi}} \boldsymbol {\Phi} \text {with} \hat {\boldsymbol {\Psi}} = \left[ \begin{array}{c c c c c} \mathbf {I} _ {1} & & & & \\ & \ddots & & \\ & & \mathbf {I} _ {i} & \hat {\boldsymbol {\Psi}} _ {i, j} \\ & & & \ddots & \\ & & & & \mathbf {I} _ {n} \end{array} \right], \boldsymbol {\Phi} = \left[ \begin{array}{c c c c c} \boldsymbol {\Phi} _ {1} & & & & \\ & \ddots & & \\ & & \boldsymbol {\Phi} _ {i} & & \\ & & & \ddots & \\ & & & & \boldsymbol {\Phi} _ {n} \end{array} \right] \\ \text {for i = 1,2,\cdots,n,}\forall j \in A _ {i}, \end{array}\tag{8}
$$

where W<sup>^</sup> and $\hat { \Psi } _ { i , j }$ denote the multilevel constraint mode matrix and its substructural component, respectively; U is the normal mode matrix consisting of all substructural normal mode matrices $\Phi _ { i }$ Here, $\hat { \Psi } _ { i , j }$ is the zero matrix when the jth substructure is not the ancestor of the ith substructure. 

Considering U<sup>d</sup> and U<sup>r</sup> in Eq. (6), the normal mode matrix U in Eq. (8) can be reordered and divided as follows: 

$$
\begin{array}{l} \boldsymbol {\Phi} = [ \boldsymbol {\Phi} _ {d} \quad \boldsymbol {\Phi} _ {r} ] \\ \text {with} \\ \boldsymbol {\Phi} _ {d} = \left[ \begin{array}{c c c c} \boldsymbol {\Phi} _ {1} ^ {d} & & & \\ & \boldsymbol {\Phi} _ {2} ^ {d} & & \\ & & \ddots & \\ & & & \boldsymbol {\Phi} _ {n} ^ {d} \end{array} \right], \boldsymbol {\Phi} _ {r} = \left[ \begin{array}{c c c c} \boldsymbol {\Phi} _ {1} ^ {r} & & & \\ & \boldsymbol {\Phi} _ {2} ^ {r} & & \\ & & \ddots & \\ & & & \boldsymbol {\Phi} _ {n} ^ {r} \end{array} \right]. \end{array}\tag{9}
$$

From Eq. (9), the transformation matrix T in Eq. (8) is decomposed into two parts as follows: 

$$
\mathbf {T} = \left[ \begin{array}{c c} \mathbf {T} _ {d} & \mathbf {T} _ {r} \end{array} \right] \text { with } \mathbf {T} _ {d} = \hat {\boldsymbol {\Psi}} \boldsymbol {\Phi} _ {d},   \mathbf {T} _ {r} = \hat {\boldsymbol {\Psi}} \boldsymbol {\Phi} _ {r},\tag{10}
$$

in which $\mathbf { T } _ { d }$ and T denote the dominant and residual parts of the transformation matrix T, respectively. 

Utilizing only $\mathbf { T } _ { d }$ in Eq. (10), the reduced mass and stiffness matrices from the recursive transformation procedure are given by 

$$
\bar {\mathbf {M}} = \mathbf {T} _ {d} ^ {T} \mathbf {M} \mathbf {T} _ {d}, \bar {\mathbf {K}} = \mathbf {T} _ {d} ^ {T} \mathbf {K} \mathbf {T} _ {d},\tag{11}
$$

where M<sup>-</sup> and K<sup>-</sup> are equal to $\mathbf { M } ^ { ( n ) }$ and $\mathbf { K } ^ { ( n ) }$ in Eq. (2), respectively. 

In the actual implementation, the matrix $\Phi _ { r }$ in Eq. (10) is not computed; the Rayleigh-Ritz procedure [40] described in Eqs. (2) and (11) and the calculation of W<sup>^</sup> in $\mathsf { E q . } ( 1 0 )$ are not explicitly performed to construct the matrices M<sup>-</sup> , K<sup>-</sup>, and $\mathbf { T } _ { d } .$ Instead, for computational efficiency, all substructural components in $\bar { \bf M } , \bar { \bf K } ,$ and $\mathbf { T } _ { d }$ are calculated using a submatrix computation strategy [22,23] given in Appendix. 

## 2.2. Interface subspace reduction

After performing the recursive transformation procedure, the tree diagram and block matrix pattern are changed, as shown in Fig. 3. From Fig. 2(a), the higher-level substructures can be regarded as an interface boundary based on the geometric perspective. Hereinafter, the higher-level substructures are referred to as the interface, and thus, we can define the interface subspace $\bar { \Omega } _ { \Gamma }$ as shown in Fig. 3. 

Unlike the AMLS and original EAMLS methods [1,22], two different cutoff frequencies $\omega _ { b }$ and $\omega _ { \Gamma }$ are given and applied separately to the bottom-level substructures and the interface, respectively. The cutoff frequency $\omega _ { \Gamma }$ is set larger than $\omega _ { b }$ to better approximate the interface behavior, and we set $\omega _ { \Gamma } = 1 6 . 5 \omega _ { c }$ and $\omega _ { b } = 1 1 \omega _ { c }$ when the highest excitation frequency is $\omega _ { c } .$ . However, these cutoff frequencies would incur the large interface subspace. Therefore, an interface subspace reduction is performed to obtain a new smaller subspace. The reason for adopting this frequency cutoff method will be discussed in Section 2.3. 

Based on Fig. 3(b) and (c), the reduced mass and stiffness matrices M<sup>-</sup> and K<sup>-</sup> in Eq. (11) can be represented by 

$$
\bar {\mathbf {M}} = \left[ \begin{array}{c c} \bar {\mathbf {I}} _ {b} & \bar {\mathbf {M}} _ {b, \Gamma} \\ \bar {\mathbf {M}} _ {b, \Gamma} ^ {T} & \bar {\mathbf {M}} _ {\Gamma} \end{array} \right],   \bar {\mathbf {K}} = \left[ \begin{array}{c c} \bar {\mathbf {K}} _ {b} & \mathbf {0} \\ \mathbf {0} & \bar {\mathbf {K}} _ {\Gamma} \end{array} \right],\tag{12}
$$

in which the subscripts b and C denote the bottom-level substructural and interface quantities, respectively. The matrices $\bar { \mathbf { M } } _ { \Gamma }$ and $\bar { \bf K } _ { \Gamma }$ denote the mass and stiffness matrices for the interface subspace $\bar { \Omega } _ { \Gamma } ,$ , respectively, and $\bar { \mathbf { M } } _ { b , \Gamma }$ denotes the mass matrix of the reduced bottom-level substructures coupled with the interface subspace. The size of the identity matrix $\bar { \mathbf { I } } _ { b } \mathrm { i } s \bar { N } _ { b } \times \bar { N } _ { b }$ , where $\bar { N } _ { b }$ is the number of dominant substructural normal modes for all bottom-level substructures. 

The eigenvalue problem for the interface subspace is defined as 

$$
\bar {\mathbf {K}} _ {\Gamma} \boldsymbol {\Xi} = \bar {\mathbf {M}} _ {\Gamma} \boldsymbol {\Xi} \boldsymbol {\Theta} \text {   with   } \boldsymbol {\Xi} = [ \boldsymbol {\Xi} _ {d} \quad \boldsymbol {\Xi} _ {r} ], \boldsymbol {\Theta} = \left[ \begin{array}{c c} \boldsymbol {\Theta} _ {d} & \boldsymbol {0} \\ \boldsymbol {0} & \boldsymbol {\Theta} _ {r} \end{array} \right],\tag{13}
$$

where N and H are the normal mode and eigenvalue matrices of the interface subspace $\bar { \Omega } _ { \Gamma }$ , respectively, and they are decomposed into the dominant and residual parts $( \Xi _ { d }$ and $\Xi _ { r } , \Theta _ { d }$ and H ). 

Utilizing the matrix N in Eq. (13), the transformation matrix for the interface subspace reduction is defined as 

$$
\mathbf {T} _ {s} = \left[ \begin{array}{c c} \bar {\mathbf {I}} _ {b} & \mathbf {0} \\ \mathbf {0} & \boldsymbol {\Xi} \end{array} \right],\tag{14}
$$

and it can be decomposed into the dominant and residual parts as 

$$
\mathbf {T} _ {s} = \left[ \begin{array}{c c} \mathbf {T} _ {s} ^ {d} & \mathbf {T} _ {s} ^ {r} \end{array} \right] \text { with } \mathbf {T} _ {s} ^ {d} = \left[ \begin{array}{c c} \bar {\mathbf {I}} _ {b} & \mathbf {0} \\ \mathbf {0} & \boldsymbol {\Xi} _ {d} \end{array} \right],   \mathbf {T} _ {s} ^ {r} = \left[ \begin{array}{c} \mathbf {0} \\ \boldsymbol {\Xi} _ {r} \end{array} \right].\tag{15}
$$

Multiplying the transformation matrix $\mathbf { T } _ { d }$ in Eq. (10) by ${ \bf T } _ { s }$ in Eq. (15), the following equation is obtained: 

$$
\hat {\mathbf {T}} _ {d} = \mathbf {T} _ {d} \mathbf {T} _ {s} = \left[ \begin{array}{c c} \mathbf {T} _ {d} \mathbf {T} _ {s} ^ {d} & \mathbf {T} _ {d} \mathbf {T} _ {s} ^ {r} \end{array} \right].\tag{16}
$$

Then, <sup>^</sup>T replaces $\mathbf { T } _ { d }$ in the transformation matrix $\mathbf { T } ( = \left[ \begin{array} { l l } { \mathbf { T } _ { d } } & { \mathbf { T } _ { r } } \end{array} \right] )$ and the following equation is obtained: 

$$
\tilde {\mathbf {T}} = \left[ \begin{array}{c c} \hat {\mathbf {T}} _ {d} & \mathbf {T} _ {r} \end{array} \right] = \left[ \begin{array}{c c c} \mathbf {T} _ {d} \mathbf {T} _ {s} ^ {d} & \mathbf {T} _ {d} \mathbf {T} _ {s} ^ {r} & \mathbf {T} _ {r} \end{array} \right].\tag{17}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/c300dff159427ae217296058e5498618783ebcb7ce914004852a84e28c71db7d.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/9b3d3af41cc587274b03490fe921547d9918b60efc356ccc2db3fdc292afd8cb.jpg)



(b)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/edc1a6dac99f6a0b79c7d00b31d24f64e7bc8df9d8c6faf8809afdf42eef740a.jpg)



(c)



Fig. 3. Description of substructures after the recursive transformation procedure: (a) tree diagram, (b) block matrix pattern for the mass matrix, and (c) block matrix pattern for the stiffness matrix. Here, X<sup>-</sup> indicates the ith reduced substructure, and $\bar { \Omega } _ { b }$ and $\bar { \Omega } _ { \Gamma }$ denote the reduced bottom-level substructures and interface subspace, respectively.


After switching the columns corresponding to $\mathbf { T } _ { d } \mathbf { T } _ { s } ^ { r }$ and ${ \bf T } _ { r }$ in $\mathtt { E q } .$ (17), the transformation matrix $\tilde { \mathbf { T } }$ is decomposed into the dominant and residual parts as follows: 

$$
\widetilde {\mathbf {T}} = \left[ \begin{array}{c c} \widetilde {\mathbf {T}} _ {d} & \widetilde {\mathbf {T}} _ {r} \end{array} \right] \text {with} \widetilde {\mathbf {T}} _ {d} = \mathbf {T} _ {d} \mathbf {T} _ {s} ^ {d}, \widetilde {\mathbf {T}} _ {r} = \left[ \begin{array}{c c} \mathbf {T} _ {r} & \mathbf {T} _ {d} \mathbf {T} _ {s} ^ {r} \end{array} \right],\tag{18}
$$

in which $\tilde { \mathbf { T } } _ { d }$ is the newly derived dominant transformation matrix based on the interface subspace reduction in the proposed method, and $\tilde { \mathbf { T } } _ { r }$ is the corresponding residual transformation matrix. 

Using Eqs. (10) and (15) in Eq. (18), the dominant and residual transformation matrices, $\tilde { \mathbf { T } } _ { d }$ and $\tilde { \mathbf { T } } _ { r }$ , can be represented as 

$$
\begin{array}{l} \widetilde {\mathbf {T}} _ {d} = \hat {\boldsymbol {\Psi}} \widetilde {\boldsymbol {\Phi}} _ {d}, \widetilde {\mathbf {T}} _ {r} = \hat {\boldsymbol {\Psi}} \widetilde {\boldsymbol {\Phi}} _ {r} \\ \text {with} \widetilde {\boldsymbol {\Phi}} _ {d} = \left[ \begin{array}{c c} \boldsymbol {\Phi} _ {b} ^ {d} & \boldsymbol {0} \\ \boldsymbol {0} & \boldsymbol {\Phi} _ {\Gamma} ^ {d} \boldsymbol {\Xi} _ {d} \end{array} \right], \widetilde {\boldsymbol {\Phi}} _ {r} = \left[ \begin{array}{c c c} \boldsymbol {\Phi} _ {b} ^ {r} & \boldsymbol {0} & \boldsymbol {0} \\ \boldsymbol {0} & \boldsymbol {\Phi} _ {\Gamma} ^ {r} & \boldsymbol {\Phi} _ {\Gamma} ^ {d} \boldsymbol {\Xi} _ {r} \end{array} \right], \end{array}\tag{19}
$$

where $\Phi _ { b } ^ { d }$ and $\Phi _ { \Gamma } ^ { d }$ are the normal mode matrices corresponding to the bottom-level substructures and the interface of $\Phi _ { d }$ in $\operatorname { E q . } \ ( 9 )$ respectively. Similarly, $\Phi _ { b } ^ { r }$ and $\Phi _ { \Gamma } ^ { r }$ are the bottom-level substructural and interface parts of $\Phi _ { r }$ in Eq. (9), respectively. 

Employing only the dominant transformation matrix $\tilde { \mathbf { T } } _ { d }$ in Eq. (18), we obtain the reduced mass and stiffness matrices based on the interface subspace reduction as follows: 

$$
\tilde {\mathbf {M}} _ {d} = \tilde {\mathbf {T}} _ {d} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {d} = \left(\mathbf {T} _ {s} ^ {d}\right) ^ {T} \bar {\mathbf {M}} \left(\mathbf {T} _ {s} ^ {d}\right) = \left[ \begin{array}{c c} \bar {\mathbf {I}} _ {b} & \tilde {\mathbf {M}} _ {c} \\ \tilde {\mathbf {M}} _ {c} ^ {T} & \tilde {\mathbf {I}} _ {\Gamma} \end{array} \right]\tag{20a}
$$

with $\widetilde { \mathbf { M } } _ { c } = \bar { \mathbf { M } } _ { b , \Gamma } \boldsymbol { \Xi } _ { d } , \widetilde { \mathbf { I } } _ { \Gamma } = \boldsymbol { \Xi } _ { d } ^ { T } \bar { \mathbf { M } } _ { \Gamma } \boldsymbol { \Xi } _ { d }$ ; 

$$
\widetilde {\mathbf {K}} _ {d} = \widetilde {\mathbf {T}} _ {d} ^ {T} \widetilde {\mathbf {K}} \widetilde {\mathbf {T}} _ {d} = \left(\mathbf {T} _ {s} ^ {d}\right) ^ {T} \bar {\mathbf {K}} \Big (\mathbf {T} _ {s} ^ {d} \Big) = \left[ \begin{array}{c c} \bar {\mathbf {K}} _ {b} & \mathbf {0} \\ \mathbf {0} & \widetilde {\mathbf {K}} _ {\Gamma} \end{array} \right]\tag{20b}
$$

$$
\tilde {\mathbf {K}} _ {\Gamma} = \boldsymbol {\Xi} _ {d} ^ {T} \bar {\mathbf {K}} _ {\Gamma} \boldsymbol {\Xi} _ {d} = \boldsymbol {\Theta} _ {d},
$$

in which $\tilde { \mathbf { I } } _ { \Gamma }$ is the identity matrix of $\stackrel { \sim } { N } _ { \Gamma } \times \stackrel { \sim } { N } _ { \Gamma } ( \stackrel { \sim } { N } _ { \Gamma }$ is the number of dominant normal modes of the interface subspace), and $\tilde { \mathbf { M } } _ { d }$ and $\tilde { \mathbf { K } } _ { d }$ are the $\tilde { N } \times \tilde { N }$ matrices $( \stackrel { \sim } { N } = \bar { N } _ { b } + \stackrel { \sim } { N } _ { \Gamma } )$ 

Then, the eigenvalue problem for $\dot { \mathbf { M } } _ { d }$ and $\widetilde { \mathbf { K } } _ { d }$ is given by 

$$
\tilde {\mathbf {K}} _ {d} \mathbf {x} _ {d} = \tilde {\omega} ^ {2} \tilde {\mathbf {M}} _ {d} \mathbf {x} _ {d},\tag{21}
$$

where $\mathbf { x } _ { d }$ is the reduced mode corresponding to the approximate natural frequency ${ \tilde { \omega } } .$ 

## 2.3. Residual mode correction

In the original EAMLS method, the residual mode effect is compensated for all substructures, except for the highest-level substructure [1]. This guarantees highly accurate solutions; however, the number of nonzero block matrices increases, which reduces the computational efficiency. Because the primary objective of model reduction methods is to deal with large FE models efficiently, the efficiency of the EAMLS method should be improved with minimal loss of accuracy. 

In the new EAMLS method, the residual mode effect is only considered for the bottom-level substructures to decrease the number of nonzero block matrices. This approach reduces the solution accuracy but can provide much more efficient calculations to handle large FE models. To minimize the loss of accuracy, the interface behavior should be accurately approximated using sufficient dominant normal modes. Therefore, the two cutoff frequencies $\omega _ { b }$ and $\omega _ { \Gamma }$ are employed, as described in Section 2.2. 

Assuming that the Rayleigh-Ritz procedure is conducted with the transformation matrix T in Eq. (18), the generalized eigenvalue problem in Eq. (1) is expressed as 

$$
\left[ \begin{array}{c c} \widetilde {\mathbf {K}} _ {d} - \omega^ {2} \widetilde {\mathbf {M}} _ {d} & - \omega^ {2} \widetilde {\mathbf {M}} _ {d r} \\ - \omega^ {2} \widetilde {\mathbf {M}} _ {d r} ^ {T} & \widetilde {\mathbf {K}} _ {r} - \omega^ {2} \widetilde {\mathbf {M}} _ {r} \end{array} \right] \left[ \begin{array}{c} \mathbf {x} _ {d} \\ \mathbf {x} _ {r} \end{array} \right] = \left[ \begin{array}{c} \mathbf {0} \\ \mathbf {0} \end{array} \right]
$$

$$
\tilde {\mathbf {K}} _ {r} = \tilde {\mathbf {T}} _ {r} ^ {T} \mathbf {K} \tilde {\mathbf {T}} _ {r}, \tilde {\mathbf {M}} _ {d r} = \tilde {\mathbf {T}} _ {d} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {r}, \tilde {\mathbf {M}} _ {r} = \tilde {\mathbf {T}} _ {r} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {r},\tag{22}
$$

and the global mode $\varphi$ in Eq. (1) can be expressed with the transformation matrix $\tilde { \mathbf { T } }$ as follows: 

$$
\boldsymbol {\varphi} = \tilde {\mathbf {T}} \mathbf {x} \text {   with   } \tilde {\mathbf {T}} = \left[ \begin{array}{c c} \tilde {\mathbf {T}} _ {d} & \tilde {\mathbf {T}} _ {r} \end{array} \right],   \mathbf {x} = \left[ \begin{array}{c} \mathbf {x} _ {d} \\ \mathbf {x} _ {r} \end{array} \right].\tag{23}
$$

Expanding the second row equation of Eq. (22), x is written as 

$$
\mathbf {x} _ {r} = \omega^ {2} \left(\tilde {\mathbf {K}} _ {r} - \omega^ {2} \tilde {\mathbf {M}} _ {r}\right) ^ {- 1} \tilde {\mathbf {M}} _ {d r} ^ {T} \mathbf {x} _ {d},\tag{24}
$$

and substituting Eq. (24) into Eq. (23), the global mode $\varphi$ can be represented in terms of $\mathbf { x } _ { d }$ without ${ \bf x } _ { r }$ as follows: 

$$
\boldsymbol {\varphi} = \left[ \tilde {\mathbf {T}} _ {d} + \omega^ {2} \tilde {\mathbf {T}} _ {r} \left(\tilde {\mathbf {K}} _ {r} - \omega^ {2} \tilde {\mathbf {M}} _ {r}\right) ^ {- 1} \tilde {\mathbf {M}} _ {d r} ^ {T} \right] \mathbf {x} _ {d}.\tag{25}
$$

Using $\tilde { \mathbf { T } } _ { r } = \hat { \Psi } \tilde { \mathbf { \Phi } } _ { } ^ { \sim }$ in Eq. (19) and $\tilde { \mathbf { M } } _ { d r } ^ { T } = \tilde { \mathbf { T } } _ { r } ^ { T } \mathbf { M } \tilde { \mathbf { T } } _ { d }$ in Eq. (22), the global mode u in Eq. (25) is rewritten as 

$$
\begin{array}{l} \boldsymbol {\varphi} = \left[ \tilde {\mathbf {T}} _ {d} + \omega^ {2} \hat {\boldsymbol {\Psi}} \tilde {\mathbf {F}} _ {r} \hat {\boldsymbol {\Psi}} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {d} \right] \mathbf {x} _ {d} \\ \text {with} \tilde {\mathbf {F}} _ {r} = \tilde {\boldsymbol {\Phi}} _ {r} \left(\tilde {\mathbf {K}} _ {r} - \omega^ {2} \tilde {\mathbf {M}} _ {r}\right) ^ {- 1} \tilde {\boldsymbol {\Phi}} _ {r} ^ {T}, \end{array}\tag{26}
$$

in which $\widetilde { \mathbf { F } } _ { r }$ is the residual flexibility matrix [1]. 

The residual flexibility matrix $\widetilde { \mathbf { F } } _ { r }$ in $\operatorname { E q . }$ (26) can be expanded as 

$$
\tilde {\mathbf {F}} _ {r} = \tilde {\mathbf {\Phi}} _ {r} \tilde {\mathbf {K}} _ {r} ^ {- 1} \tilde {\mathbf {\Phi}} _ {r} ^ {T} + \omega^ {2} \tilde {\mathbf {\Phi}} _ {r} \tilde {\mathbf {K}} _ {r} ^ {- 1} \tilde {\mathbf {M}} _ {r} \tilde {\mathbf {K}} _ {r} ^ {- 1} \tilde {\mathbf {\Phi}} _ {r} ^ {T} + O (\omega^ {4}) + O (\omega^ {6}) + \dots ,\tag{27}
$$

and it can be approximated by neglecting the terms related to the unknown x as follows: 

$$
\tilde {\mathbf {F}} _ {r} \approx \tilde {\mathbf {F}} _ {r s} = \tilde {\mathbf {\Phi}} _ {r} \tilde {\mathbf {K}} _ {r} ^ {- 1} \tilde {\mathbf {\Phi}} _ {r} ^ {T},\tag{28}
$$

where $\tilde { \mathbf { F } } _ { r s }$ is the static part of the residual flexibility matrix [1]. 

To prevent the reduced mass and stiffness matrices from being fully populated, the residual flexibility is only considered for the bottom-level substructures in $\tilde { \mathbf { F } } _ { r s }$ . Then, the approximate residual flexibility matrix $\tilde { \mathbf { F } } _ { r b }$ is obtained as follows: 

$$
\tilde {\mathbf {F}} _ {r s} \approx \tilde {\mathbf {F}} _ {r b} = \left[ \begin{array}{c c c c c} \tilde {\mathbf {F}} _ {1} ^ {\sim r b} & & & & \\ & \tilde {\mathbf {F}} _ {2} ^ {\sim r b} & & & \\ & & \ddots & & \\ & & & \tilde {\mathbf {F}} _ {n _ {b}} ^ {\sim r b} & \\ & & & & \mathbf {0} \end{array} \right] = \left[ \begin{array}{c c} \tilde {\mathbf {F}} _ {b} ^ {\sim r b} & \mathbf {0} \\ \mathbf {0} & \mathbf {0} \end{array} \right]
$$

$$
\widetilde {\mathbf {F}} _ {i} ^ {r b} = \left(\mathbf {K} _ {i, i} ^ {(i - 1)}\right) ^ {- 1} - \boldsymbol {\Phi} _ {i} ^ {d} \left(\boldsymbol {\Lambda} _ {i} ^ {d}\right) ^ {- 1} \left(\boldsymbol {\Phi} _ {i} ^ {d}\right) ^ {T} \text {   for   } i = 1, 2, \dots , n _ {b},\tag{29}
$$

in which $n _ { b }$ denotes the number of bottom-level substructures. The matrices $\left( \mathbf { K } _ { i , i } ^ { ( i - 1 ) } \right) ^ { - 1 }$ and $\widetilde { \mathbf { F } } _ { i } ^ { r b }$ are the full and residual flexibility matrices for the ith substructure, respectively, and $\widetilde { \mathbf { F } } _ { b } ^ { r b }$ is the residual flexibility matrix for all bottom-level substructures. Note that we do not explicitly compute $\left( \mathbf { K } _ { i , i } ^ { ( i - 1 ) } \right) ^ { - 1 }$ , a fully populated matrix, when calculating $\widetilde { \mathbf { F } } _ { i } ^ { r b }$ . 

Using $\tilde { \mathbf { F } } _ { r b }$ in Eq. (29) instead of $\widetilde { \mathbf { F } } _ { r }$ in Eq. (26), the approximate global mode $\tilde { \varphi } _ { \epsilon }$ is defined as 

$$
\boldsymbol {\varphi} \approx \tilde {\boldsymbol {\varphi}} _ {e} = \left[ \tilde {\mathbf {T}} _ {d} + \omega^ {2} \hat {\boldsymbol {\Psi}} \tilde {\mathbf {F}} _ {r b} \hat {\boldsymbol {\Psi}} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {d} \right] \mathbf {x} _ {d}.\tag{30}
$$

From Eq. (30), the transformation matrix for the new EAMLS method is defined as 

$$
\tilde {\mathbf {T}} _ {e} = \tilde {\mathbf {T}} _ {d} + \omega^ {2} \tilde {\mathbf {T}} _ {a} \mathrm{with} \tilde {\mathbf {T}} _ {a} = \hat {\boldsymbol {\Psi}} \tilde {\mathbf {F}} _ {r b} \hat {\boldsymbol {\Psi}} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {d},\tag{31}
$$

where $\tilde { \mathbf { T } } _ { a }$ is the additional transformation matrix incorporating the residual mode effect for the bottom-level substructures. 

Employing Eq. (18), the dominant transformation matrix $\tilde { \mathbf { T } } _ { d }$ in Eq. (31) is represented by 

$$
\widetilde {\mathbf {T}} _ {d} = \mathbf {T} _ {d} \mathbf {T} _ {s} ^ {d} = \left[ \begin{array}{c c} \boldsymbol {\Phi} _ {b} ^ {d} & \widetilde {\mathbf {T}} _ {c} ^ {d} \\ \mathbf {0} & \widetilde {\mathbf {T}} _ {\Gamma} ^ {d} \end{array} \right] \text {with} \widetilde {\mathbf {T}} _ {c} ^ {d} = \mathbf {T} _ {b, \Gamma} ^ {d} \boldsymbol {\Xi} _ {d},   \widetilde {\mathbf {T}} _ {\Gamma} ^ {d} = \mathbf {T} _ {\Gamma} ^ {d} \boldsymbol {\Xi} _ {d},\tag{32}
$$

in which $\mathbf { T } _ { b , \mathrm { I } } ^ { d }$ and $\boldsymbol { \mathbf { T } } _ { \Gamma } ^ { d }$ are the submatrices of $\mathbf { T } _ { d }$ in Eq. (10) expressed as 

$$
\mathbf {T} _ {d} = \left[ \begin{array}{c c} \boldsymbol {\Phi} _ {b} ^ {d} & \mathbf {T} _ {b, \Gamma} ^ {d} \\ \mathbf {0} & \mathbf {T} _ {\Gamma} ^ {d} \end{array} \right].\tag{33}
$$

Here, $\mathbf { T } _ { b , \Gamma } ^ { d }$ and $\mathbf { T } _ { \Gamma } ^ { d }$ denote the coupled and interface quantities in $\mathbf { T } _ { d } ,$ respectively. 

The additional transformation matrix $\tilde { \mathbf { T } } _ { a }$ in Eq. (31) is expressed as 

$$
\tilde {\mathbf {T}} _ {a} = \hat {\boldsymbol {\Psi}} \tilde {\mathbf {F}} _ {r b} \hat {\boldsymbol {\Psi}} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {d} = \left[ \begin{array}{c c} \tilde {\mathbf {F}} _ {b} ^ {r b} \mathbf {M} _ {b} \boldsymbol {\Phi} _ {b} ^ {d} & \tilde {\mathbf {F}} _ {b} ^ {r b} \mathbf {Q} _ {b} \\ \mathbf {0} & \mathbf {0} \end{array} \right]\tag{34}
$$

with $\mathbf { Q } _ { b } = \mathbf { M } _ { b } \mathbf { \widetilde { T } } _ { c } ^ { d } + \mathbf { M } _ { b , \Gamma } \mathbf { \widetilde { T } } _ { \Gamma } ^ { d }$ 

where $\mathbf { M } _ { b }$ and $\mathbf { M } _ { b , \mathrm { I } }$ are the submatrices of M in $\operatorname { E q . } \left( 1 \right)$ represented by the bottom-level substructural and interface parts as follows: 

$$
\mathbf {M} = \left[ \begin{array}{c c} \mathbf {M} _ {b} & \mathbf {M} _ {b, \Gamma} \\ \mathbf {M} _ {b, \Gamma} ^ {T} & \mathbf {M} _ {\Gamma} \end{array} \right].\tag{35}
$$

Here, $\mathbf { M } _ { b } , \mathbf { M } _ { b , \Gamma }$ , and ${ \bf { M } } _ { \Gamma }$ denote the bottom-level substructural, coupled, and interface quantities, respectively. Note that, unlike the original EAMLS method, it is unnecessary to calculate the multilevel constraint mode W<sup>^</sup> for the compensation procedure of the residual mode effect. 

In Eq. (34), $\widetilde { \mathbf { F } } _ { b } ^ { r b } \mathbf { M } _ { b } \mathbf { \Phi } \Phi _ { b } ^ { d }$ is the zero matrix owing to the mass orthogonality of $\widetilde { \mathbf { F } } _ { b } ^ { r b }$ and $\Phi _ { b } ^ { d } ,$ and thus $\tilde { \mathbf { T } } _ { a }$ is rewritten as 

$$
\tilde {\mathbf {T}} _ {a} = \left[ \begin{array}{c c} \mathbf {0} & \tilde {\mathbf {T}} _ {m} ^ {a} \\ \mathbf {0} & \mathbf {0} \end{array} \right] \text {with} \tilde {\mathbf {T}} _ {m} ^ {a} = \tilde {\mathbf {F}} _ {b} ^ {r b} \mathbf {Q} _ {b}.\tag{36}
$$

Using $\tilde { \mathbf { T } } _ { e }$ in Eq. (31), the reduced mass and stiffness matrices are defined as 

$$
\tilde {\mathbf {M}} _ {e} = \tilde {\mathbf {T}} _ {e} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {e} = \tilde {\mathbf {M}} _ {d} + \omega^ {2} \tilde {\mathbf {T}} _ {d} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {a} + \omega^ {2} \tilde {\mathbf {T}} _ {a} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {d} + \omega^ {4} \tilde {\mathbf {T}} _ {a} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {a},\tag{37a}
$$

$$
\tilde {\mathbf {K}} _ {e} = \tilde {\mathbf {T}} _ {e} ^ {T} \tilde {\mathbf {K}} \tilde {\mathbf {T}} _ {e} = \tilde {\mathbf {K}} _ {d} + \omega^ {2} \tilde {\mathbf {T}} _ {d} ^ {T} \tilde {\mathbf {K}} \tilde {\mathbf {T}} _ {a} + \omega^ {2} \tilde {\mathbf {T}} _ {a} ^ {T} \tilde {\mathbf {K}} \tilde {\mathbf {T}} _ {d} + \omega^ {4} \tilde {\mathbf {T}} _ {a} ^ {T} \tilde {\mathbf {K}} \tilde {\mathbf {T}} _ {a},\tag{37b}
$$

and the reduced eigenvalue problem is given as 

$$
\begin{array}{r l} & {\left[ \tilde {\mathbf {K}} _ {d} + \omega^ {2} \tilde {\mathbf {T}} _ {d} ^ {T} \mathbf {K} \tilde {\mathbf {T}} _ {a} + \omega^ {2} \tilde {\mathbf {T}} _ {a} ^ {T} \mathbf {K} \tilde {\mathbf {T}} _ {d} + \omega^ {4} \tilde {\mathbf {T}} _ {a} ^ {T} \mathbf {K} \tilde {\mathbf {T}} _ {a} \right] \mathbf {x} _ {d}} \\ & {= \tilde {\omega} _ {e} ^ {2} \left[ \tilde {\mathbf {M}} _ {d} + \omega^ {2} \tilde {\mathbf {T}} _ {d} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {a} + \omega^ {2} \tilde {\mathbf {T}} _ {a} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {d} + \omega^ {4} \tilde {\mathbf {T}} _ {a} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {a} \right] \mathbf {x} _ {d}.} \end{array}\tag{38}
$$

Substituting $\tilde { \mathbf { T } } _ { d }$ in Eq. (32) and $\tilde { \mathbf { T } } _ { a }$ in Eq. (36) into the reduced matrices in Eq. (37) and using the mass and stiffness orthogonality of $\widetilde { \mathbf { F } } _ { b } ^ { r b }$ and $\Phi _ { b } ^ { d } ,$ , the following relations are obtained: [39] 

$$
\tilde {\mathbf {T}} _ {a} ^ {T} \mathbf {K} \tilde {\mathbf {T}} _ {a} = \tilde {\mathbf {T}} _ {d} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {a} = \tilde {\mathbf {T}} _ {a} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {d},\tag{39a}
$$

$$
\tilde {\mathbf {T}} _ {d} ^ {T} \mathbf {K} \tilde {\mathbf {T}} _ {a} = \tilde {\mathbf {T}} _ {a} ^ {T} \mathbf {K} \tilde {\mathbf {T}} _ {d} = \mathbf {0}.\tag{39b}
$$

Employing Eq. (39) and assuming $\omega \approx \tilde { \omega } _ { e }$ , the reduced eigenvalue problem in Eq. (38) is rewritten as 

$$
\tilde {\mathbf {K}} _ {d} \mathbf {x} _ {d} = \tilde {\omega} _ {e} ^ {2} \bigg [ \tilde {\mathbf {M}} _ {d} + \omega^ {2} \tilde {\mathbf {T}} _ {d} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {a} + \omega^ {4} \tilde {\mathbf {T}} _ {a} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {a} \bigg ] \mathbf {x} _ {d}.\tag{40}
$$

For the unknown natural frequency x in Eq. (40), the following relation from Eq. (21) is obtained: [1] 

$$
\omega^ {2} \mathbf {x} _ {d} \approx \tilde {\omega} ^ {2} \mathbf {x} _ {d} = \tilde {\mathbf {H}} \mathbf {x} _ {d} \text {with} \tilde {\mathbf {H}} = \tilde {\mathbf {M}} _ {d} ^ {- 1} \tilde {\mathbf {K}} _ {d}.\tag{41}
$$

Neglecting the fourth-order term of x $( \omega ^ { 4 } \widetilde { \mathbf { T } } _ { a } ^ { T } \mathbf { M } \widetilde { \mathbf { T } } _ { a } )$ in Eq. (40) and substituting Eq. (41) into Eq. (40), the new reduced eigenvalue problem is given as 

$$
\tilde {\mathbf {K}} _ {d} \mathbf {x} _ {d} = \tilde {\omega} _ {e} ^ {2} \left[ \tilde {\mathbf {M}} _ {d} + \tilde {\mathbf {T}} _ {d} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {a} \tilde {\mathbf {H}} \right] \mathbf {x} _ {d}.\tag{42}
$$

Finally, the new reduced mass and stiffness matrices in the proposed method are defined as 

$$
\begin{array}{l} \tilde {\mathbf {M}} _ {e} = \tilde {\mathbf {M}} _ {d} + \tilde {\mathbf {T}} _ {d} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {a} \tilde {\mathbf {H}}, \\ \tilde {\mathbf {K}} _ {e} = \tilde {\mathbf {K}} _ {d}, \end{array}\tag{43a}
$$

ð43bÞ 

in which $\tilde { \mathbf { T } } _ { d } ^ { T } \tilde { \mathbf { M T } } _ { a } \tilde { \mathbf { H } }$ is the additional mass matrix for the residual mode correction; $\tilde { \mathbf { M } } _ { e }$ and $\tilde { \mathbf { K } } _ { e }$ are the $\tilde { N } \times \tilde { N }$ matrices with the same dimension as $\tilde { \mathbf { M } } _ { d }$ and $\mathbf { K } _ { d }$ in Eq. (20). 

Using Eq. (41) for the unknown x, the transformation matrix for the proposed method $\tilde { \mathbf { T } } _ { e }$ in Eq. (31) becomes 

$$
\tilde {\mathbf {T}} _ {e} = \tilde {\mathbf {T}} _ {d} + \tilde {\mathbf {T}} _ {a} \tilde {\mathbf {H}},\tag{44}
$$

where $\tilde { \mathbf { T } } _ { e }$ is the $N \times { \tilde { N } }$ matrix. The transformation matrix provides approximate global responses from reduced responses. Through this back transformation procedure, the approximate global mode $\tilde { \varphi } _ { e }$ can be obtained as follows: 

$$
\pmb {\varphi} \approx \tilde {\pmb {\varphi}} _ {e} = \tilde {\mathbf {T}} _ {e} \mathbf {x} _ {d}.\tag{45}
$$

The computation flows of two methods, the original and new EAMLS methods, are compared in Fig. 4. 

## 3. Formulation for efficient computation

In the original enhanced AMLS (EAMLS) method [1], the transformation matrix of the AMLS method is enhanced by considering the residual mode effect. To do so, global matrix operations are required, and thus the computational efficiency deteriorates rapidly for the large FE models. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/adc82a359025c99324dd7597b5953bc06825b18c7e89a460398a8a427f9b7d0b.jpg)



(a)



(b)



Fig. 4. Flow charts for the reduction procedures: (a) the original EAMLS method and (b) the new EAMLS method



Table 1



Procedure of the new EAMLS method.


<table><tr><td>Procedure</td><td>Operation</td></tr><tr><td rowspan="2">Recursive transformation</td><td>Substructuring as Eq. (1)</td></tr><tr><td>Recursive transformation on M and K by Eq. (A.1)</td></tr><tr><td rowspan="5">Interface subspace reduction</td><td>Construction of <eq>\bar{M}_{\Gamma}</eq> and <eq>\bar{K}_{\Gamma}</eq> as shown in Eq. (12)</td></tr><tr><td>Solving <eq>\bar{K}_{\Gamma} \Xi_{d} = \bar{M}_{\Gamma} \Xi_{d} \Theta_{d}</eq> in Eq. (13)</td></tr><tr><td>Construction of <eq>\tilde{M}_{d}</eq> and <eq>\tilde{K}_{d}</eq> as shown in Eq. (48)</td></tr><tr><td>Calculation of <eq>T_{d}</eq> by Eq. (A.2)</td></tr><tr><td>Computation of <eq>\tilde{T}_{c}</eq> and <eq>\tilde{T}_{\Gamma}</eq> by Eqs. (50) and (51)</td></tr><tr><td rowspan="5">Residual mode correction</td><td>Calculation of <eq>T_{m}^{a}</eq> by Eq. (52)</td></tr><tr><td>Computation of E by Eq. (54)</td></tr><tr><td>Solving <eq>\tilde{H} = \tilde{M}_{d}^{-1} \tilde{K}_{d}</eq> in Eq. (41)</td></tr><tr><td>Calculation of <eq>\tilde{T}_{d}MT_{a} \tilde{H}</eq> as shown in Eq. (56)</td></tr><tr><td>Construction of <eq>\tilde{M}_{e}</eq> and <eq>\tilde{K}_{e}</eq> by Eq. (43)</td></tr><tr><td>Solution of the reduced system</td><td>Solving the reduced eigenvalue problem in Eq. (42)</td></tr><tr><td>Back transformation</td><td>Back transformation by Eq. (58)</td></tr></table>

To resolve this drawback, in this section, we provide the formulation of the proposed method at a submatrix level without global matrix operations. The procedure is listed in Table 1. 

In the proposed method, $\begin{array} { r } { \tilde { \mathbf { M } } _ { d } , \tilde { \mathbf { K } } _ { d } , \tilde { \mathbf { T } } _ { d } , \tilde { \mathbf { T } } _ { a } , } \end{array}$ and $\tilde { \bf H }$ in Eq. (43) are calculated to construct the reduced mass and stiffness matrices $\tilde { \mathbf { M } } _ { e }$ and $\tilde { \mathbf { K } } _ { e } .$ . 

To calculate $\tilde { \mathbf { M } } _ { d }$ and $\tilde { \mathbf { K } } _ { d } ,$ , we firstly compute the reduced mass and stiffness matrices M<sup>-</sup> and K<sup>-</sup> in Eq. (11) using submatrix computations given in Appendix. The matrices M<sup>-</sup> and K<sup>-</sup> are described as follows: 

$$
\begin{array}{l} \bar {\mathbf {M}} = \left[ \begin{array}{c c c c c} \bar {\mathbf {M}} _ {1, 1} & & & & \\ & \ddots & & & \\ & & \bar {\mathbf {M}} _ {i, i} & \bar {\mathbf {M}} _ {i, j} & \\ & \text {sym.} & & \ddots & \\ & & & & \bar {\mathbf {M}} _ {n, n} \end{array} \right], \bar {\mathbf {K}} = \left[ \begin{array}{c c c c c} \bar {\mathbf {K}} _ {1, 1} & & & & \\ & \ddots & & \\ & & \bar {\mathbf {K}} _ {i, i} & & \\ & & & \ddots & \\ & & & & \bar {\mathbf {K}} _ {n, n} \end{array} \right] \\ \text {for i = 1,2,\cdots,n, \forall j\in A_{i}.} \end{array}\tag{46}
$$

After solving the eigenvalue problem for the interface subspace in Eq. (13), the submatrix form of $\Xi _ { d }$ is represented as 

$$
\boldsymbol {\Xi} _ {d} = \left[ \begin{array}{c c c c} \left(\boldsymbol {\Xi} _ {1} ^ {d}\right) ^ {T} & \left(\boldsymbol {\Xi} _ {2} ^ {d}\right) ^ {T} & \dots & \left(\boldsymbol {\Xi} _ {n _ {h}} ^ {d}\right) ^ {T} \end{array} \right] ^ {T},\tag{47}
$$

where $\Xi _ { i } ^ { d }$ corresponds to the $( i + n _ { b } )$ Þth reduced substructure in Eq. (46), and $n _ { h }$ denotes the number of higher-level substructures. 

Using Eqs. (46) and (47), the reduced matrices $\tilde { \mathbf { M } } _ { d }$ and $\tilde { \mathbf { K } } _ { d }$ in Eq. (20) are obtained as follows: 

$$
\tilde {\mathbf {M}} _ {d} = \left[ \begin{array}{c c c c} \bar {\mathbf {I}} _ {1} & & & \tilde {\mathbf {M}} _ {1} ^ {c} \\ & \ddots & & \vdots \\ & & \bar {\mathbf {I}} _ {n _ {b}} & \tilde {\mathbf {M}} _ {n _ {b}} ^ {c} \\ \left(\tilde {\mathbf {M}} _ {1} ^ {c}\right) ^ {T} & \dots & \left(\tilde {\mathbf {M}} _ {n _ {b}} ^ {c}\right) ^ {T} & \tilde {\mathbf {I}} _ {\Gamma} \end{array} \right], \tilde {\mathbf {K}} _ {d} = \left[ \begin{array}{c c c c} \boldsymbol {\Lambda} _ {1} ^ {d} & & & \\ & \ddots & & \\ & & \boldsymbol {\Lambda} _ {n _ {b}} ^ {d} & \\ & & & \boldsymbol {\Theta} _ {d} \end{array} \right]
$$

with $\tilde { \mathbf { M } } _ { i } ^ { c } = \sum _ { j } \bar { \mathbf { M } } _ { i , j } \Xi _ { j - n _ { b } } ^ { d }$ for i <sub>¼</sub> 1; 2; <sub>- - -</sub> ; n<sub>b</sub>; 8j 2 A<sub>i</sub>: 

ð48Þ 

The dominant transformation matrix $\tilde { \mathbf { T } } _ { d }$ in Eq. (32) is calculated after computing $\mathbf { T } _ { d }$ in Eq. (10). The matrix $\mathbf { T } _ { d }$ is also computed using submatrix computations given in Appendix and represented as 

$$
\mathbf {T} _ {d} = \left[ \begin{array}{c c c c c} \mathbf {T} _ {1, 1} ^ {d} & & & & \\ & \ddots & & \\ & & \mathbf {T} _ {i, i} ^ {d} & \mathbf {T} _ {i, j} ^ {d} & \\ & & & \ddots & \\ & & & & \mathbf {T} _ {n, n} ^ {d} \end{array} \right] \text {for} i = 1, 2, \dots , n, \forall j \in A _ {i}.\tag{49}
$$

Employing N in Eq. (47) and $\mathbf { T } _ { d }$ in Eq. $( 4 9 ) , \tilde { \mathbf { T } } _ { d }$ in Eq. (32) is calculated by 

$$
\begin{array}{l} \tilde {\mathbf {T}} _ {d} = \left[ \begin{array}{c c c c} \boldsymbol {\Phi} _ {1} ^ {d} & & & \tilde {\mathbf {T}} _ {1} ^ {c} \\ & \ddots & & \vdots \\ & & \boldsymbol {\Phi} _ {n _ {b}} ^ {d} & \tilde {\mathbf {T}} _ {n _ {b}} ^ {c} \\ & & & \tilde {\mathbf {T}} _ {1} ^ {\Gamma} \\ & & & \vdots \\ & & & \tilde {\mathbf {T}} _ {n _ {h}} ^ {\Gamma} \end{array} \right] \\ \text {with} \tilde {\mathbf {T}} _ {i} ^ {c} = \sum_ {j} \mathbf {T} _ {i, j} ^ {d} \boldsymbol {\Xi} _ {j - n _ {b}} ^ {d}, \tilde {\mathbf {T}} _ {k} ^ {\Gamma} = \sum_ {l} \mathbf {T} _ {k + n _ {b}, l} ^ {d} \boldsymbol {\Xi} _ {l - n _ {b}} ^ {d} \\ \text {for i = 1,2,\cdots,n_{b},\forall j\in A_{i}}, \\ k = 1, 2, \dots , n _ {h}, \forall l \in (\{k + n _ {b} \} \cup A _ {k + n _ {b}}), \end{array}\tag{50}
$$

in which $\tilde { \mathbf { T } } _ { i } ^ { c }$ and $\tilde { \mathbf { T } } _ { k } ^ { \Gamma }$ are the submatrices of $\widetilde { \mathbf { T } } _ { c } ^ { d }$ and $\tilde { \mathbf { T } } _ { \Gamma } ^ { d }$ in Eq. (32), respectively, as follows: 

$$
\widetilde {\mathbf {T}} _ {c} ^ {d} = \left[ \left(\widetilde {\mathbf {T}} _ {1} ^ {\mathrm{c}}\right) ^ {T} \quad \left(\widetilde {\mathbf {T}} _ {2} ^ {\mathrm{c}}\right) ^ {T} \quad \dots \quad \left(\widetilde {\mathbf {T}} _ {n _ {b}} ^ {\mathrm{c}}\right) ^ {T} \right] ^ {T},   \widetilde {\mathbf {T}} _ {\Gamma} ^ {d} = \left[ \left(\widetilde {\mathbf {T}} _ {1} ^ {\Gamma}\right) ^ {T} \quad \left(\widetilde {\mathbf {T}} _ {2} ^ {\Gamma}\right) ^ {T} \quad \dots \quad \left(\widetilde {\mathbf {T}} _ {n _ {h}} ^ {\Gamma}\right) ^ {T} \right] ^ {T}.\tag{51}
$$

The additional transformation matrix $\tilde { \mathbf { T } } _ { a }$ in Eq. (36) can be obtained by calculating $\tilde { \mathbf { T } } _ { m } ^ { a }$ as follows: 

$$
\mathbf {\tilde {T}} _ {m} ^ {a} = \left[ \left(\mathbf {\tilde {T}} _ {1} ^ {m}\right) ^ {T} \quad \left(\mathbf {\tilde {T}} _ {2} ^ {m}\right) ^ {T} \dots \quad \left(\mathbf {\tilde {T}} _ {n _ {b}} ^ {m}\right) ^ {T} \right] ^ {T} = \left[ \left(\mathbf {\tilde {F}} _ {1} ^ {\sim r b} \mathbf {Q} _ {1}\right) ^ {T} \quad \left(\mathbf {\tilde {F}} _ {2} ^ {\sim r b} \mathbf {Q} _ {2}\right) ^ {T} \dots \quad \left(\mathbf {\tilde {F}} _ {n _ {b}} ^ {\sim r b} \mathbf {Q} _ {n _ {b}}\right) ^ {T} \right] ^ {T}
$$

$$
\text { with } \mathbf {Q} _ {i} = \mathbf {M} _ {i, i} \tilde {\mathbf {T}} _ {i} ^ {c} + \sum_ {j} \mathbf {M} _ {i, j} \tilde {\mathbf {T}} _ {j - n _ {b}} ^ {\Gamma} \text { for } i = 1, 2, \dots , n _ {b}, \forall j \in A _ {i},\tag{52}
$$

where $\tilde { \mathbf { F } } _ { i } ^ { r b }$ can be calculated by Eq. (29). 

Then, to compute the additional mass matrix $\widetilde { \mathbf { T } } _ { d } ^ { T } \mathbf { M } \widetilde { \mathbf { T } } _ { a } \widetilde { \mathbf { H } }$ in Eq. (43a) efficiently, $\widetilde { \mathbf { T } } _ { d } ^ { T } \mathbf { M } \widetilde { \mathbf { T } } _ { a }$ and $\tilde { \bf H }$ are calculated separately. In Eq. (39a), $\widetilde { \mathbf { T } } _ { d } ^ { T } \mathbf { M } \widetilde { \mathbf { T } } _ { a }$ can be represented by 

$$
\tilde {\mathbf {T}} _ {d} ^ {T} \mathbf {M} \tilde {\mathbf {T}} _ {a} = \left[ \begin{array}{c c} \mathbf {0} & \mathbf {0} \\ \mathbf {0} & \mathbf {E} \end{array} \right] \text {with} \mathbf {E} = \mathbf {Q} _ {b} ^ {T} \tilde {\mathbf {F}} _ {b} ^ {r b} \mathbf {Q} _ {b},\tag{53}
$$

in which E is calculated using $\mathbf { Q } _ { i }$ and $\stackrel { \sim } { \mathbf { T } } _ { i } ^ { m }$ in Eq. (52) as follows: 

$$
\mathbf {E} = \mathbf {Q} _ {b} ^ {T} \mathbf {\tilde {F}} _ {b} ^ {r b} \mathbf {Q} _ {b} = \sum_ {i = 1} ^ {n _ {b}} \mathbf {Q} _ {i} ^ {T} \mathbf {\tilde {T}} _ {i} ^ {m}.\tag{54}
$$

The non-symmetric matrix H computed by Eq. (41) can be partitioned into the bottom-level substructural and interface quantities as 

$$
\tilde {\mathbf {H}} = \left[ \begin{array}{c c} \tilde {\mathbf {H}} _ {b} & \tilde {\mathbf {H}} _ {b, \Gamma} \\ \tilde {\mathbf {H}} _ {\Gamma , b} & \tilde {\mathbf {H}} _ {\Gamma} \end{array} \right].\tag{55}
$$

After calculating E in Eq. (54) and $\tilde { \bf H }$ in Eq. (55), $\tilde { \mathbf { T } } _ { d } ^ { T } \tilde { \mathbf { M } } \tilde { \mathbf { T } } _ { a } \tilde { \mathbf { H } }$ in Eq. (43a) is computed by 

$$
\mathbf {\tilde {T}} _ {d} ^ {T} \mathbf {M} \mathbf {\tilde {T}} _ {a} \mathbf {\tilde {H}} = \left[ \begin{array}{c c} \mathbf {0} & \mathbf {0} \\ \mathbf {0} & \mathbf {E} \end{array} \right] \left[ \begin{array}{c c} \mathbf {\tilde {H}} _ {b} & \mathbf {\tilde {H}} _ {b, \Gamma} \\ \mathbf {\tilde {H}} _ {\Gamma , b} & \mathbf {\tilde {H}} _ {\Gamma} \end{array} \right] = \left[ \begin{array}{c c} \mathbf {0} & \mathbf {0} \\ \mathbf {\tilde {E H}} _ {\Gamma , b} & \mathbf {\tilde {E H}} _ {\Gamma} \end{array} \right].\tag{56}
$$

Finally, by summing $\tilde { \mathbf { M } } _ { d }$ in Eq. (48) and $\tilde { \mathbf { T } } _ { d } ^ { T } \tilde { \mathbf { M } } \tilde { \mathbf { T } } _ { c }$ H in Eq. (56), the reduced mass matrix for the proposed method M can be efficiently computed. The reduced stiffness matrix for the proposed method $\tilde { \mathbf { K } } _ { e }$ is equal to $\tilde { \mathbf { K } } _ { d }$ in $\operatorname { E q . }$ (48). 

For the back transformation to obtain the approximate global mode from the reduced mode, $\tilde { \varphi } _ { \epsilon }$ and $\mathbf { x } _ { d }$ in Eq. (45) are represented as 

$$
\tilde {\widetilde {\boldsymbol {\varphi}}} _ {e} = \left[ \begin{array}{c} \tilde {\widetilde {\boldsymbol {\varphi}}} _ {b} ^ {e} \\ \tilde {\widetilde {\boldsymbol {\varphi}}} _ {\Gamma} ^ {e} \end{array} \right], \mathbf {x} _ {d} = \left[ \begin{array}{c} \mathbf {x} _ {b} ^ {d} \\ \mathbf {x} _ {\Gamma} ^ {d} \end{array} \right]
$$

with 

$$
\begin{array}{l} \widetilde {\boldsymbol {\varphi}} _ {b} ^ {e} = \left[ \left(\widetilde {\boldsymbol {\varphi}} _ {1} ^ {e}\right) ^ {T} \quad \left(\widetilde {\boldsymbol {\varphi}} _ {2} ^ {e}\right) ^ {T} \quad \dots \quad \left(\widetilde {\boldsymbol {\varphi}} _ {n _ {b}} ^ {e}\right) ^ {T} \right] ^ {T}, \\ \mathbf {x} _ {b} ^ {d} = \left[ \left(\mathbf {x} _ {1} ^ {d}\right) ^ {T} \quad \left(\mathbf {x} _ {2} ^ {d}\right) ^ {T} \quad \dots \quad \left(\mathbf {x} _ {n _ {b}} ^ {d}\right) ^ {T} \right] ^ {T}, \end{array}\tag{57}
$$

where $\widetilde { \pmb { \varphi } } _ { b } ^ { e }$ and $\tilde { \varphi } _ { \Gamma } ^ { e }$ denote the approximate global modes corresponding to the bottom-level substructures and interface, respectively. The vectors $\mathbf { x } _ { b } ^ { d }$ and $\mathbf { x } _ { \Gamma } ^ { d }$ denote the reduced modes corresponding to the bottom-level substructures and the interface, respectively. The substructural components $\tilde { \varphi } _ { i } ^ { e }$ and $\mathbf { x } _ { i } ^ { d }$ denote the approximate global and reduced modes corresponding to the ith substructure, respectively. 

Using Eqs. (44) and (57) in Eq. (45), the approximate global mode $\tilde { \varphi } _ { e }$ is calculated by 

$$
\begin{array}{l} \widetilde {\boldsymbol {\varphi}} _ {i} ^ {e} = \boldsymbol {\Phi} _ {i} ^ {d} \mathbf {x} _ {i} ^ {d} + \widetilde {\mathbf {T}} _ {i} ^ {c} \mathbf {x} _ {\Gamma} ^ {d} + \widetilde {\mathbf {T}} _ {i} ^ {m} \mathbf {y} \\ \text {with} \mathbf {y} = \widetilde {\mathbf {H}} _ {\Gamma , b} \mathbf {x} _ {b} ^ {d} + \widetilde {\mathbf {H}} _ {\Gamma} \mathbf {x} _ {\Gamma} ^ {d} \text {for} i = 1, 2, \dots , n _ {b}, \end{array}\tag{58a}
$$

$$
\tilde {\boldsymbol {\varphi}} _ {\Gamma} ^ {e} = \tilde {\mathbf {T}} _ {\Gamma} ^ {d} \mathbf {x} _ {\Gamma} ^ {d}.\tag{58b}
$$

There is no global matrix operation during the reduction procedure, including the back transformation. The computational efficiency of the original EAMLS method is greatly improved through the procedures presented in Sections 2 and 3. 

## 4. Numerical examples

In order to investigate the performance of the new enhanced AMLS (EAMLS) method, we consider several numerical examples: a centrifugal impeller, a femur, and an airplane. Table 2 lists the number of DOFs and nonzero entries in the global FE matrices for each example. 

The performance of the new EAMLS method is compared to that of the AMLS and original EAMLS methods. To measure the solution accuracy, the following relative error in the natural frequency is used: 

$$
e _ {i} = \frac {\left| \bar {\omega} _ {i} - \omega_ {i} \right|}{\omega_ {i}},\tag{59}
$$

where $e _ { i }$ is the relative error in the natural frequency for the ith mode, and $\omega _ { i }$ and $\omega _ { i }$ are the ith natural frequency obtained by the global FE model and the ith approximate natural frequency obtained by the reduced model, respectively. Here, the rigid body modes are not used when evaluating the solution accuracy. The efficiency is evaluated by the computation time and required computer memory. Note that the computation time is the elapsed time for the entire reduction procedure, including the back transformation. 

The global FE matrices are reordered and automatically partitioned into several submatrices using METIS [26], open source software for unstructured graph partitioning. The frequency cutoff method is used to select the dominant normal modes as in Section 2.2. All numerical computations are performed using MATLAB 2016b on a personal computer (PC) with Intel i7 7700 3.60 GHz and 32 GB of memory. 

## 4.1. Centrifugal impeller

We consider a centrifugal impeller, a key component of a centrifugal compressor, as shown in Fig. 5. The impeller is modeled using 66,201 four-node tetrahedral elements (52,332 DOFs), and the FE model is partitioned into 236 substructures. The same size of the reduced models is considered to compare the AMLS, EAMLS, and new EAMLS methods. The number of approximate global modes with natural frequencies sought in this example is 60. 

For the AMLS and original EAMLS methods, 1519 dominant substructural normal modes for the substructures are retained. In other words, the size of the reduced models obtained by the two methods, N, is 1519. For the new EAMLS method, the numbers of dominant normal modes for the bottom-level substructures and interface subspace are 1130 and 389, respectively $( \bar { N } _ { b } = 1 1 3 0$ and $\stackrel { \sim } { N } _ { \Gamma } = 3 8 9 )$ . Thus, the size of the reduced model, ${ \tilde { N } } ,$ is 1519 $( \stackrel { \sim } { N } = \bar { N } _ { b } + \stackrel { \sim } { N } _ { \Gamma } )$ 

In this example, the solution accuracy is measured by the relative error in the natural frequencies in $\operatorname { E q . }$ (59). Also we consider the relative error in the modes defined by the angle between the global mode obtained by the global FE model and approximate global mode obtained by the reduced model as follows: 


Table 2



Finite element models for each example.


<table><tr><td rowspan="2">Example</td><td rowspan="2">DOFs</td><td colspan="2">Number of nonzero entries</td></tr><tr><td>Mass matrix</td><td>Stiffness matrix</td></tr><tr><td>Centrifugal impeller</td><td>52,332</td><td>626,844</td><td>1,879,836</td></tr><tr><td>Femur</td><td>610,260</td><td>7,992,858</td><td>23,978,034</td></tr><tr><td>Airplane</td><td>2,025,180</td><td>28,224,198</td><td>84,666,422</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/eb3bb6bdd044378cf13203175b4d4d95aba9f4e4dd3aa6fdc11f0e4df2f7baec.jpg)



Fig. 5. Centrifugal impeller (66,201 four-node tetrahedral elements and 52,332 DOFs).


$$
\varepsilon_ {i} = \arccos \frac {\left| \boldsymbol {\varphi} _ {i} \cdot \bar {\boldsymbol {\varphi}} _ {i} \right|}{\left\| \boldsymbol {\varphi} _ {i} \right\| _ {2} \| \bar {\boldsymbol {\varphi}} _ {i} \| _ {2}},\tag{60}
$$

where $\mathbf { \delta } _ { E i } , \mathbf { \delta } _ { \varphi _ { i } }$ , and $\bar { \varphi } _ { i }$ are the ith relative error in the mode, ith global mode, and ith approximate global mode, respectively. 

Figs. 6 and 7 show the relative errors in the natural frequencies and modes, respectively. The accuracy of the new EAMLS method is comparable to that of the original EAMLS method. In the low frequency range, the original EAMLS method represents more accurate solutions than the new method because it uses the residual flexibility matrices for all substructures, except for the highestlevel substructure. On the other hand, the new EAMLS method provides the consistent solution accuracy in the frequency range of interest due to the use of more substructural normal modes and the residual flexibility matrices for the bottom-level substructures. 

The computation times are shown in Fig. 8, and their details are listed in Table 3. In the original EAMLS method, the calculation of the multilevel constraint mode matrix and EAMLS transformation matrix induces a large computation time. Consequently, the original EAMLS method requires 374.94 s for the entire reduction procedure, while the new EAMLS method requires only 61.57 s. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/d5a9d8737e60d0e7a91f40996313634e7d84648063b343e62e373e40608cb245.jpg)



Fig. 6. Relative errors in the natural frequencies for the centrifugal impeller: $\tilde { N } = \tilde { N } = 1 5 1 9 .$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/848584a6fddc35569807cab039bd07bb591f2134c64af4068caa1b9e617d866f.jpg)



Fig. 7. Relative errors in the modes for the centrifugal impeller: $\bar { N } = \tilde { N } = 1 5 1 9$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/972e67e7a454848375fe5d85e2faf62b58e664c3ce9b017e1e37723e8a6673e6.jpg)



Fig. 8. Computation times for the centrifugal impeller: $\begin{array} { r } { \bar { N } = \tilde { N } = 1 5 1 9 . } \end{array}$


In Table 4, we list the densities of the reduced mass and stiffness matrices defined by a fraction of the number of nonzero entries to the total number of entries in the matrix. In the original EAMLS method, the densities of the reduced mass and stiffness matrices are 100%, while the densities for the proposed method are 44.71% and 0.07%, respectively. 

The computational efficiency is remarkably improved through the new EAMLS method. 

## 4.2. Femur

Here, a femur is considered, as shown in Fig. 9. The femur is anisotropic; however, it is assumed to be an isotropic material [41]. For modeling the femur, 936,887 four-node tetrahedral elements are used, and the number of DOFs is 610,260. The FE model is partitioned into 1971 substructures. The number of approximate global modes with natural frequencies sought in this example is 100. 

As mentioned in Section 1, calculating the multilevel constraint mode matrix in the original EAMLS method requires large computer memory (over 32 GB) when 1971 substructures are used. Thus, the reduced model cannot be constructed using the original EAMLS method because there is not enough memory in the PC. In this example, reduced models are obtained using the AMLS and new EAMLS methods. The size of the reduced models is 4264 $( \stackrel { - } { N } = \stackrel { \sim } { N } = 4 2 6 4 )$ . The maximum memory allocated during the reduction procedures is listed in Table 5. 


Specific computation times for the centrifugal impeller. The ratios in this table are defined as follows: (ratio) (computation time) (total computation time of the AMLS method) 100.


<table><tr><td rowspan="2">Method</td><td rowspan="2">Procedure</td><td colspan="2">Computation time</td></tr><tr><td>(s)</td><td>Ratio (%)</td></tr><tr><td rowspan="5">AMLS</td><td>Recursive transformation</td><td>48.74</td><td>81.78</td></tr><tr><td>Reduced eigenvalue problem</td><td>0.85</td><td>1.42</td></tr><tr><td>Calculation of the AMLS transformation matrix</td><td>9.75</td><td>16.36</td></tr><tr><td>Back transformation</td><td>0.26</td><td>0.44</td></tr><tr><td>Total</td><td>59.60</td><td>100.00</td></tr><tr><td rowspan="7">EAMLS</td><td>Recursive transformation</td><td>50.30</td><td>84.39</td></tr><tr><td>Calculation of the multilevel constraint mode matrix</td><td>64.33</td><td>107.94</td></tr><tr><td>Calculation of the EAMLS transformation matrix</td><td>255.23</td><td>428.24</td></tr><tr><td>Construction of the reduced model</td><td>3.84</td><td>6.44</td></tr><tr><td>Reduced eigenvalue problem</td><td>1.00</td><td>1.68</td></tr><tr><td>Back transformation</td><td>0.24</td><td>0.40</td></tr><tr><td>Total</td><td>374.94</td><td>629.09</td></tr><tr><td rowspan="6">New EAMLS</td><td>Recursive transformation</td><td>49.46</td><td>82.99</td></tr><tr><td>Interface subspace reduction</td><td>7.91</td><td>13.27</td></tr><tr><td>Residual mode correction</td><td>3.53</td><td>5.92</td></tr><tr><td>Reduced eigenvalue problem</td><td>0.51</td><td>0.86</td></tr><tr><td>Back transformation</td><td>0.16</td><td>0.27</td></tr><tr><td>Total</td><td>61.57</td><td>103.31</td></tr></table>


Reduced matrices information for the centrifugal impeller.


<table><tr><td rowspan="2">Method</td><td rowspan="2">Reduced DOFs</td><td colspan="2">Density of the matrix (%)</td></tr><tr><td>Reduced mass matrix</td><td>Reduced stiffness matrix</td></tr><tr><td>AMLS</td><td>1519</td><td>29.17</td><td>0.07</td></tr><tr><td>EAMLS</td><td>1519</td><td>100.00</td><td>100.00</td></tr><tr><td>New EAMLS</td><td>1519</td><td>44.71</td><td>0.07</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/3745fb67333d9f57d427a2c615f64ef83ac34cc58c4ed305ed8855cac0023fc8.jpg)



Fig. 9. Femur (936,887 four-node tetrahedral elements and 610,260 DOFs).


The solution accuracy of the new EAMLS method is compared to that of the AMLS method using the relative error in the natural frequency in Eq. (59) and the modal assurance criterion (MAC). The MAC is defined a 

$$
\operatorname{MAC} (i, j) = \frac {\left| \boldsymbol {\varphi} _ {i} ^ {T} \bar {\boldsymbol {\varphi}} _ {j} \right| ^ {2}}{\left(\boldsymbol {\varphi} _ {i} ^ {T} \boldsymbol {\varphi} _ {i}\right) \left(\bar {\boldsymbol {\varphi}} _ {j} ^ {T} \bar {\boldsymbol {\varphi}} _ {j}\right)}.\tag{61}
$$


Table 5



Memory usage histories for the femur.


<table><tr><td>Method</td><td>Procedure</td><td>Maximum memory allocated (GB)</td></tr><tr><td rowspan="5">AMLS</td><td>Recursive transformation</td><td>12.77</td></tr><tr><td>Reduced eigenvalue problem</td><td>8.68</td></tr><tr><td>Calculation of the transformation matrix</td><td>13.15</td></tr><tr><td>Back transformation</td><td>6.84</td></tr><tr><td>Maximum value</td><td>13.15</td></tr><tr><td rowspan="6">EAMLS</td><td>Recursive transformation</td><td>15.58</td></tr><tr><td>Calculation of the multilevel constraint mode matrix</td><td>Out of memory</td></tr><tr><td>Calculation of the EAMLS transformation matrix</td><td>N/A</td></tr><tr><td>Construction of the reduced model</td><td>N/A</td></tr><tr><td>Reduced eigenvalue problem</td><td>N/A</td></tr><tr><td>Back transformation</td><td>N/A</td></tr><tr><td rowspan="6">New EAMLS</td><td>Recursive transformation</td><td>13.74</td></tr><tr><td>Interface subspace reduction</td><td>14.85</td></tr><tr><td>Residual mode correction</td><td>3.59</td></tr><tr><td>Reduced eigenvalue problem</td><td>3.78</td></tr><tr><td>Back transformation</td><td>3.46</td></tr><tr><td>Maximum value</td><td>14.85</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/326f489ca97e2fd67388e0c075909e87cbb486ea351c84b947d472f10184f61a.jpg)



Fig. 10. Relative errors in the natural frequencies for the femur: $\stackrel { \triangledown } { \bar { N } } = \stackrel { \sim } { N } = 4 2 6 4 .$


Fig. 10 shows the relative errors in the natural frequencies, and Fig. 11 shows the MACs obtained by the AMLS and new EAMLS methods. The computation times are listed in Table 6. The results show that the new EAMLS method provides excellent solution accuracy with minimal additional computational cost compared to that of the AMLS method, while the original EAMLS method fails to reduce the global FE model. 

## 4.3. Airplane

In order to verify the performance of the new EAMLS method for a large FE model with more than 2 million DOFs, we consider an airplane modeled by three-node shell elements [40,42–45] (see Fig. 12). The numbers of elements and DOFs used in the global FE model are 675,056 and 2,025,180, respectively, and the FE model is partitioned into 2046 substructures. As in Section 4.2, the reduced model cannot be constructed using the original EAMLS method. The number of approximate natural frequencies sought in this example is 400. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/6ea3814e9b921fa37953d143ca8fbf79d8a96dffbd1acc68e3cca0648c5716a6.jpg)


In this example, the reduced model obtained by the new EAMLS method $( \tilde { N } = 5 4 9 2$ with $\bar { N } _ { b } = 4 7 2 7$ and $\tilde { N } _ { \Gamma } = 7 6 5 )$ is compared to the reduced models with the same size $( \bar { N } = 5 4 9 2 )$ and similar accuracy $( N = 2 3 , 8 7 2 )$ obtained by the AMLS method. 

Fig. 13 shows the relative errors in the natural frequencies obtained by the three different reduced models $( { \bar { N } } = 5 4 9 2 ,$ $N = 2 3 , 8 7 2$ , and $\stackrel { \sim } { N } = 5 4 9 2 )$ . The reduced model obtained by the new EAMLS method $( \stackrel { \sim } { N } = 5 4 9 2 )$ is four times smaller than that obtained by the AMLS method $( \bar { N } = 2 3 , 8 7 2 ) ;$ however, its solution accuracy is similar to that of the AMLS method. 


Table 6



Computation times for the femur. The ratios in this table are defined as follows: (ratio) (computation time)  (total computation time of the AMLS method) <sub></sub> 100


<table><tr><td rowspan="2">Method</td><td colspan="2">Computation time</td></tr><tr><td>(s)</td><td>Ratio (%)</td></tr><tr><td>AMLS</td><td>3317.03</td><td>100.00</td></tr><tr><td>EAMLS</td><td>N/A</td><td>-</td></tr><tr><td>New EAMLS</td><td>3603.93</td><td>108.65</td></tr></table>


(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/75c893ada86f6afd83ef9b16cdf6155599b7ff7f9c9eaa4469bfb233d8b12d14.jpg)



(b)



Fig. 11. MACs for the femur: (a) the AMLS method and (b) the new EAMLS method: $\bar { N } = \stackrel { \sim } { N } = 4 2 6 4 .$



Table 7


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/df08e1feddeef294263a4c40e3fdaeed54d2fb7f0349f29b64b81e376da97d6f.jpg)



Fig. 12. Airplane (675,056 three-node shell elements and 2,025,180 DOFs).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/958db3f3-2eb1-430f-8469-22715c284930/5daf743ac7bad0c774070f99d1bdf1e8df1b5b5fb5ea0191d50411b13422236c.jpg)



Fig. 13. Relative errors in the natural frequencies for the airplane: N<sup>-</sup> <sub>¼</sub> 5492; N<sup>-</sup> <sub>¼</sub> 23; 872; and N<sup></sup> <sub>¼</sub> 5492. Markers are placed at 20-natural-frequency intervals.


Tables 7 and 8 list the computation times and densities of the reduced matrices, respectively. These results show that, for the same DOFs as that of the AMLS method (N 5492), the new EAMLS method constructs a more accurate reduced model with an additional cost of 2.16%. In addition, the reduced model obtained by the new EAMLS method has less number of nonzero entries. For similar accuracy to that of the AMLS method $( \bar { N } = 2 3 , 8 7 2 )$ , the new EAMLS method constructs a smaller reduced model with less computation time. 


Computation times for the airplane. The ratios in this table are defined as follows: (ratio) (computation time)  (total computation time of the AMLS method) 100.


<table><tr><td rowspan="2">Method</td><td colspan="2">Computation time</td></tr><tr><td>(s)</td><td>Ratio (%)</td></tr><tr><td>AMLS (<eq>\bar{N} = 5492</eq>)</td><td>5177.36</td><td>100.00</td></tr><tr><td>AMLS (<eq>\bar{N} = 23,872</eq>)</td><td>6104.18</td><td>117.90</td></tr><tr><td>EAMLS</td><td>N/A</td><td>-</td></tr><tr><td>New EAMLS (<eq>\tilde{N} = 5492</eq>)</td><td>5289.42</td><td>102.16</td></tr></table>


Table 8



Reduced matrices information for the airplane.


<table><tr><td rowspan="2">Method</td><td rowspan="2">Reduced DOFs</td><td colspan="2">Density of the matrix (%)</td></tr><tr><td>Reduced mass matrix</td><td>Reduced stiffness matrix</td></tr><tr><td>AMLS</td><td>5492</td><td>48.140</td><td>0.018</td></tr><tr><td>AMLS</td><td>23,872</td><td>18.610</td><td>0.004</td></tr><tr><td>EAMLS</td><td>N/A</td><td>-</td><td>-</td></tr><tr><td>New EAMLS</td><td>5492</td><td>25.934</td><td>0.018</td></tr></table>

## 5. Conclusions

In this study, we presented an algorithm to improve the computational efficiency of the enhanced AMLS (EAMLS) method. With minimal loss of accuracy, we effectively reduced the matrix operation terms that cause a slowdown and require excessive computer memory in the original EAMLS method. An interface subspace reduction was employed to construct a compact reduced model. In addition, a residual mode correction was derived by the residual flexibility matrices for only the bottom-level substructures, and it was applied to only the mass matrix. As a result, the calculation of the multilevel constraint mode matrix, one of the bottlenecks in the original EAMLS method, was not performed explicitly. The formulation of the new EAMLS method was presented at a submatrix level for an efficient computation. 

Through numerical examples, we demonstrated that the new EAMLS method significantly reduced the memory requirements and computation time of the original EAMLS method with similar accuracy. Furthermore, with a similar accuracy to that of the AMLS method, the efficiency of the new EAMLS method was greater than that of the AMLS method to reduce a large FE model, which could not be handled by the original EAMLS method. 

Future efforts to develop parallel implementation for large FE models that are difficult to handle on a single processor would be valuable. The new EAMLS method could also be extended to an alternative to classical eigenvalue solvers, such as the subspace iteration [40,46] and Lanczos methods [47,48]. 

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) 

(No. NRF-2018R1A2B3005328). This work was also supported by ‘‘Human Resources Program in Energy Technology” of the Korea Institute of Energy Technology Evaluation and Planning (KETEP), granted financial resource from the Ministry of Trade, Industry & Energy, Republic of Korea (No. 20184030202000). 

## Appendix. Submatrix formulation for the recursive transformation

We provide the submatrix formulation to compute the mass, stiffness, and transformation matrices (M<sup>-</sup> , K<sup>-</sup>, and T ) in Eqs. (10) and (10) and (11). The detailed derivation is well described in Refs. [22,23]. 

The substructural components of $\bar { \mathbf { M } } ^ { ( i ) }$ and $\bar { \mathbf { K } } ^ { ( i ) } \left( \mathrm { f o r } i = 1 , 2 , \cdots , n \right)$ in Eq. (2) are computed to construct the matrices M<sup>-</sup> and K<sup>-</sup> in Eq. (11) as follows: 

$$
\mathbf {K} _ {i, i} ^ {(i - 1)} \boldsymbol {\Phi} _ {i} ^ {d} = \mathbf {M} _ {i, i} ^ {(i - 1)} \boldsymbol {\Phi} _ {i} ^ {d} \boldsymbol {\Lambda} _ {i} ^ {d},\tag{A.1a}
$$

$$
\boldsymbol {\Psi} _ {i, j} = - \left(\mathbf {K} _ {i, i} ^ {(i - 1)}\right) ^ {- 1} \mathbf {K} _ {i, j} ^ {(i - 1)} \text {   for   } \forall j \in A _ {i},\tag{A.1b}
$$

$$
\mathbf {K} _ {j, l} ^ {(i)} = \mathbf {K} _ {j, l} ^ {(i - 1)} + \boldsymbol {\Psi} _ {i, j} ^ {T} \mathbf {K} _ {i, l} ^ {(i - 1)} \text {   for   } \forall j, l \in A _ {i},\tag{A.1c}
$$

$$
\mathbf {M} _ {j, l} ^ {(i)} = \mathbf {M} _ {j, l} ^ {(i - 1)} + \boldsymbol {\Psi} _ {i, j} ^ {T} \mathbf {M} _ {i, l} ^ {(i - 1)} + \left(\mathbf {M} _ {i, j} ^ {(i - 1)}\right) ^ {T} \boldsymbol {\Psi} _ {i, l} + \boldsymbol {\Psi} _ {i, j} ^ {T} \mathbf {M} _ {i, i} ^ {(i - 1)} \boldsymbol {\Psi} _ {i, l} \text {   for   } \forall j, l \in A _ {i},\tag{A.1d}
$$

$$
\mathbf {M} _ {i, j} ^ {(i)} = \left(\boldsymbol {\Phi} _ {i} ^ {d}\right) ^ {T} \left(\mathbf {M} _ {i, i} ^ {(i - 1)} \boldsymbol {\Psi} _ {i, j} + \mathbf {M} _ {i, j} ^ {(i - 1)}\right) \text {   for   } \forall j, l \in A _ {i},\tag{A.1e}
$$

$$
\mathbf {M} _ {j, l} ^ {(i)} = \mathbf {M} _ {j, l} ^ {(i - 1)} + \mathbf {M} _ {j, i} ^ {(i - 1)} \boldsymbol {\Psi} _ {i, l} \text {   for   } \forall j \in D _ {i}, \forall l \in A _ {i},\tag{A.1f}
$$

$$
\bar {\mathbf {M}} _ {j, i} = \mathbf {M} _ {j, i} ^ {(i - 1)} \boldsymbol {\Phi} _ {i} ^ {d} \text {   for   } \forall j \in D _ {i},\tag{A.1g}
$$

$$
\bar {\mathbf {K}} _ {i, i} = \left(\boldsymbol {\Phi} _ {i} ^ {d}\right) ^ {T} \mathbf {K} _ {i, i} \boldsymbol {\Phi} _ {i} ^ {d} = \boldsymbol {\Lambda} _ {i} ^ {d},\tag{A.1h}
$$

$$
\bar {\mathbf {M}} _ {i, i} = \left(\boldsymbol {\Phi} _ {i} ^ {d}\right) ^ {T} \mathbf {M} _ {i, i} \boldsymbol {\Phi} _ {i} ^ {d} = \bar {\mathbf {I}} _ {i},\tag{A.1i}
$$

$$
\bar {\mathbf {K}} _ {i, j} = \mathbf {0} \text {   for   } \forall j \in A _ {i},\tag{A.1j}
$$

where the superscripts (i) and (i 1) denote matrices in the ith and (i 1)th transformations, respectively, and the mass and stiffness matrices with subscripts denote the substructural components. The matrix $\bar { \mathbf { I } } _ { i }$ in Eq. (A.1i) is the identity matrix corresponding to the ith reduced substructure. Note that, in the ith transformation, if the ancestor set $A _ { i }$ (or descendant set D ) is an empty set, the corresponding equations are not computed. 

The substructural components of $\mathbf { T } _ { d }$ in Eq. (10) are computed as follows: 

$$
\mathbf {T} _ {i, i} ^ {d} = \boldsymbol {\Phi} _ {i} ^ {d} \text {   for   } i = 1, 2, \dots , n\tag{A.2a}
$$

$$
\mathbf {T} _ {i, j} ^ {d} = \boldsymbol {\Psi} _ {i, j} \mathbf {T} _ {j, j} ^ {d} + \sum_ {k} \boldsymbol {\Psi} _ {i, k} \mathbf {T} _ {k, j} ^ {d} \text {   for   } \forall i \in D _ {j}, j = 1, 2, \dots , n, \forall k \in (A _ {i} \cap D _ {j}).
$$

Similarly, if the ancestor set $A _ { i }$ (or descendant set $D _ { j } )$ is an empty set, the corresponding equations are not computed. 

## References



[1] Kim JG, Boo SH, Lee PS. An enhanced AMLS method and its performance. Comput Methods Appl Mech Eng 2015:287:90–111. https://doi org/10.1016/i cma.2015.01.004 





[2] Hurty WC. Dynamic analysis of structural systems using component modes. AIAA J 1965;3:678–85. https://doi.org/10.2514/3.2947. 





[3] Craig RR, Bampton MCC. Coupling of substructures for dynamics analyses. AIAA I1968:6:1313. https://doi.org/10.2514/3.4741 





[4] Hintz RM. Analytical methods in component modal synthesis. AIAA J 1975:13:1007-16. https://doi.org/10.2514/3.60498 





[5] Park KC, Park YH. Partitioned component mode synthesis via a flexibility approach. AIAA J 2004;42:1236–45. https://doi.org/10.2514/1.10423 





[6] Rixen DJ. A dual Craig-Bampton method for dynamic substructuring. J Comput Appl Math 2004;168:383–91. https://doi.org/10.1016/j.cam.2003.12.014. 





[7] Kim JG, Lee PS. An enhanced Craig-Bampton method. Int J Numer Methods Eng 2015;103:79–93. https://doi.org/10.1002/nme.4880 





[8] Boo SH, Kim JH, Lee PS. Towards improving the enhanced Craig-Bampton method. Comput Struct 2018;196:63–75. https://doi.org/10.1016/ j.compstruc.2017.10.017. 





[9] Gibanica M, Abrahamsson TJS, Rixen DJ. A reduced interface component mode synthesis method using coarse meshes. Proc Eng 2017;199:348–53. https:// doi.org/10.1016/j.proeng.2017.09.031. 





[10] Battiato G, Firrone CM, Berruti TM, Epureanu BI. Reduction and coupling of substructures via Gram-Schmidt Interface modes. Comput Methods Appl Mech Eng 2018;336:187–212. https://doi.org/10.1016/j. cma.2018.03.001. 





[11] Wu L, Tiso P, Van Keulen F. Interface reduction with multilevel Craig-Bampton substructuring for component mode synthesis. AIAA J 2018;56:1–15. https:// doi.org/10.2514/1.j056196. 





[12] Cui J, Wang X, Xing J, Zheng G. An eigenvector-based iterative procedure for the free-interface component modal synthesis method. Int J Numer Methods Eng 2018:116:723–40. https://doi.org/10.1002/nme,5941 





[13] Tian W, Weng S, Xia Y, Zhu H, Gao F, Sun Y, et al. An iterative reduced-order substructuring approach to the calculation of eigensolutions and eigensensitivities. Mech Syst Signal Process 2019;130:361–77. https://doi. org/10.1016/LYMSSP.2019.05.006 





[14] Herrada FJ, García-Martínez J, Fraile A, Hermanns LKH, Montáns FJ. A method for performing efficient parametric dynamic analyses in large finite element models undergoing structural modifications. Eng Struct 2017;131:625–38. https://doi.org/10.1016/i.engstruct.2016.10.026. 





[15] García-Martínez J, Herrada FJ, Hermanns LKH, Fraile A, Montáns FJ. Accelerating parametric studies in computational dynamics: Selective modal re-orthogonalization versus model order reduction methods. Adv Eng Softw 2017:108:24-36. https://doi,org/10.1016/i,advengsoft,2017.02.006. 





[16] Lee J, Cho M. An interpolation-based parametric reduced order model combined with component mode synthesis. Comput Methods Appl Mech Eng 2017;319:258–86. https://doi.org/10.1016/j.cma.2017.02.010. 





[17] Voormeeren SN, De Klerk D, Rixen DJ. Uncertainty quantification in experimental frequency based substructuring. Mech Syst Signal Process 2010;24:106–18. https://doi.org/10.1016/j.ymssp.2009.01.016. 





[18] Rohe DP, Allen MS. Investigation of the effectiveness of using an experiment to validate experimental substructure models. Mech Syst Signal Proces 2014;43:192–216. https://doi.org/10.1016/j.ymssp.2013.08.026. 





[19] Weng S, Zhu HP, Xia Y, Mao L. Damage detection using the eigenparameter decomposition of substructural flexibility matrix. Mech Syst Signal Process 2013;34:19–38. https://doi.org/10.1016/j.ymssp.2012.08.001. 





[20] Kim J, Kim JG, Yun G, Lee PS, Kim DN. Toward modular analysis of supramolecular protein assemblies. J Chem Theory Comput 2015;11:4260–72. https://doi.org/10.1021/acs.jctc.5b00329. 





[21] Bennighof JK, Kim C. An adaptive multi-level substructuring method for efficient modeling of complex structures. 33rd Struct. Struct. Dyn. Mater. Conf., American Institute of Aeronautics and Astronautics; 1992. doi: 10.2514/ 6.1992-2327 





[22] Kaplan MF. Implementation of automated multi-level substructuring for frequency response analysis of structures [dissertation]. Austin (TX): University of Texas at Austin; 2001. 





[23] BennighofJK, Lehoucq RB. An automated multilevel substructuring method for eigenspace computation in linear elastodynamics. SIAM J Sci Comput 2004:25:2084. https://doiorg/10.1137/S1064827502400650 





[24] Kim M. An efficient eigensolution method and its implementation for large structural systems [dissertationl. Austin (TX): University of Texas at Austin: 2004. 





[25] George A. Nested dissection of a regular finite element mesh. SIAM J Numer Anal 1973:10:345-63. https://doi.org/10.1137/0710032 





[26] Karypis G, Kumar V. A fast and high quality multilevel scheme for partitioning irregular graphs. SIAM LSci Comput 1998:20:359–92. https://doi org/10.1137/ S1064827595287997 





[27] Yang C, Gao W, Bai Z, Li XS, Lee LQ, Husbands P, et al. An algebraic substructuring method for large-Scale eigenvalue calculation. SIAM J Sci Comput 2005;27:873–92. https://doi.org/10.1137/040613767 





[28] Boo SH. Lee PS. A dynamic condensation method using algebraic substructuring. Int J Numer Methods Eng 2017;109:1701–20. https://doi.org/ 10.1002/nme.5349. 





[29] Boo SH, Lee PS. An iterative algebraic dynamic condensation method and its performance. Comput Struct 2017;182:419–29. https://doi.org/10.1016/ j.compstruc.2016.12.011 





[30] Boo SH, Oh MH. Automated static condensation method for local analysis of large finite element models. Struct Eng Mech 2017;61:807–16. https://doi.org/ 10.12989/sem 2017.61.6.807 





[31] Kropp A, Heiserer D. Efficient broadband vibro-acoustic analysis of passenger car bodies using an FE-based component mode synthesis approach. J Comput Acoust 2003;11:139–57. https://doi.org/10.1142/S0218396X03001870. 





[32] Yang C, Gao W, Bai Z, Li XS, Lee LQ, Husbands P, et al. Algebraic sub-structuring for electromagnetic applications. Springer; 2006. p. 364–73. https://doi.org/ 10.1007/11558958_43 





[33] Rachowicz W, Zdunek A. Automated multi-level substructuring (AMLS) for electromagnetics. Comput Methods Appl Mech Eng 2009;198:1224–34. https://doi.org/10.1016/j.cma.2008.10.011. 





[34] Song Q, Chen P, Sun S. An exact reanalysis algorithm for local non-topological high-rank structural modifications in finite element analysis. Comput Struct 2014;143:60–72. https://doi.org/10.1016/j.compstruc.2014.07.014. 





[35] Bekas C, Saad Y. Computation of smallest eigenvalues using spectral Schur complements. SIAM J Sci Comput 2005;27:458–81. https://doi.org/10.1137/ 040603528. 





[36] Baek SM. Study on the multi-level substructuring scheme and system condensation for the large-scaled structural dynamic analysis [dissertation]. Seoul: Seoul National University; 2012. 





[37] Yin J, Voss H, Chen P. Improving eigenpairs of automated multilevel substructuring with subspace iterations. Comput Struct 2013;119:115–24. https://doi.org/10.1016/j.compstruc.2013.01.004. 





[38] Elssel K, Voss H. An A priori bound for automated multilevel substructuring. SIAM J Matrix Anal Appl 2006;28:386–97. https://doi.org/10.1137/040616097. 





[39] Boo SH, Kim JG, Lee PS. Error estimation for the automated multi-level substructuring method. Int J Numer Methods Eng 2016;106:927–50. https:/ doi.org/10.1002/nme.5161. 





[40] Bathe KJ. Finite element procedures. 2nd ed. Watertown (MA): Klaus-Jürgen Bathe; 2014. 





[41] Ashman RB, Cowin SC, Van Buskirk WC, Rice JC. A continuous wave technique for the measurement of the elastic properties of cortical bone. J Biomech 1984:17:349-61, https://doi.org/10.1016/0021-9290(84)90029-0 





[42] Lee PS, Bathe KJ. Development of MITC isotropic triangular shell finite elements. Comput Struct 2004;82:945–62. https://doi.org/10.1016/ j.compstruc.2004.02.004. 





[43] Lee Y, Lee PS, Bathe KJ. The MITC3+ shell element and its performance. Comput Struct 2014;138:12–23. https://doi.org/10.1016/j.compstruc.2014.02.005. 





[44] Jeon HM, Lee Y, Lee PS, Bathe KJ. The MITC3+ shell element in geometric nonlinear analysis. Comput Struct 2015;146:91–104. https://doi.org/10.1016/ j.compstruc.2014.09.004. 





[45] Lee Y, Jeon HM, Lee PS, Bathe KJ. The modal behavior of the MITC3+ triangular shell element. Comput Struct 2015;153:148–64. https://doi.org/10.1016/ j.compstruc.2015.02.033. 





[46] Bathe KJ, Wilson EL. Solution methods for eigenvalue problems in structural mechanics. Int J Numer Methods Eng 1973;6:213–26. https://doi.org/10.1002/ nme.1620060207 





[47] Lanczos C. An iteration method for the solution of the eigenvalue problem of linear differential and integral operators. J Res Natl Bur Stand 1950;45:255. https://doi.org/10.6028/jres.045.026. 





[48] Grimes R, Lewis J, Simon H. A shifted block Lanczos algorithm for solving sparse symmetric generalized eigenproblems. SIAM J Matrix Anal Appl 1994;15:228–72. https://doi.org/10.1137/s0895479888151111. 

