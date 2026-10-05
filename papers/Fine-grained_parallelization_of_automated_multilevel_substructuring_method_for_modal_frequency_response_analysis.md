# Fine-grained parallelization of automated multilevel substructuring method for modal frequency response analysis

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/bfb2ccbb3e0d1a7ad103e37db874b60dd63dfd1208501b6a511efa11a9adb12e.jpg)


Guidong Wang <sup>a</sup> , Yujie Wang <sup>a</sup>, Xin Hu <sup>a,b</sup>, Xiangyang Cui <sup>a,b,*</sup>, Xianzhong Yu <sup>c</sup>, Senhai Liu <sup>c</sup> 

<sup>a</sup> State Key Laboratory of Advanced Design and Manufacturing Technology for Vehicle, Hunan University, Changsha 410082, China 

<sup>b</sup> Hunan Maixi Software Co., Changsha 410082, China 

<sup>c</sup> Jiangling Motors Co, Ltd., Nanchang 330200, China 

## A R T I C L E I N F O

Keywords: AMLS method Modal frequency response analysis DAG Task-based parallelism 

## A B S T R A C T

In this work, a fine-grained parallel automated multilevel substructuring (AMLS) method is proposed to improve the efficiency of modal frequency response analysis for large-scale structural models. AMLS is a multilevel extension of component mode synthesis (CMS), offering a hierarchical structure that enables extensive opportunities for parallel computation. To leverage this structure, a task-based parallel model is first introduced for the substructure transformation phase. This model breaks the synchronization barrier, allowing parent nodes to execute without waiting for descendant nodes. A directed acyclic graph (DAG) model is then used to enable parallel execution of tasks, further enhancing computational efficiency. Moreover, a priority-based task pool scheduling strategy is employed to optimize task management and execution. In addition, damping matrix and residual vector computations are integrated into the parallelization framework to further enhance overall effi ciency. The accuracy and computational efficiency of the proposed parallel AMLS method are verified through a series of numerical experiments. Specifically, the method demonstrates significant performance improvements in both weak and strong scalability tests. Furthermore, the proposed parallel AMLS method is validated through engineering case studies. These results confirm that it is well-suited for large-scale structural analyses with practical engineering applications. 

## 1. Introduction

In the field of structural dynamics, frequency response analysis is essential for understanding the behavior of structures subjected to timevarying or harmonic loads. This type of analysis is critical for applica tions such as noise, vibration, and harshness (NVH) evaluations in automotive engineering, as well as in the design of resilient civil and aerospace structures [1–3]. Accurate prediction of dynamic behavior over a wide frequency range is crucial for ensuring safety, optimizing performance, and enhancing user comfort. However, the increasing scale and complexity of modern engineering structures necessitate advanced computational strategies that can efficiently handle largescale finite element models. 

Frequency response analysis methods can be broadly classified into two categories: direct methods and modal methods [4]. The direct method solves the system’s response at each frequency point without transformation. While this provides high accuracy, it requires substantial computational resources, making it impractical for large-scale problems or fine frequency resolutions. In contrast, the modal method improves efficiency by representing the system’s response as a combination of its natural modes. For large structures, the modal method is several times faster than the direct method, making it widely adopted in engineering. However, extracting enough modes to maintain accuracy remains a performance bottleneck, particularly when dealing with wide frequency ranges and dense modes. 

To address these challenges, component mode synthesis (CMS) methods have been developed [5–9]. CMS is a reduction technique used to simplify the dynamic analysis of large structures by decomposing them into smaller substructures. Each substructure is represented by a reduced set of modes that capture its key dynamic characteristics. These modes are then combined to approximate the dynamic behavior of the entire system. Depending on the handling of interface degrees of freedom (DOFs), CMS can be classified into three types: fixed-interface, free-interface, and hybrid-interface [10]. Each type offers different levels of flexibility and accuracy in modeling the coupling between substructures, enabling efficient analysis of complex systems. 

Building on this, Bennighof’s team introduced the more streamlined and faster Automated Multi-Level Substructuring (AMLS) method [11,12]. The AMLS method extends CMS by automating the partitioning of large structures into hierarchical substructures. Unlike CMS, which typically focuses on a single level of decomposition, AMLS recursively divides the structure into smaller, more manageable substructures at multiple levels. AMLS operates at the algebraic level, bypassing the need to handle complex boundary conditions directly. This simplification allows for greater flexibility and efficiency in large-scale simulations. 

Over the past two decades, the AMLS method has evolved rapidly and proven effective in various fields. Initially, R.B. Lehocuq applied AMLS to eigenspace computation in linear elastodynamics [13], while A. Zdunek extended it to solve electromagnetic Maxwell eigenproblems [14]. Heinrich Voss adapted AMLS for eigenfrequency analysis of fluid–solid interaction systems [15]. In terms of implementation, Kaplan first identified parallelization opportunities in AMLS [16], and Phill Seung Lee developed a hybrid approach combining coarse- and finegrained parallelism [17]. Our team has also previously explored GPUaccelerated AMLS solvers [18]. 

However, these parallel implementations share a common limita tion. In the AMLS framework, substructures are organized hierarchically as a tree structure, where each node represents a substructure. In these approaches, parent nodes must wait for all of their child nodes to complete before initiating their own computations, because child sub structures are responsible for updating stiffness and mass contributions needed by their ancestors. Although this control scheme is straightfor ward to implement, it leads to inefficient thread utilization and increased idle time, particularly in unbalanced substructure trees. 

To address this bottleneck, we propose a fine-grained parallelization strategy based on explicit task decomposition and dependency analysis. Rather than treating each substructure transformation as a monolithic task, our approach divides it into six distinct computational stages. These subtasks are modeled as nodes in a directed acyclic graph (DAG), with edges denoting precise data dependencies. This structure enables finer-grained concurrency by allowing subtasks to proceed as soon as their input dependencies are met. For instance, when the stiffness matrix of a substructure has been updated by its children, stiffness-related operations can begin immediately. This does not require waiting for all mass-related updates to be completed. This relaxation of the strict execution order between parent and child nodes significantly enhances the level of parallelism. Additionally, a priority-based task scheduling strategy is introduced to further improve load balancing and optimize thread usage. 

Furthermore, while AMLS has been effectively applied to eigenvalue analysis in finite element models, its application in modal frequency response analysis remains limited. Specifically, the projection of damping matrices and the computation of residual vectors have not been explored in depth. Damping effects are essential for accurately representing the dynamic behavior of structures [19,20], and residual vectors play a critical role in compensating for the contributions of truncated higher-order modes [21]. However, both procedures can easily become computational bottlenecks in large-scale simulations. Projecting nonproportional damping matrices into the modal subspace typically involves explicit dense matrix operations. When dealing with models containing millions of degrees of freedom and thousands of retained modes, such projections incur significant memory and computational costs. On the other hand, the computation of residual vectors, commonly based on equivalent static deformation techniques, requires solving large-scale sparse linear systems, which further challenges computational efficiency. 

The AMLS framework offers an efficient solution to these problems [16]. By reusing the transformation matrices constructed during the modal reduction phase, both the damping matrices and excitation vectors can be projected into reduced-dimensional subspaces, significantly lowering computational and memory demands. Nevertheless, unlike stiffness and mass matrices, damping matrices often vary greatly in sparsity patterns and numerical structures across different engineering applications. Similar diversity exists in excitation vectors, especially under multiple loading conditions or distributed forces. Such variability limits the direct applicability of conventional AMLS parallelization strategies. 

To address these challenges, we extend the previously introduced fine-grained task decomposition and priority-based scheduling strategies to the processing of damping matrices and residual vectors, incorporating targeted optimizations for their unique characteristics. In this study, we propose an integrated solution that combines the hierarchical projection capabilities of AMLS with parallel execution mechanisms, tailored to general damping models and arbitrary excitation conditions. 

The remainder of this paper is organized as follows. Chapter 2 provides an overview of the theoretical background of frequency response analysis and the AMLS method. Chapter 3 introduces the fine-grained task parallel algorithm for the model transformation phase within the AMLS framework. Chapter 4 describes the treatment of damping effects and residual vector computation in detail. Chapter 5 demonstrates the proposed approach through engineering case studies. Finally, Chapter 6 summarizes the contributions and outlines future research directions. 

## 2. Background

This section provides an overview of the fundamental principles behind frequency response analysis and introduces the computational framework of the AMLS method. 

## 2.1. Modal frequency response analysis

Frequency response analysis is a key tool in structural dynamics for understanding how structures respond to harmonic loads. With the increasing complexity of modern engineering structures, efficient computational strategies are essential to handle large-scale finite element models. This section outlines the foundational theory of frequency response analysis, which forms the basis for developing effective computational methods. 

## 2.1.1. Fundamental theory

Frequency response analysis is used to evaluate the steady-state behavior of structures under harmonic excitations. The dynamic response of a system is described by the equation of motion: 

$$
\boldsymbol {M} \ddot {\boldsymbol {u}} + \boldsymbol {C} \dot {\boldsymbol {u}} + \boldsymbol {K} \boldsymbol {u} = \boldsymbol {f} e ^ {i \Omega t}\tag{1}
$$

where M is the mass matrix, C is the damping matrix, K is the stiffness matrix, u is the displacement vector, f is the load vector, and Ω is the excitation frequency. For harmonic loading, the steady-state displace ment response is assumed to take the form: 

$$
\boldsymbol {u} = \boldsymbol {d e} ^ {i \Omega t}\tag{2}
$$

Substituting this into the equation of motion yields: 

$$
\left[ \boldsymbol {K} - \Omega^ {2} \boldsymbol {M} + i \Omega \boldsymbol {C} \right] d e ^ {i \Omega t} = \boldsymbol {f} e ^ {i \Omega t}\tag{3}
$$

Solving the linear system in equation (3) directly to obtain the displacement vector is known as the direct frequency response analysis method. While it is highly accurate, it becomes computationally demanding for large-scale finite element models, particularly when a fine frequency resolution or a wide frequency range is required. 

To reduce computational effort, the modal method provides an effective alternative by decomposing the structural response into contributions from its natural vibration modes. First, an eigenvalue analysis is performed by solving: 

$$
\boldsymbol {K} \boldsymbol {v} = \lambda \boldsymbol {M} \boldsymbol {v}\tag{4}
$$

where λ denotes the eigenvalue (natural frequencies) and v denotes the eigenvector (mode shapes). The eigenvectors are assembled into a matrix V. The displacement can then be expressed as: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/6207a40f3e88d2d71873962202234558740ba956dbf82c2d6001b966c3a6e3f6.jpg)



Fig. 1. A complete flowchart of AMLS method.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/ad6cd5dc1624faa492fbad673efad3b1593d26377caf58ca21784e69a56a87cb.jpg)



Fig. 2. (a) Single-level partitioned structures, (b) Substructure tree.


$$
\boldsymbol {u} = \boldsymbol {V} d e ^ {i \Omega t}\tag{5}
$$

Transforming equation (1) into modal coordinates leads to: 

$$
\left[ \boldsymbol {V} ^ {T} \boldsymbol {K} \boldsymbol {V} - \Omega^ {2} \boldsymbol {V} ^ {T} \boldsymbol {M} \boldsymbol {V} + i \Omega \boldsymbol {V} ^ {T} \boldsymbol {C} \boldsymbol {V} \right] d e ^ {i \Omega t} = \boldsymbol {V} ^ {T} f e ^ {i \Omega t}\tag{6}
$$

In practical modal analysis, only a limited number of dominant modes are retained. As a result, the system dimension is much smaller than in the original equation (3). This leads to a significant reduction in the computational cost of modal frequency response analysis compared to the direct method. 

## 2.1.2. Residual vector

The modal frequency response method typically retains only a limited number of dominant modes to reduce computational cost. However, this approach introduces approximation errors, as neglected higher-order modes may still significantly affect the overall structural response. To address this, residual vectors are introduced into the solution space to account for the contributions of these omitted modes. 

In this approach, the inertial forces from the retained normal modes are first removed from the excitation force vectors F. This defines the static residual force ${ \overline { { F } } } ,$ as expressed in: 

$$
\overline {{\boldsymbol {F}}} = \boldsymbol {F} - \boldsymbol {M} \boldsymbol {V} \boldsymbol {V} ^ {T} \boldsymbol {F}\tag{7}
$$

where V denotes the matrix of retained eigenvectors and F is the global load vectors. This residual force $\overline { F }$ is then used to compute a static correction deformation vector Z by solving: 

$$
\boldsymbol {K} \boldsymbol {Z} = \overline {{\boldsymbol {F}}}\tag{8}
$$

After solving for $z ,$ it is orthogonalized with respect to the existing modal vectors $\pmb { V }$ to ensure linear independence. The next step involves augmenting the modal subspace by including Z as a new basis vector. The enriched eigenvalue problem in this extended subspace is formu lated as: 

$$
\mathbf {Z} ^ {\mathrm{T}} \mathbf {K Z V} _ {r} = \mathbf {Z} ^ {\mathrm{T}} \mathbf {M Z V} _ {r} \boldsymbol {\Lambda} _ {r}\tag{9}
$$

where V<sub>r</sub> denotes the eigenvectors and Λ<sub>r</sub> the represents the associated eigenvalues. By incorporating these residual vectors, the modal basis is effectively extended to form [V ZV<sub>r</sub>]. This enhanced basis allows the method to account for the contributions of neglected higher-order modes without explicitly computing all global modes. As a result, the accuracy of the frequency response analysis is significantly improved while maintaining high computational efficiency. 

## 2.2. Automated multilevel substructuring method

Despite the significant computational savings provided by the modal frequency response method compared to the direct approach, solving the large-scale eigenvalue problem in equation (4) remains a major performance bottleneck. 

To address this challenge, a variety of model order reduction (MOR) techniques have been developed, such as the Guyan reduction [22], Kidder Method [6,23], system equivalent reduction expansion process [24] and several CMS variants like free interface component mode synthesis method [25,26] and Craig-Bampton method [27]. However, these methods typically rely heavily on manual substructuring guided by engineering experience, requiring attention to specific modeling details such as element types and assembly interfaces. Furthermore, conventional CMS approaches necessitate explicit classification of boundary and internal degrees of freedom, increasing implementation complexity and reducing automation potential. 

In contrast, the AMLS method provides a fully automatic and hierarchical partitioning strategy based on the sparsity pattern of the global matrices. It eliminates the need to distinguish interface degrees of freedom, greatly simplifying the transformation process and improving automation. These characteristics make AMLS especially well-suited for large-scale structural dynamics problems and task-parallel computing environments. 

Given these advantages, AMLS has become the method of choice for modal analysis of large-scale finite element models in engineering practice. Its theoretical foundation and implementation for frequency response analysis were systematically described by Kaplan [12,16]. The overall computation was divided into five distinct phases, as illustrated 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/16461cdb27194c0ff9e59bcd7a618d1d855cd45d0f4b83ce4c7334cc733d1b5b.jpg)



Fig. 3. (a) Two-level partitioned structures, (b) Substructure tree.


schematically in Fig. 1. 

Phase 1 involves assembling the global stiffness, mass, and damping matrices, as well as the load vectors, as shown in equation (1), which defines the full dynamic system. Phases 2 to 4 focus on solving the eigenvalue problem in equation (4) within a given frequency range of interest. Finally, Phase 5 conducts the modal frequency response analysis using the obtained modal data. Overall, Phases 1 and 5 are consistent with standard modal-based frequency response procedures, while Phases 2 through 4 represent the core of the AMLS method for efficient modal reduction. 

Specifically, Phase 2 performs hierarchical partitioning of the global model into nested substructures using graph-based algorithms. Phase 3 extracts dominant modal information from each substructure and assembles a significantly smaller reduced eigenvalue problem. Phase 4 solves this reduced problem and reconstructs an approximate solution to the original eigenvalue system through back-transformation. The following sections provide detailed explanations of each phase. 

## 2.2.1. Preprocessing and partitioning

Phase 1 and Phase 2 represent the preprocessing stages of the AMLS algorithm. In Phase 1, the global matrices and load vectors are assembled, as described in equation (1). In Phase 2, the finite element matrices are reordered and partitioned into multiple substructures using graph partitioning tools such as METIS [28,29]. This process is conceptually similar to the substructuring in the CMS method. However, AMLS employs a hierarchical nested partitioning strategy. 

Fig. 2a shows an example of a single-level partitioned structure, where the interface between partitioned regions is also treated as a substructure. Fig. 2b depicts the corresponding substructure tree. After a single-level partitioning and reordering, the global stiffness matrix can be represented in a block form as: 

$$
K = \left[ \begin{array}{c c c} K _ {1 1} & & K _ {1 3} \\ & K _ {2 2} & K _ {2 3} \\ s y m. & & K _ {3 3} \end{array} \right]\tag{10}
$$

Substructures in Fig. 2, can be further recursively divided, as demonstrated in Figs. 3a and 3b. Through this recursive process, the matrix pattern evolves into a deeper nested block structure. In a two level partition, the stiffness matrix block form can be extended to: 

$$
K = \left[ \begin{array}{c c c c c c c} K _ {1 1} & & K _ {1 1} & & & & K _ {1 7} \\ & K _ {2 2} & K _ {1 1} & & & & K _ {2 7} \\ & & K _ {3 3} & & & & K _ {3 7} \\ & & & K _ {4 4} & & K _ {4 6} & K _ {4 7} \\ & & & & K _ {5 5} & K _ {5 6} & K _ {5 7} \\ & & s y m. & & & K _ {6 6} & K _ {6 7} \\ & & & & & & K _ {7 7} \end{array} \right]\tag{11}
$$

This hierarchical recursive partitioning automatically generates hundreds or even thousands of substructures, making large-scale model reduction much easier. This approach improves computational efficiency and eliminates the need for manual interface definition. By generating more smaller substructures, the computational cost of each substructure is reduced. 

## 2.2.2. Transformation and Solution

Phase $^ { 3 , }$ known as the model transformation phase, aims to transform the global system into a reduced subspace where the frequency response problem can be solved more efficiently. In practice, this phase is accomplished by performing the transformation process shown in Algorithm 1 for each substructure. 

Algorithm 1. Transformation procedure for the $i$ -th substructure  
1 Factorize $\pmb{K}_{ii}$ 2 For $j \in A_i$ 3 Construct the constraint modes matrix $\Psi_{ij} \colon \Psi_{ij} = -\pmb{K}_{ii}^1\pmb{K}_{ij}$ 4 Solve the dominant mode matrices $\Phi_i$ and $\Lambda_i \colon \pmb{K}_{ii}\Phi_i = \pmb{M}_{ii}\Phi_i\Lambda_i$ 5 For $j \in A_i$ 6 For $k \in \tilde{A}_j$ 7 $\pmb{K}_{jk} \leftarrow \pmb{K}_{jk} + \pmb{\Psi}_{ij}^T\pmb{K}_{ik}$ 8 $\pmb{M}_{jk} \leftarrow \pmb{M}_{jk} + \pmb{\Psi}_{ij}^T\pmb{M}_{ik} + \pmb{M}_{ij}^T\pmb{\Psi}_{ik} + \pmb{\Psi}_{ij}^T\pmb{M}_{ii}\pmb{\Psi}_{ik}$ 9 $\pmb{M}_{ij} \leftarrow \Phi_i^T (\pmb{M}_{ii}\pmb{\Psi}_{ij} + \pmb{M}_{ij})$ 10 For $j \in D_i$ 11 For $k \in A_i$ 12 $\pmb{M}_{jk} \leftarrow \pmb{M}_{jk} + \pmb{M}_{ji}\pmb{\Psi}_{ik}$ 13 $\overline{\pmb{M}}_{ji} \leftarrow \pmb{M}_{ji}\Phi_i$ 

In Algorithm $1 , A _ { i }$ denotes the ancestor set of substructure $i ,$ and $D _ { i }$ denotes its descendant set. The notation $\tilde { A } _ { j }$ represents the union of the ancestor set of substructure j with substructure j itself. The essence of Algorithm 1 is that each substructure is transformed using a local transformation matrix, which captures the dominant modal information. The transformation matrix $\pmb { T } _ { i }$ for the i-th substructure can be expressed as: 

$$
\boldsymbol {T} _ {i} = \left[ \begin{array}{c c c c c} \boldsymbol {I} & & & & \\ & \boldsymbol {\Phi} _ {i} & & \boldsymbol {\Psi} _ {i j} & \\ & & \boldsymbol {I} & & \\ & & & \ddots & \\ & & & & \boldsymbol {I} \end{array} \right] \quad j \in A _ {\mathrm{i}}\tag{12}
$$

where $\pmb { \phi } _ { i }$ contains the retained modal vectors computed in line 4 of Algorithm $^ { 1 , }$ and $\boldsymbol { \varPsi } _ { i j }$ represents the constraint modes obtained in line 3 of Algorithm 1. The transformation process described in Algorithm 1 essentially computes the transformed stiffness and mass matrices of the i-th substructure as: 

$$
\boldsymbol {K} ^ {(i)} = \boldsymbol {T} _ {i} ^ {T} \boldsymbol {K T}, \boldsymbol {M} ^ {(i)} = \boldsymbol {T} _ {i} ^ {T} \boldsymbol {M T}\tag{13}
$$

where $\pmb { K } ^ { \mathrm { ( i ) } }$ and $\pmb { M } ^ { \mathrm { ( i ) } }$ denote the stiffness and mass matrices after transformation, respectively. 

After all substructures have been transformed, the overall transformation matrix T for the global system can be expressed as: 


Table 1



Task definitions in the model transformation phase of AMLS.


<table><tr><td>Task label</td><td>Task Description</td><td>Corresponding Step in Algorithm 1</td></tr><tr><td>A</td><td>Stiffness matrix decomposition</td><td>Line 1</td></tr><tr><td>B</td><td>Computation of constraint mode matrices</td><td>Line 3</td></tr><tr><td>C</td><td>Solving the eigenvalue problem</td><td>Line 4</td></tr><tr><td>D</td><td>Updating the stiffness matrices of ancestor nodes</td><td>Line 7</td></tr><tr><td>E</td><td>Updating the mass matrices of ancestor nodes</td><td>Line 8 and Line 9</td></tr><tr><td>F</td><td>Updating the mass matrices of descendant nodes</td><td>Line 12 and Line 13</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/4bd12b8ce634e52f769c9723867c2ff03d5ea9ddc31dcfc55c7e0ab19f741e1b.jpg)



Fig. 4. Task dependency relationships for model transformation in the AMLS Method.


$$
\begin{array}{l} \boldsymbol {T} = \boldsymbol {T} _ {1} \boldsymbol {T} _ {2}... \boldsymbol {T} _ {n} = \left[ \begin{array}{c c c c c} \boldsymbol {\Phi} _ {1} & & & & \\ & \ddots & & \\ & & \boldsymbol {\Phi} _ {i} & \boldsymbol {T} _ {i j} \\ & & & \ddots & \\ & & & & \boldsymbol {\Phi} _ {n} \end{array} \right] \\ \boldsymbol {T} _ {i j} = \boldsymbol {\Psi} _ {i j} \boldsymbol {V} _ {j} + \sum_ {k} \boldsymbol {\Psi} _ {i k} \boldsymbol {T} _ {k j} \quad \text { for } j \in A _ {i}, k \in (A _ {i} \cap D _ {j}) \end{array}\tag{14}
$$

where n denotes the total number of substructures resulting from the hierarchical partitioning. Phase 4 involves solving the reduced eigen value problem: 

$$
\overline {{\boldsymbol {K}}} \boldsymbol {x} = \mu \overline {{\boldsymbol {M}}} \boldsymbol {x}, \quad w i t h \overline {{\boldsymbol {K}}} = \boldsymbol {T} ^ {T} \boldsymbol {K} \boldsymbol {T}, \overline {{\boldsymbol {M}}} = \boldsymbol {T} ^ {T} \boldsymbol {M} \boldsymbol {T}\tag{15}
$$

The eigenvalues μ obtained from this reduced system are accurate approximations to the original eigenvalues λ. The eigenvectors of the reduced problem are denoted by x, and when multiple modes are considered, these eigenvectors are assembled as a matrix X. The dimension of this reduced problem is significantly smaller than that of the original system in equation (3), resulting in substantial computational savings. Once the reduced eigenvector xare obtained, the global eigenvectors can be reconstructed via: 

$$
\boldsymbol {v} = \boldsymbol {T x}\tag{16}
$$

Phase 5 utilizes the modal information (λ and v) in the subsequent frequency response analysis. The final outputs from this analysis typically include displacements, velocities, accelerations, forces, stresses, and strains. These results provide comprehensive insight into the dy namic behavior of the structure under harmonic excitations. 

## 3. Task-Based parallel model transformation

Model transformation is the most computationally intensive phase in the AMLS method, and its parallel implementation has been a primary research focus. Previous algorithms primarily focused on multi threading or the use of accelerated devices like GPUs to speed up the transformation of higher-level nodes in the substructure tree. These strategies have improved performance, but they still suffer from significant synchronization overhead. In particular, parent nodes must wait until all child computations are finished before starting, which restricts parallel granularity. 

To overcome this limitation, a directed acyclic graph (DAG)-based execution model is introduced for upper-level substructures. The transformation procedure is divided into finer subtasks, and their dependencies are expressed through a DAG [30,31]. With this structure, a task is allowed to start execution once all its dependencies are completed. Waiting time is reduced, and parallelism is increased. A priority-based task pool management strategy is also adopted to enhance scheduling efficiency. Together, these techniques improve parallel performance and scalability by reducing unnecessary synchronization. 

## 3.1. Task decomposition

Based on the computational structure of Algorithm 1, the model transformation of a single substructure is decomposed into six subtasks, labeled A through F. These tasks are summarized in Table 1, along with their corresponding steps in Algorithm 1. 

In parallel implementations, some subtasks of a single substructure can be executed concurrently, while some others must follow a strict sequence. For example, the stiffness matrix decomposition must be performed before any other task. However, tasks such as computation of constraint mode matrices and solving the eigenvalue problem can be computed simultaneously. The task dependencies for a single substructure are illustrated in Fig. 4. This directed acyclic graph clearly shows which operations can be parallelized and which are serialized. 

In traditional node-level parallelism, the transformation of a parent substructure is delayed until all child substructures complete their computations. This creates significant synchronization overhead. A key improvement in our approach is that only the updated stiffness matrices from the children are required to initiate the parent’s transformation. Eigenvalue solutions of child nodes are no longer a blocking dependency. 

This concept is illustrated using substructure 3 in Fig. 5. Fig. 5a shows its position in the substructure tree. Fig. 5b depicts the traditional dependency graph, where all child computations must finish before task A of the parent starts. In contrast, Fig. 5c shows the optimized task graph. Task A of the parent node can now begin immediately after tasks D of its children are completed. 

The full task graph for substructure 3, including all six tasks, is shown in Fig. 6. This graph reveals additional opportunities for parallelism 

To support efficient execution, a dynamic task scheduling strategy i employed. Tasks are categorized into two states: ready and waiting. Ready tasks are placed in a queue and can be executed immediately. Waiting tasks are held in a work pool until their dependencies are resolved. Each task is associated with a dependency counter. When a task finishes, it decrements the counters of dependent tasks. Once a counter reaches zero, the corresponding task is moved to the ready queue. Threads fetch tasks from the queue dynamically and execute them without global synchronization. This mechanism minimizes idle time and reduces synchronization overhead. Fig. 7 demonstrates this dynamic scheduling mechanism. As a result, better parallel scalability is achieved compared to conventional methods. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/c4bdf7518636497a551fbcbbc1eb63060e268615b0ef4364a94c70d45037a379.jpg)



Fig. 5. Comparison of task dependencies for substructure 3: (a) structure tree, (b) traditional dependencies, (c) optimized dependencies.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/692984ad47425a0b91de46367b7e28f796d9589003d6a961ac295ae3e4db2aec.jpg)



Fig. 6. Complete task dependency graph for substructure 3.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/43d74467cde231ceb61ed96e3534169b2c6c8bd9c0bbce7bc991b2202fc71bf6.jpg)



Fig. 7. Dynamic task scheduling and dependency management.


## 3.2. Dynamic load balancing strategy

While decomposing each substructure into six subtasks improves fine-grained parallelism, it also introduces scheduling overhead. In particular, excessive task fragmentation at the leaf level may result in underutilized threads and increased runtime overhead. In contrast, upper-level nodes involve fewer but more computationally intensive tasks, which can lead to thread idling. 

To address these challenges, two complementary strategies are employed. First, computations from neighboring nodes are aggregated to form a subtree task. Each subtree is treated as a single computational unit and is assigned to one thread. This reduces scheduling overhead and improves locality. Following the approach proposed by Phill-Seung Lee [17], a cut level is defined as: 

$$
c u t \_ l e v e l = \log 2 (n u m T h r e a d s) + 4\tag{17}
$$

where numThreads is the number of threads used during execution. All nodes below the cut level are grouped into subtrees, and task decomposition is skipped for them. Each subtree is computed serially by a dedicated thread. 

Second, a priority-based scheduling scheme is introduced to improve task execution. Tasks are classified into multiple types and assigned different priorities. When multiple tasks are pending execution, those with higher priorities are executed first. In our implementation, tasks belonging to cut-level subtrees receive the highest priority, followed by stiffness matrix decomposition, constraint mode computation, ancestor stiffness updates, eigenvalue solution, ancestor mass updates, and finally, descendant mass updates. This priority configuration is designed to maximize concurrency by prioritizing tasks that can unlock a larger number of dependent operations. For example, cut-level subtree tasks are scheduled first because completing them typically enables the execution of a larger number of dependent tasks. Within each priority class, tasks are executed in first-in-first-out (FIFO) order based on readiness. 

Fig. 8 illustrates a representative example. Four tasks become ready in the order B1, A2, A1, and C1. According to the priority rules, they are executed in the order A2, A1, B1, and C1. This example demonstrates how task prioritization balances concurrency and execution order. Experimental results confirm that the combination of subtree aggregation and priority scheduling effectively mitigates the imbalance and overhead associated with fine-grained task division, leading to improved parallel scalability. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/4a0489f47dae30c2612319b2d75b89230dc2df58c31b255d951cb22c46c6a1de.jpg)



Fig. 8. Example of task readiness and execution order with priority scheduling.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/4ed25302a9b76ab28579d8e8a3e3a0635ed79eeab64b1d4cbe2855a3c638e802.jpg)



Fig. 9. Example classification of substructures.


## 4. Parallel projection of damping matrix and force vector

In modal frequency response analysis, there are two additional timeconsuming steps that cannot be ignored besides solving the large-scale generalized eigenvalue problem. The first is the projection of the damping matrix into the modal subspace. The second is the computation of residual vectors for static corrections. These operations require largescale dense matrix multiplications and linear system solutions, leading to significant computational and memory demands. 

In the AMLS framework, these projections are simplified and accelerated by using hierarchical transformations and reduced subspaces. As a result, computational cost is reduced, and memory efficiency is improved. In this chapter, the projection processes for damping matrices and force vectors within AMLS are reviewed. Parallel implementations are then proposed to further improve efficiency. The developed strategies are designed to handle arbitrary damping distributions and load cases. 

## 4.1. Damping matrix projection

## 4.1.1. Computational procedure

After obtaining modal information, the stiffness matrix, mass matrix, and damping matrix are projected into the modal subspace, as described in Equation (6). However, since the eigenvectors V are normalized with respect to the mass matrix, the projections of the mass and stiffness matrices can be simplified. In this case, $\pmb { V } ^ { T } \pmb { M } \pmb { V } = \pmb { I }$ and $\begin{array} { r } { \pmb { V } ^ { T } \pmb { K } \pmb { V } = \quad \pmb { \Lambda } , } \end{array}$ where I is the identity matrix and Λ is a diagonal matrix containing the eigenvalues. Therefore, only the damping matrix projection $\pmb { V } ^ { \operatorname { T } } \pmb { C } \pmb { V }$ needs to be computed explicitly. 

In AMLS, as shown in Equation (16), the global eigenvector matrix can be expressed as $\mathbf { } V = T \mathbf { } X .$ . Using this formulation, the damping matrix projection can be reformulated as: 

$$
\boldsymbol {V} ^ {\mathrm{T}} \boldsymbol {C V} = \boldsymbol {X} ^ {\mathrm{T}} (\boldsymbol {T} ^ {\mathrm{T}} \boldsymbol {C T}) \boldsymbol {X}\tag{18}
$$

Here, $\pmb { X } \in { \cal R } ^ { N _ { M } \mathrm { ~ x ~ } N _ { F } }$ represents the reduced modal matrix, with $N _ { M }$ denoting the dimension of the reduced subspace, and N<sub>F</sub> the number of retained modes. These values usually range from a few thousand to tens of thousands. As a result, the main computational bottleneck is shifted to the evaluation of $T ^ { \mathrm { T } } C T .$ 

Algorithm 2. Damping matrix transformation procedure for the full structure
1 Compute projected local damping matrix: $\overline{C_{ii}} = \Phi_i^T C_{ii}\Phi_i$ 2 For $j \in A_i$ 3 For $k \in \bar{A}_j$ 4 $C_{jk} \leftarrow C_{jk} + \Psi_{ij}^T C_{ik} + C_{ij}^T \Psi_{ik} + \Psi_{ij}^T C_{ii} \Psi_{ik}$ 5 $C_{ij} \leftarrow \Phi_i^T (C_{ii} \Psi_{ij} + C_{ij})$ (continued on next column) 


(continued )


<table><tr><td colspan="2">Algorithm 2. Damping matrix transformation procedure for the i-th substructure</td></tr><tr><td>6</td><td>For <eq>j \in D_i</eq></td></tr><tr><td>7</td><td>For <eq>k \in A_i</eq></td></tr><tr><td>8</td><td><eq>C_{jk} \leftarrow C_{jk} + C_{ji}\Psi_{ik}</eq></td></tr><tr><td>9</td><td><eq>\overline{C_{ji}} \leftarrow C_{ji}\Phi_i</eq></td></tr></table>

A direct approach reuses the structure of Algorithm 1, originally designed for the mass and stiffness matrices transformation. The corresponding damping projection routine is shown as Algorithm 2. An additional step is inserted at the beginning to compute the local damping transformation for each substructure. 

However, this direct approach is inefficient for general damping matrices. The main reason is the significant variation in sparsity patterns and nonzero distributions among different damping matrices. In practice, the damping matrix C can include various contributions, such as structural element damping, viscous damping, and other frequencydependent damping effects. 

In modal frequency response analysis, Equation (6) can be expanded into a more detailed form: 

$$
\left[ \boldsymbol {V} ^ {T} \boldsymbol {K} \boldsymbol {V} - \Omega^ {2} \boldsymbol {V} ^ {T} \boldsymbol {M} \boldsymbol {V} + i \boldsymbol {G} \boldsymbol {V} ^ {T} \boldsymbol {K} \boldsymbol {V} + i \boldsymbol {V} ^ {T} \boldsymbol {C} _ {\mathrm{GE}} \boldsymbol {V} + i \Omega \boldsymbol {V} ^ {T} \boldsymbol {C} _ {1} \boldsymbol {V} \right] d e ^ {i \Omega t} = \boldsymbol {V} ^ {T} f e ^ {i \Omega t}\tag{19}
$$

where G represents a uniform structural damping coefficient, $c _ { \mathrm { G E } }$ denotes the structural element damping matrix, and $c _ { 1 }$ corresponds to viscous damping generated by damper elements. Viscous damping assumes that the damping force is proportional and opposite to the structural velocity: 

$$
\boldsymbol {f} (t) = - c \dot {\boldsymbol {u}} (t)\tag{20}
$$

where c is the viscous damping coefficient. The matrix $c _ { 1 }$ is assembled from discrete viscous damping elements, such as bushes or springs, which are typically few in number. As a result, $c _ { 1 }$ is extremely sparse and has very low rank. In contrast, the matrix $c _ { \mathrm { G E } }$ is constructed by summing the damping contributions of all structural elements that include material damping properties, and is expressed as: 

$$
\pmb {C} _ {\mathrm{GE}} = i \sum_ {m} \pmb {g} _ {e, m} \pmb {k} _ {e, m}\tag{21}
$$

where $g _ { e , m }$ denotes the element level damping coefficient, and $k _ { e , m }$ is the corresponding element stiffness matrix. Although $c _ { \mathrm { G E } }$ contains fewer nonzero entries than the global stiffness matrix, it is significantly denser than $c _ { 1 }$ and exhibits a notably different sparsity pattern. These characteristics motivate the development of more efficient projection strategies beyond the direct approach in Algorithm 2. 

## 4.1.2. Parallel implementation

In the parallel implementation of damping matrix projection, no explicit distinction is made between the structural damping matrix $c _ { \mathrm { G E } }$ and the viscous damping matrix $\mathbf { C } _ { 1 } .$ . All damping components are uniformly represented by the matrix C. This abstraction enables the design of a general and efficient parallel strategy that is independent of the specific sparsity pattern. 

To manage the projection process effectively, substructures in the substructure tree are classified into three categories based on the distribution of nonzero entries in their local damping matrices. Type 1 substructures have no nonzero damping entries and whose descendants also have no nonzero damping entries. Type 2 substructures have nonzero damping entries locally, but their descendants do not. Type 3 substructures have nonzero damping entries in one or more of their descendant substructures. An example of this classification is shown in Fig. 9. Yellow nodes represent substructures with nonzero damping entries, while white nodes represent those with no nonzero entries. In this example, substructures 1, 2, and 5 are identified as Type 1; 

Table 4 


Table 2



Task definitions for damping matrix projection in AMLS.


<table><tr><td>Task label</td><td>Task Description</td><td>Corresponding Step in Algorithm 2</td></tr><tr><td>A</td><td>transformation of the diagonal block</td><td>Line 1</td></tr><tr><td>B</td><td>updates to ancestor substructures</td><td>Line 2–5</td></tr><tr><td>C</td><td>updates to descendant substructures</td><td>Line 6–9</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/c349228fb303bb5126c48c9c8cbd05f0b79fece5f096c707ca81b6f6f3d92ca5.jpg)



Fig. 10. Task dependency graph for Type 3 substructure in damping matrix projection


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/cbd68ffd34b9ef569d3b81eced0af459890f15f257ba55da1614a45bd58b100a.jpg)



Fig. 11. Rectangular plate model.


substructures 3 and 4 as Type 2; and substructures 6 and 7 as Type 3. 


Summary of model configurations used in the weak scalability test.


Since Type 1 substructures do not contribute to the damping matrix projection, they are excluded from the parallel computation. For Type 2 substructures, the computation is treated as an independent 

<table><tr><td rowspan="2">Threads</td><td colspan="2">Time (s)</td></tr><tr><td>PAMLS-1</td><td>PAMLS-2</td></tr><tr><td>1</td><td>16.945</td><td>16.965</td></tr><tr><td>2</td><td>17.864</td><td>17.853</td></tr><tr><td>4</td><td>22.755</td><td>17.872</td></tr><tr><td>8</td><td>24.952</td><td>18.211</td></tr><tr><td>16</td><td>27.944</td><td>23.981</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/41ea8c7d4dc4ca3371e0354374fb7d6904be7b58785510f294b7dc572ed35c27.jpg)



Time cost of the transformation phase under weak scaling.



Table 3


<table><tr><td rowspan="2">Threads Used</td><td rowspan="2">DOFs</td><td colspan="2">Number of non-zero entries</td><td rowspan="2">Number of substructures</td></tr><tr><td>Stiffness</td><td>Mass</td></tr><tr><td>1</td><td>408,720</td><td>11,169,564</td><td>204,360</td><td>1095</td></tr><tr><td>2</td><td>717,258</td><td>19,631,355</td><td>358,629</td><td>2047</td></tr><tr><td>4</td><td>1,244,880</td><td>34,111,368</td><td>622,440</td><td>4095</td></tr><tr><td>8</td><td>2,083,290</td><td>57,131,571</td><td>1,041,645</td><td>8189</td></tr><tr><td>16</td><td>3,564,324</td><td>97,811,082</td><td>1,782,162</td><td>15,411</td></tr></table>


Fig. 13. Normalized time comparison.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/129c83c1947ab550934402787dd4f961f030cf34be0151a32e6fcf00a795dfd7.jpg)



Fig. 14. Body-in-white model used for strong scalability analysis.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/2bcb8ec7137691672044eb6404fcfdaa5b707ed1f2c6d5e5f8a15411b23b9fba.jpg)



Fig. 12. Effect of thread count on numerical accuracy.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/806f23921665802e3ab56f3b95f316f4af831a477c0f6094ede6e6d9b52e8122.jpg)



Fig. 15. Comparison of acceleration response curves.



Table 5



Time for Body-in-White Model Frequency Response Analysis


<table><tr><td>Threads</td><td>Time (s)</td><td>Speedup</td></tr><tr><td>1</td><td>1287.299</td><td>1.00</td></tr><tr><td>2</td><td>827.511</td><td>1.56</td></tr><tr><td>4</td><td>479.848</td><td>2.68</td></tr><tr><td>8</td><td>292.053</td><td>4.42</td></tr><tr><td>16</td><td>191.457</td><td>6.73</td></tr><tr><td>24</td><td>162.728</td><td>7.93</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/0c41327e5adcd10ab4e5c23ff0e15d68bbd3b645c40b67d28d95c076947a7163.jpg)



Fig. 16. Sedan door model.



Table 6


<table><tr><td></td><td>PAMLS-1</td><td>PAMLS-2</td></tr><tr><td>Time (s)</td><td>68.463</td><td>53.127</td></tr></table>

computational task. In contrast, Type 3 substructures are divided into three distinct steps, as labeled A to $\mathrm { C } ,$ to expose finer levels of paral lelism. These steps are summarized in Table 2, along with their corresponding steps in Algorithm 2. Fig. 10 shows the task dependency relationships for substructure 7, as illustrated in Fig. 9. This fine-grained task decomposition enables higher parallel granularity, improving computational efficiency. 

As with the model transformation phase, task priorities are assigned to manage execution order. Tasks associated with Type 2 substructures are given the highest priority, while diagonal transformations for Type 3 substructures receive the lowest priority. All other tasks are assigned equal priority. This structure ensures that critical tasks are processed first, while also balancing the computational load. The task scheduling strategy introduced in Section 3.2 is applied here to manage parallel execution efficiently. 

Several practical considerations are addressed in the implementa tion. First, Type 2 substructure computations are performed using sparse matrix formats to exploit the sparsity of the damping matrix $^ { c , }$ significantly reducing memory usage. Second, updates to ancestor nodes in Type 3 substructures are not executed as a single monolithic operation. Instead, each ancestor node update is handled as an individual task, computed by a single thread. This strategy improves load balancing and reduces thread contention. The same approach is applied for updating descendant nodes. 

## 4.2. Force projection

In the AMLS framework, the static response in the modal subspace can be computed using the following formulation [16]: 

$$
\pmb {X} _ {s} = \overline {{\pmb {K}}} \pmb {T} ^ {T} \pmb {F}\tag{22}
$$

where F denotes the matrix of load vectors. Based on this, the residual vector computation procedure in AMLS is summarized in Algorithm 3. In this procedure, the major computational costs arise from line 1 and line 7, both involving transformations with the matrix T. Notably, line 7, the left multiplication by T corresponds to equation (16) in the AMLS method and has already been implemented with an efficient parallel strategy. Therefore, we focus on parallelizing the transformation in line 1, specifically the left multiplication of ${ \pmb T } ^ { \mathrm { T } } { \pmb F } .$ 

Algorithm 3. Residual vector computation in the AMLS framework
1 Compute projected force vector: $\overline{F} = T^{T}F$ 2 $F_{r} = \overline{F} - \overline{M}XX^{T}\overline{F}$ 3 $\overline{K}X_{F} = F_{r}$ 4 Orthogonalize $X_{F}$ with respect to $X$ 5 Form reduced matrices: $K_{r} = X_{F}^{T}\overline{K}X_{F}, M_{r} = X_{F}^{T}\overline{M}X_{F}$ 6 Solve the eigenvalue problem: $K_{r}X_{r} = M_{r}X_{F}\Lambda_{r}$ 7 Output residual vectors: $V_{r} = TX_{F}X_{r}$ 

In our implementation, the global load matrix F is not transformed as a single dense matrix. Instead, each force vector f is processed individually, column by column. This strategy is adopted because F is usually sparse in practical engineering scenarios. Each force vector f contains nonzero entries only at degrees of freedom corresponding to the loaded nodes. Explicitly forming the global T matrix would lead to excessive memory usage and unacceptable computational cost. To overcome this, the transformation $\pmb { T } ^ { T } \pmb { f }$ can be computed recursively as: 

$$
\overline {{\boldsymbol {f}}} = \boldsymbol {T} ^ {T} \boldsymbol {f} = \boldsymbol {T} _ {n} ^ {T} \boldsymbol {T} _ {n - 1} ^ {T}... \boldsymbol {T} _ {2} ^ {T} \boldsymbol {T} _ {1} ^ {T} \boldsymbol {f}\tag{23}
$$

Without loss of generality, both f and f can be partitioned according to substructures as: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/ea19e9f5ddca9d1f386898614b8280fcf5e6b0d5b19b93b55b3f508938d611b0.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/556e0c63-2615-46fb-be79-0884155c1ddf/fb461af99ec4bd3142f155dbadcc4db2424af5b17dbdb7eb8da1cd5c5c9cd0f5.jpg)



(b)



Fig. 17. Acceleration response curves: (a) magnitude response,(b) phase response.


$$
\boldsymbol {f} = \left[ \boldsymbol {f} _ {1} ^ {T}, \boldsymbol {f} _ {2} ^ {T},..., \boldsymbol {f} _ {n} ^ {T} \right] ^ {T}, \overline {{\boldsymbol {f}}} = \left[ \overline {{\boldsymbol {f}}} _ {1} ^ {T}, \overline {{\boldsymbol {f}}} _ {2} ^ {T},..., \overline {{\boldsymbol {f}}} _ {n} ^ {T} \right] ^ {T}\tag{24}
$$

According to equation (12), each substructure contribution $\bar { \pmb f } _ { i } ^ { T }$ can be expressed as: 

$$
\overline {{\boldsymbol {f}}} _ {i} ^ {T} = \left(\sum_ {j} \boldsymbol {f} _ {j} ^ {T} \boldsymbol {\Psi} _ {j i} + \boldsymbol {f} _ {i} ^ {T}\right) \boldsymbol {\Phi} _ {i}\tag{25}
$$

The detailed procedure for computing $\bar { \pmb f }$ is described in Algorithm 4. First, the set of substructures $S _ { 2 }$ involved in the transformation is determined. Then, a post-order traversal of these substructures is performed, where lines 8 to 10 are executed for each substructure. 


(continued )


1 Initialize the result vector: $\bar{f} = 0$ 2 Define a initial set $S_{1}: S_{1} = \{i \mid f_{i} \neq 0\}$ 3 Initialize the traversal set $S_{2} = S_{1}$ 4 For $i \in S_{1}$ 5 $S_{2} \leftarrow S_{2} \cup A_{i}$ 

(continued on next column) 

<table><tr><td colspan="2">Algorithm 4. Procedure for computing <eq>\bar{f}</eq></td></tr><tr><td>6</td><td>Perform post-order traversal of the substructure tree, for each substructure i:</td></tr><tr><td>7</td><td><eq>\textbf{If } i \in S_2</eq></td></tr><tr><td>8</td><td><eq>\bar{f}_i = (f_i^T \Phi_i)^T</eq></td></tr><tr><td>9</td><td><eq>\textbf{For } j \in P_i</eq></td></tr><tr><td>10</td><td><eq>f_j \leftarrow f_j + \Psi_{ij}^T f_i</eq></td></tr><tr><td>11</td><td>Output <eq>\bar{f}</eq></td></tr></table>

It is noted that three levels of parallelism are exploited during the projection of force vectors. The first level exists at the column level, where each column of the global load matrix F can be processed independently. The second level occurs within the computation of each individual force vector. During the transformation described in Algorithm 4, substructures can be processed in parallel by traversing the substructure tree in post-order. As detailed in Algorithm 5, tasks for the left and right child nodes are launched using the spawn directive and executed concurrently. A sync barrier ensures that both children complete before 

the parent node proceeds. The third level of parallelism is realized by employing multithreaded BLAS routines to accelerate local matrix–vector multiplications.

Algorithm 5. Parallel transformation of force vector via postorder traversal procedure postorder (i)

input: substructure number i

1 $l \leftarrow$ left child of the i-th substructure

2 $r \leftarrow$ right child of the i-th substructure

3    spawn postorder (l)

4    spawn postorder (r)

5    sync

6    If $i \in S_{2}$ 7 $\bar{f}_{i} = f_{i}^{\mathrm{T}}\Phi_{i}$ 8    For $j \in P_{i}$ 9 $f_{j} \leftarrow f_{j} + \Psi_{ij}^{\mathrm{T}}f_{i}$ 

In our implementation, the first and third levels are primarily utilized. Specifically, the parallel strategy is selected dynamically based on the number of columns in F. When this number exceeds twice the number of available threads, first-level parallelism is applied. In this case, each column is assigned to a separate thread. Otherwise, the force vectors are processed sequentially, and multithreaded BLAS is employed to accelerate the transformation of each column. 

## 5. Number experiments

In this section, a series of numerical experiments are conducted to evaluate the computational performance of the proposed parallel AMLS solver, particularly for its application in frequency response analysis. The solver is implemented in C++ and compiled using GCC 9.5.0. Intel oneAPI 2024 components are employed to accelerate numerical operations, including the use of oneMKL for dense matrix kernels. The solver is parallelized using OpenMP for shared-memory execution. 

All experiments were conducted on a Linux server running Ubuntu 22.04. The system was equipped with an Intel® Xeon® Gold 6248R processor offering 24 physical cores, along with 256 GB of DDR4 memory. 

## 5.1. Weak scalability

Weak scalability is an important metric for evaluating the perfor mance of parallel algorithms. It measures how the computational time scales when both the problem size and the number of processing threads increase proportionally. In this section, the weak scalability of the pro posed task-based parallel strategy for the AMLS method is assessed. 

For this test, a rectangular plate model with dimensions 2 m by 0.5 m is considered. The model is discretized using Q4 shell elements, as shown in Fig. 11. To simulate increasing computational load, we refine the mesh by decreasing the element size, thereby increasing the number of DOFs and matrix size. In AMLS, the computational cost does not scale linearly with either the number of elements or substructures. To overcome this, we follow the approach proposed by Phill-Seung Lee [17]. Specifically, a single-threaded AMLS solver is used to measure the computational time of different mesh configurations. In this case, it is assumed that the computational load of a finer mesh is approximately twice that of a coarser mesh if the computation time for the finer mesh is roughly double. 

For comparison, the traditional parallel AMLS strategy proposed by Kaplan [16] is adopted as the baseline method, denoted as PAMLS-1, and compared with the proposed fine-grained task-based implementa tion, referred to as PAMLS-2. 

PAMLS-1 adopts a coarse-grained parallel approach that exploits two main levels of concurrency: (1) substructures on different branches of the substructure tree can be processed independently; and (2) within each substructure, updates to the stiffness and mass matrices for ancestor nodes can be performed in parallel. 

To evaluate and compare the performance of PAMLS-1 and PAMLS-

2, five plate models with increasing mesh densities are constructed. Each model is assigned a number of threads proportional to its expected workload: $1 , 2 , 4 , 8 ,$ and 16 threads. The model parameters are summarized in Table 3. 

In this experiment, the objective is to extract modal information below 1000 Hz. Fig. 12 presents the relative frequency errors of the first 20 non-rigid-body modes between PAMLS-1 and PAMLS-2 under different thread configurations. The relative error for each mode is defined as: 

$$
\varepsilon_ {i} = \frac {\left| \lambda_ {i} ^ {(2)} - \lambda_ {i} ^ {(1)} \right|}{\lambda_ {i} ^ {(1)}}\tag{26}
$$

where $\lambda _ { i } ^ { ( 1 ) }$ and $\lambda _ { i } ^ { ( 2 ) }$ represent the i-th eigenvalue computed using PAMLS-1 and PAMLS-2, respectively. Across all thread configurations, the relative errors remain consistently below $1 0 ^ { - 4 }$ , indicating that the proposed fine-grained parallel scheme maintains high modal accuracy while improving computational performance. 

Table 4 presents the transformation times of PAMLS-1 and PAMLS-2 under weak scaling conditions, while the normalized time comparison is illustrated in Fig. 13. The normalized time $N _ { \mathrm { t } }$ is defined as: 

$$
N _ {\mathrm{t}} = \mathrm{T} _ {\mathrm{t}} / \mathrm{T} _ {1} (2 7).
$$

where $\mathrm { T } _ { 1 }$ is the transformation time using a single thread, T is the transformation time using t threads. In the ideal case of perfect linear scalability, the normalized time would be exactly 1.0 for all thread counts. The results indicate that PAMLS-2 outperforms PAMLS-1 in most test configurations, particularly as the number of threads increases. 

At 2 threads, the performance of PAMLS-1 and PAMLS-2 is comparable. However, for 4 to 8 threads, PAMLS-2 demonstrates clear advantages. Its fine-grained task decomposition and DAG-based scheduling enable parent and child tasks to execute with minimal synchronization delays. In contrast, PAMLS-1 enforces a strict parent-afterchildren rule, introducing idle time and reducing parallel efficiency. As the thread count increases to 16, the normalized time rises for both methods. This is mainly due to the presence of large substructures near the root of the hierarchy. These require operations such as matrix factorization and eigenvalue decomposition, which inherently do not scale well beyond a certain level due to their limited parallelism. Despite this, the proposed parallel strategy shows reasonable scalability under increasing computational loads, confirming the efficiency of the taskbased parallel implementation compared to previous methods. 

## 5.2. Strong scalability

Strong scalability is an another important metric for evaluating the performance of parallel algorithms. It measures how the computational time changes when the problem size remains constant and the number of processing threads increases. In this section, we evaluate the strong scalability of the proposed parallel approach by applying it to a body-in white model shown in Fig. 14. 

The model utilizes multiple element types, including CHEXA, CTE-TRA, CPENTA, Q4, CTRIA3, and CBUSH. It contains a total of 1,553,705 elements and 1,599,722 nodes, resulting in 9,404,385 degrees of freedom. The stiffness matrix, mass matrix, damping matrix, and excitation vector are extracted for the analysis. 

The proposed parallel algorithms are applied using varying number of threads to assess the performance improvements across different parallel configurations. To verify the consistency of results under different parallel configurations, the acceleration response curves from the single-thread and 24-thread executions are compared in Fig. 15. As shown, the curves are nearly identical, confirming the numerical stability and thread safety of the parallel implementation. 

The execution times and corresponding speedup ratios are presented in Table 5. The results indicate a significant reduction in computation time as the number of threads increases. The execution time decreases from 1287.299 s with a single thread to 162.728 s with 24 threads. The peak speedup achieved with 24 threads is approximately 7.93x. Although this is below ideal linear scaling, it reflects the expected limitations of operations such as matrix factorization, which inherently scale poorly. In this model, there are 50 substructures with more than 1000 DOFs, with the largest reaching 7137 DOFs. All of these large substructures are concentrated near the upper levels of the substructure tree, which severely limits concurrency. This observation is consistent with the findings of Phill-Seung Lee [17]. For such large substructures, operations like stiffness matrix factorization and eigenvalue solution typically reach their performance ceiling at around five threads, resulting in diminished returns when more threads are added. Additionally, this is a real-world engineering case with complex connectivity, which increases the difficulty of computing the coupled matrices. Despite these challenges, the parallel performance remains impressive. 

Notably, the continued acceleration from 16 to 24 threads demonstrates that the proposed algorithm has good scalability and efficiently utilizes computational resources as both model size and thread count increase. These results confirm that the proposed task-based strategy provides strong scalability. 

## 5.3. Engineering case

This section demonstrates the practical applicability of the proposed frequency response analysis algorithm by applying it to a sedan door model, as shown in Fig. 16. The model consists of 204,743 nodes, 211,596 elements, and 1,225,714 DOFs, representing a typical automotive component. The model is discretized using a combination of shell and solid elements to capture the complex structural behavior of the door. 

The analysis focuses on computing frequency response curves across a frequency range from 1 Hz to 500 Hz, sampled at 1 Hz intervals. Modal information is extracted up to 750 Hz to meet engineering requirements and ensure modal completeness. The cutoff frequency for each substructure is set to five times the base frequency, which is consistent with the default parameter in commercial AMLS implementations such as the AMSES(Automatic Multi-level Sub-structuring Eigensolver Solution) solver in OptiStruct. This choice ensures a balance between solution accuracy and computational efficiency. 

Both the traditional parallel AMLS algorithm (PAMLS-1) and the proposed algorithm (PAMLS-2) were used for the frequency response analysis. In both cases, 307 modes were computed within the frequency range of 0–750 Hz. 

The computation times for each method are shown in Table 6. The time difference between the two methods is attributed not only to the improved parallel efficiency of the model transformation phase but also to the algorithms introduced in Chapter 4, which significantly reduce the computational load. 

In terms of memory usage during the eigenvalue problem stage, both PAMLS-1 and PAMLS-2 exhibit similar memory requirements. This is because the underlying computations are identical, and only the execution order of tasks differs. However, in the subsequent stages, PAMLS-1 incurs significantly higher memory consumption. 

Specifically, solving the residual vectors in PAMLS-1 relies on the Pardiso solver for large sparse linear systems, requiring approximately 2.6 GB of memory under double-precision arithmetic. Additionally, projecting the damping matrix using the V<sup>T</sup>CV approach requires temporary storage for the CV results, consuming about 2.8 GB. In contrast, PAMLS-2 operates on a reduced-order model with only 3580 degrees of freedom. Even after accounting for temporary data generated during projection, the total memory usage remains below 500 MB. 

The comparison of acceleration response curves is visualized in Fig. 17. In this figure, the purple line represents PAMLS-1, and the red line represents PAMLS-2. The results show that both methods produce nearly identical accuracy, with the curves overlapping. This indicates that the proposed method maintains high precision while offering improved efficiency. 

## 6. Conclusion

This work presents an enhanced parallel AMLS method for improving the efficiency of modal frequency response analysis. A taskbased parallel model was developed to optimize the substructure transformation phase, overcoming synchronization barriers and enabling better computational performance. The method also incorporates parallel approaches for computing damping matrices and residual vectors. Numerical experiments demonstrated the effectiveness of the proposed method, with significant speedup and improved scalability. 

Future work will focus on extending the AMLS method to distributed computing environments to handle even larger models efficiently. The integration of GPU acceleration and hybrid MPI-OpenMP parallelism will be explored to further optimize performance, enabling real-time simulations for complex structural systems. These advancements will enhance the scalability and applicability of AMLS in industries like aerospace, automotive, and civil engineering. 

## CRediT authorship contribution statement

Guidong Wang: Writing – original draft, Supervision, Software, Methodology, Investigation, Formal analysis, Conceptualization. Yujie Wang: Methodology, Formal analysis, Data curation. Xin Hu: Validation, Resources, Methodology. Xiangyang Cui: Resources, Funding acquisition, Data curation. Xianzhong Yu: Resources, Funding acquisition. Senhai Liu: Resources, Funding acquisition. 

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. 

## Acknowledgement

This work was supported by the National Key R&D Program of China (2022YFB2503505); the National Natural Science Foundation of China (12302259); the China Postdoctoral Science Foundation (22022 M721099). 

## Data availability

No data was used for the research described in the article. 

## References



[1] Azadi S, Azadi M, Zahedi F. NVH analysis and improvement of a vehicle body structure using DOE method. J Mech Sci Technol 2009;23(11):2980–9. 





[2] Kropp A, Heiserer D. Efficient broadband vibro-acoustic analysis of passenger car bodies using an FE-based component mode synthesis approach. J Comput Acoust 2003;11(02):139–57 





[3] Ma Z-D, Kikuchi N, Hagiwara I. Structural topology and shape optimization for a frequency response problem. Comput Mech 1993;13(3):157–74. 





[4] Ma Z-D, Hagiwara I. Sensitivity analysis methods for coupled acoustic-structural systems part II: direct frequency response and its sensitivities. AIAA J 1991;29(11): 1796–801. 





[5] Hurty WC. Dynamic analysis of structural systems using component modes. AIAA J 1965;3(4):678–85 





[6] Guyan RJ. Reduction of stiffness and mass matrices. AIAA J 1965;3(2):380. 





[7] Jr C, Roy R, Bampton MCC. Coupling of substructures for dynamic analyses. AIAA J 1968;6(7):1313 9. 





[8] MacNeal RH. A hybrid method of component mode synthesis. Comput Struct 1971; 1(4):581–601. 





[9] Craig Jr, Roy R. A review of time-domain and frequency-domain component mode synthesis method.” (1985). 





[10] Sun Y, Yu Lu, Song Z. Review on the theories and applications of dynamic condensation and component mode synthesis methods in solving EEM-based structural dynamics,Acta Mech Solida Sin 2023:36(3):361-89 





[11] Bennighof, Jeffrey, Connie kim. An adaptive multi-level substructuring method for efficient modelingof complex structures 33rd structures. Structural Dynamics and 





[12] Bennighof, Jeffrey K., and Matthew F. Kaplan. “Frequency window implementation of adaptive multi-level substructuring.” (1998): 409-418. 





[13] Bennighof JK, Lehoucq RB. An automated multilevel substructuring method for eigenspace computation in linear elastodynamics. SIAM J Sci Comput 2004;25(6): 2084–106. 





[14] Rachowicz W, Zdunek A. Automated multi-level substructuring (AMLS) for electromagnetics. Comput Methods Appl Mech Eng 2009;198(13-14):1224–34. 





[15] Stammberger M, Voss H. Automated multi-level sub-structuring for fluid solid interaction problems. Numer Linear Algebra Appl 2011;18(3):411 27. 





[16] Kaplan MF. Implementation of automated multilevel substructuring for frequency response analysis of structures. The University of Texas at Austin; 2001. 





[17] Hyun C, Lee P-S. A load balancing algorithm for the parallel automated multilevel substructuring method. Comput Struct 2021;257:106649. 





[18] Wang G, et al. A GPU-Accelerated automated multilevel substructuring method for modal analysis of structures. Comput Struct 2024;305:107516. 





[19] Yamaguchi T, Kurosawa Y, Enomoto H. Damped vibration analysis using finite element method with approximated modal damping for automotive double walls with a porous material. J Sound Vib 2009;325(1-2):436–50. 





[20] Vasques CMA, Moreira RAS, Dias Rodrigues J. Viscoelastic Damping Technologies-Part I: Modeling and Finite Element Implementation. Journal of advanced research Mech Eng 2010;1:2. 





[21] Dickens JM, Nakagawa JM, Wittbrodt MJ. A critique of mode acceleration and modal truncation augmentation methods for modal response analysis. Comput Struct 1997;62(6):985–98. 





[22] Elssel Kolja, and Heinrich Voss. “Multilevel extended algorithms in structural dynamics on parallel computers.” Advances in Parallel Computing. Vol. 13. North-Holland, 2004. 243-250. 





[23] Kidder RL. Reduction of structural frequency equations. AIAA J 1973;11(6):892 





[24] Dinh-Cong D, Truong TT, Nguyen-Thoi T. A comparative study of different dynamic condensation techniques applied to multidamage identification of FGM and FG-CNTRC plates. Eng Comput 2022;38:3951–75. 





[25] O’Callahan J. System equivalent reduction and expansion process. Proceedings of the 7th International Modal analysis conference, Society of Experimental Mechanics. 1989;7:29–37. 





[26] Craig RR, Chang CJ. Free-interface methods of substructure coupling for dynami analysis. AIAA J 1976;14(11):1633 5 





[27] Hou SN. Review of modal synthesis techniques and a new approach. Shock Vib Bull 1969;40:25 39. 





[28] Craig RR, Bampton MCC. Coupling of substructures for dynamic analyses. AIAA J 1968;6(7):1313–9. 





[29] George A. Nested dissection of a regular finite element mesh. SIAM J Numer Anal 1973;10(2):345–63. 





[30] Karypis, George, and Vipin Kumar. “METIS: A software package for partitioning unstructured graphs.” Partitioning Meshes, and Computing Fill-Reducing Orderings of Sparse Matrices, Version 4.0 (1998).Ueter, Niklas, et al. “Parallel path progression DAG scheduling.” IEEE Transactions on Computers 72.10 (2023): 3002-3016. 





[31] Kim K, Eijkhout V. A parallel sparse direct solver via hierarchical DAG scheduling. Acm transactions on mathematical software (toms) 2014;41(1):1–27 

