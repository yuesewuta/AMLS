Copyright 

by 

Matthew Frederick Kaplan 

2001 

The Dissertation Committee for Matthew Frederick Kaplan certifies that this 

is the approved version of the following dissertation: 

# Implementation of Automated Multilevel Substructuring for Frequency Response Analysis of Structures

Committee: 

Jeffrey K. Bennighof, Supervisor 

Roy R. Craig, Jr. 

Roy R. Craig, Jr.
Linda J. Hayes 

Donald O. Harman 

Richard C. Stearman
Rohst A. and Chiu 

Robert A. van de Geijn 

# Implementation of Automated Multilevel Substructuring for Frequency Response Analysis of Structures

by 

Matthew Frederick Kaplan, B.S.A.S.E. 

Dissertation 

Presented to the Faculty of the Graduate School of 

The University of Texas at Austin 

in Partial Fulfillment 

of the Requirements 

for the Degree of 

Doctor of Philosophy 

The University of Texas at Austin 

December 2001 

UMI Number: 3037508 

Copyright 2002 by Kaplan, Matthew Frederick
All rights reserved. 

UMI® 

UMI Microform 3037508
Copyright 2002 ProQuest Information and Learning Company.
All rights reserved. This microform edition is protected against unauthorized copying under Title 17, United States Code. 

ProQuest Information and Learning Company
300 North Zeeb Road
PO Box 1346
Ann Arbor, MI 48106-1346 

To my wife and family, who knew I could do it and made it possible to try. 

## Acknowledgments

This research project took much longer than anyone thought and has reached heights no one really expected at the beginning. There are many people who have contributed along the way. I would like to thank Osni Marques, Inderjit Dhillon, and Rich Lehoucq for their assistance with various eigensolver issues over the years. Thanks also go to Scott Messec and Reuben Reyes for their help with networking, computers, visualization, and many other practical matters. I am grateful to Mintae Kim, Chang-Wan Kim, and Ruijie Liu for being such good students that they were able to take over my work. I am indebted to Dr. Robert van de Geijn for his advice, insight, and perspective from the other side of the street. I am likewise indebted to Dr. Suzanne Weaver Smith for her advice, insight, and perspective from the other side of the country. It has been my privilege to have worked with Dr. Roy Craig. I would like to thank Dr. Craig for setting such a high standard and for innumerable acts of kindness. Two people who have supported this research both morally and substantively are Chris Tutt and Mladen Chargin. I am very grateful to both of them. I most especially appreciate my partner in crime, Mark Muller, without whom this research would never have gone so far. Of the many other people who have helped me along in this process, I would like to thank Paul Conti, Jeanie Duvall, Nita Pollard, and Donna Soward. 

During my tenure, this research has received financial support from the Texas Higher Education Coordinating Board, ONR, Ford Motor Company, IBM, Cray, Hewlett-Packard, Sun Microsystems, SGI, Sandia National Laboratories, and NASA. In addition, several hardware vendors provided extensive access to their technical support personnel. I would like to thank Doug Petesch, Martin Feyereisen, David Whitaker, Dave Strenski, Dawson Deuermeyer, and Mark Kelly. I also received funding from AFOSR through the National Defense Science and Engineering Fellowship Program. 

Finally, I would like to thank Dr. Jeff Bennighof for putting up with me and helping me grow academically and professionally. I must also thank Dr. Bennighof for having a great idea and letting me share the ride. It was a great ride while it lasted. Thank you! 

MATTHEW FREDERICK KAPLAN 

The University of Texas at Austin 

December 2001 

# Implementation of Automated Multilevel Substructuring for Frequency Response Analysis of Structures

Publication No. ____ 

Matthew Frederick Kaplan, Ph.D. 

The University of Texas at Austin, 2001 

Supervisor: Jeffrey K. Bennighof 

In the design of vehicles, such as automobiles, aircraft, spacecraft, or submarines, it is important to be able to accurately predict dynamic behavior of the structure. With the extremely high cost of building physical prototypes of these vehicles, there is a growing emphasis on analysis of computer models. In this dissertation, a method known as Automated Multilevel Substructuring (AMLS) is presented for accurately solving frequency response problems involving large, complex models with millions of degrees of freedom. Conventional methods for addressing these problems, such as mode superposition using a Lanczos eigensolver or model reduction using component mode synthesis, are reviewed. 

The Automated Multilevel Substructuring (AMLS) method partitions finite element models into substructures, similar to component mode synthesis methods, but uses an automated partitioning procedure that reduces the burden on the analyst. The finite element matrices are projected onto a reduced subspace, on which the frequency response is computed. Two frequency response algorithms are presented. Both methods require the solution of a global eigenvalue problem on the reduced subspace. The first method uses straightforward mode superposition. The second method employs a new iterative approach in which the modal frequency response leads to a residual problem that is solved using an iterative splitting method. The global eigensolution and frequency response algorithms are specifically designed to take advantage of the properties of the reduced subspace. Numerical examples are presented for models with millions of degrees of freedom. The performance and accuracy of the AMLS method are compared to the standard commercial software package for large-scale linear dynamic analysis. These examples establish that AMLS can be used to accurately obtain the response of very large models with significantly less computational resources than competing methods. In comparison to the modal frequency response obtained with the standard commercial software package using a shifted block Lanczos algorithm, AMLS ran up to 6.4 times faster, used less memory, and required an order of magnitude less data transfer. Thus, the AMLS method makes it possible to do frequency response analysis of large, complex structures at higher frequencies than was previously practical. 

## Contents

Acknowledgments v   
Abstract vii   
List of Tables xiii   
List of Figures xv   
Chapter 1 Introduction 1   
1.1 Brief History of AMLS . 2   
1.2 An Overview of the AMLS Method . 3   
1.3 Review of Competing Methods . 3   
1.4 Author's Contribution . 8   
1.5 Outline of Dissertation . 8   
Chapter 2 Survey of Numerical Methods 10   
2.1 Solution Methods for Large Linear Systems of Equations . 11   
2.1.1 Direct Methods . 11   
2.1.2 Iterative Methods . 13   
2.1.3 Multigrid Methods and Domain Decomposition . 16   
2.2 Eigensolution Techniques . 20   
2.2.1 Power Methods . 21   
2.2.2 Similarity Transformation Methods . 26   
2.2.3 Krylov and Lanczos Algorithms . 27 

2.2.4 Other Acceleration Techniques 32  
2.3 Model Reduction Techniques 33  
2.3.1 Static Reduction 34  
2.3.2 Mode Superposition Method 35  
2.3.3 Dynamic Reduction 36  
2.3.4 Component Mode Synthesis 37  

Chapter 3 Phases 1 and 2: Matrix Generation and Automated Partitioning 43  
3.1 Phase 1: Generating Finite Element Matrices 43  
3.2 Phase 2: Automated Substructuring 45  
3.2.1 Matrix Reordering 46  
3.2.2 Matrix Partitioning 46  
3.2.3 Special Considerations 48  

Chapter 4 Phase 3: Model Transformation 50  
4.1 CMS on a Single Level 50  
4.2 Multilevel Substructuring 54  
4.3 Greedy Algorithm 62  
4.3.1 Bottom Level Substructure Transformation 64  
4.3.2 Higher Level Substructure Transformation 65  
4.3.3 Highest Level Substructure Transformation 68  
4.3.4 Cost of the Greedy Algorithm 68  
4.4 Transformation Matrix 69  
4.5 Lazy Algorithm 76  
4.6 Greedy vs. Lazy 79  
4.7 Transformation Matrix Revisited 82  
4.8 Opportunities for Parallelism 84  

Chapter 5 Phase 4: Global Eigensolution 86  
5.1 Motivation for Eigensolver Design 88  
5.2 Preliminary Eigensolution Algorithm 89  
5.3 Starting Subspace 89 

5.4 Inverse Iteration 91  
5.5 “2 × 2” Acceleration 92  
5.6 Divergence Eigenvalue Problem 93  
5.7 Residual Flexibility 97  
5.8 Rigid-Body Modes 98  
5.9 Eigensolution Algorithm 100  
5.10 Cost of Eigensolution 103  
5.11 Opportunities for Parallelism 107  

Chapter 6 Phase 5: Frequency Response Analysis 108  
6.1 Mode Superposition Algorithm 109  
6.2 Iterative Algorithm 111  
6.3 Extrapolation 117  
6.4 Orthogonality 118  
6.5 Obtaining the Frequency Response 121  
6.6 Cost of Frequency Response 121  
6.7 Opportunities for Parallelism 123  

Chapter 7 Numerical Results and Performance 125  
7.1 Quarter Panel Model 128  
7.1.1 Effect of Substructure Eigenvector Cutoff Frequency $\omega_{ss}$ 129  
7.1.2 Effect of the Global Modes Cutoff Frequency $\omega_g$ 130  
7.1.3 Effect of the Substructure Eigenvector Cutoff Frequency for the Starting Subspace $\omega_B$ 132  
7.1.4 Effect of the Starting Subspace Cutoff Factor $\alpha_B$ 133  
7.1.5 Convergence of Eigenvalues 136  
7.1.6 Quality of Residual Estimate 137  
7.1.7 Overall Performance and Accuracy 140  
7.2 Body-In-White Model 144  
7.2.1 Effect of Substructure Eigenvector Cutoff Frequency $\omega_{ss}$ 146  
7.2.2 Effect of Global Modes Cutoff Frequency $\omega_g$ 146  
7.2.3 Effect of Residual Flexibility 149 

7.2.4 Mode Superposition Versus Iterative Algorithm 151  
7.2.5 Overall Performance and Accuracy 151  
7.3 Trim Body Model 155  
7.3.1 Effect of Global Modes Cutoff Frequency $\omega_{g}$ 156  
7.3.2 Mode Superposition Versus Iterative Algorithm 157  
7.3.3 Overall Performance and Accuracy 159  
Chapter 8 Conclusions and Advanced Developments in AMLS 163  
8.1 Conclusions 163  
8.2 Current and Future Developments in AMLS 165  
8.2.1 Improving Performance 165  
8.2.2 New Capabilities 166  
8.2.3 New Algorithms for Current Capabilities 167  
8.3 Final Remarks 169  
Appendix A Estimating the Residual 170  
References 173  
Vita 183 

## List of Tables

2.1 Algorithm for the Conjugate Gradient method . . . . . . . . . . . . . . . . . 16
2.2 Example of basic two-grid algorithm . . . . . . . . . . . . . . . . . 18
2.3 Operations required for Lanczos steps . . . . . . . . . . . . . . . . 31
5.1 Preliminary Phase 4 algorithm . . . . . . . . . . . . . . . 90
5.2 Parameter summary for Phase 4 algorithm . . . . . . . . . . . . 104
7.1 Computer Configurations . . . . . . . . . . . . . . 127
7.2 Effect of $\omega_{ss}$ on cost of PHASE3 and PHASE4 for QTR model . 130
7.3 Effect of $\omega_g$ on the cost of PHASE4 and PHASE5 for QTR model 130
7.4 Effect of $\omega_B$ on the cost of PHASE4 for QTR model 132
7.5 Effect of $\alpha_B$ on the cost and accuracy of PHASE4 for QTR model 134
7.6 Performance characteristics for each phase of AMLS method on QTR model 141
7.7 Performance comparison between AMLS method and MSC.Nastran on QTR model 142
7.8 Effect of $\omega_{ss}$ on cost of PHASE3 and PHASE4 for BIW model on IBM workstation 146
7.9 Effect of $\omega_g$ on the cost of PHASE4 for BIW model on IBM workstation 148
7.10 Effect of $\omega_g$ on the cost of PHASE4 and PHASE5 for BIW model on IBM workstation 149
7.11 Comparison of the cost of iterative and mode superposition algorithms for BIW model on IBM workstation 151
7.12 Performance characteristics for each phase of AMLS method on BIW model 153 

7.13 Performance comparison between AMLS method and MSC.Nastran on BIW model ..... 153
7.14 Effect of $\omega_{g}$ on the cost of PHASE4 and PHASE5 for CAB model on IBM workstation ..... 157
7.15 Comparison of the cost of iterative and mode superposition methods for CAB model on IBM workstation ..... 158
7.16 Performance characteristics for each phase of AMLS method on CAB model . 160
7.17 Performance comparison between AMLS method and MSC.Nastran on CAB model ..... 160 

## List of Figures

1.1 Flowchart of AMLS method . . . . . . . . . . . . . . . . . . . . . 4
2.1 Two discretizations of a square plate for a multigrid algorithm . . . . . . . 17
2.2 Rectangular plate with two components . . . . . . . . . . . . . . 38
3.1 Flowchart of Phases 1 and 2 . . . . . . . . . . . . . 44
3.2 Sparsity pattern for reordered matrix . . . . . . . . . . . 47
4.1 Flowchart for Phase 3 algorithm . . . . . . . . . . . . 51
4.2 Substructure tree diagram for rectangular plate model . . . . . . . . 52
4.3 S4 model: a square plate partitioned into 4 subdomains 54
4.4 Possible substructuring for S4 model 55
4.5 S4 model substructure tree 55
4.6 Progress of a right-looking algorithm for the S4 model 63
4.7 Progress of the greedy algorithm for the S4 model stiffness and mass matrices 67
4.8 Substructure A defined for S4 model 70
4.9 Progress of a left-looking algorithm for the S4 model 77
4.10 Progress of the lazy algorithm for the S4 model stiffness and mass matrices 80
5.1 Flowchart for Phase 4 algorithm 87
6.1 Flowchart of Phase 5 algorithms 110
6.2 Frequency response of single degree of freedom system 112
7.1 Four-door automobile with rear quarter panel highlighted 128 

7.2 Substructure tree diagram for the QTR model ..... 129
7.3 Effect of $\omega_{ss}$ on accuracy of natural frequencies for QTR model ..... 131
7.4 Effect of $\omega_B$ on accuracy of natural frequencies for QTR model ..... 133
7.5 Number of global modes as a function of cutoff frequency for QTR model .. 135
7.6 Effect of $\alpha_B$ on the accuracy of natural frequencies for QTR model ..... 135
7.7 Accuracy of PHASE4 natural frequencies compared to the available accuracy in transformed QTR model ..... 136
7.8 Comparison of convergence of natural frequencies for several eigensolution strategies ..... 138
7.9 Close-up view of convergence of natural frequencies for several eigensolution strategies ..... 138
7.10 Comparison of estimated and computed residuals for the QTR model ..... 139
7.11 Sparsity structure of finite element stiffness matrix for QTR model ..... 140
7.12 Sparsity structure of transformed mass matrix for QTR model ..... 141
7.13 Error in natural frequencies for the QTR model ..... 143
7.14 Point mobility for the QTR model ..... 144
7.15 Transfer mobility for the QTR model ..... 145
7.16 Effect of $\omega_{ss}$ on accuracy of natural frequencies for BIW model ..... 147
7.17 Effect of $\omega_g$ on accuracy of point inertance for BIW model ..... 148
7.18 Effect of residual flexibility on accuracy of point inertance for BIW model .. 150
7.19 Close-up view of residual flexibility effect on point interance for BIW model 150
7.20 Comparison of point inertance of mode superposition and iterative algorithms for BIW model ..... 152
7.21 Error in the natural frequencies for BIW model ..... 154
7.22 Point inertance for BIW model ..... 154
7.23 Transfer inertance for BIW model ..... 155
7.24 Full size truck with cab section highlighted ..... 156
7.25 Point inertance for CAB model ..... 158
7.26 Close-up view of point inertance for CAB model ..... 159
7.27 Error in the natural frequencies for CAB model ..... 161
7.28 Transfer inertance for CAB model ..... 162 

## Conventions

The following conventions will be observed in all equations. lower case bold letters will be used to denote vectors. CAPITAL LETTERS are reserved for matrices, except $N$ which will be used as a scalar integer. The letter $\Phi$ is reserved for eigenvector matrices. In the text, PROGRAM and SUBPROGRAM titles will be given in italic, capital letters. Reading from or writing to disk during the course of a program's execution will be referred to collectively as $I/O$ . Other notational conventions will be described as needed. 

## Chapter 1

## Introduction

The Automated Multilevel Substructuring (AMLS) method was developed from the beginning with the goal of solving for the frequency response of complex structure models. A frequency response function (FRF) is the ratio of the response measured at one location to the force input at the same location or at some other location as a function of frequency. The response may be measured in terms of displacement, velocity, or acceleration and may be obtained by solving a linear system of equations of the form 

$$
\left[ K (1 + i \gamma) - \omega^ {2} M + i \omega B + i \Gamma \right] X (\omega) = F (\omega)\tag{1.0.1}
$$

where K is the stiffness matrix, M is the mass matrix, B is the viscous damping matrix, $\Gamma$ is the structural damping matrix, $\gamma$ is the global structural damping factor, $\omega$ is the excitation frequency, X is the matrix of displacements, and F is the matrix of load vectors. 

The types of structures this method addresses include (but are not limited to) aircraft, spacecraft, automobiles, and submarines. For these structures, building and testing physical prototypes is extremely expensive and time-consuming. This places a premium on the ability to accurately predict a structure's dynamic behavior with computer models within the design process. In the automobile industry, for example, we routinely see models of entire car bodies with over a million degrees of freedom. At this point, the analysis of such models is limited by the capabilities of current methods to frequencies less than 200 Hz. While analysts would like to investigate higher frequency behavior, two major factors make that impractical. First, the time required to compute the response is too great. The design cycle for a car typically takes three to four years between concept and production. To keep up with the short design cycles, analysts in the car industry need the results in less than a day. At 400 Hz, the analysis for some models by conventional methods could take more than two days of dedicated computer time. The second reason is model accuracy. At higher frequencies, the finite element model may not be accurate enough to generate results that correlate well with the actual response of the structure. It is likely that more refined models with even greater numbers of degrees of freedom will be required to reliably predict behavior at higher frequencies. 

In the AMLS method, the finite element model of the structure is projected onto a reduced subspace by a transformation based on component mode synthesis. The frequency response problem (or the generalized eigenvalue problem) can then be solved more efficiently on the reduced subspace. Because the generality of the finite element formulation allows for deformations that the structure is not likely to physically reproduce, such as deformations that are highly discontinuous in stress on element boundaries, there is potentially a much smaller subspace that can still accurately represent the response of the structure. The AMLS method takes advantage of this reduced subspace to enable the use of larger models and significantly reduce the computation time required, extending the practical range of frequency response analysis without sacrificing accuracy. 

## 1.1 Brief History of AMLS

AMLS was first presented by Bennighof and Kim [1] in 1992 in the context of frequency response for one-dimensional problems. This paper described the basics of the transformation, but took an entirely different approach to obtain the frequency response from the method presented here. Also, the partitioning into substructures was not yet automated. In 1993, the method was applied to three-dimensional structure models for fluid/structure coupling problems by Bennighof [2]. Bennighof and Kaplan [3, 4] extended the capabilities of the approach for frequency response problem by looking at ways to efficiently compute the response over wider frequency ranges. By 1998, the automated substructure partitioning had been implemented. In the same year, a new algorithm was developed by Bennighof and Kaplan [5] for obtaining the frequency response of the transformed model. That paper was the first to present the basics of the frequency response algorithm presented here and introduced the term “Automated Multilevel Substructuring.” The AMLS method has continued to improve in performance $[6]$ and acquire new capabilities $[7, 8]$ . 

## 1.2 An Overview of the AMLS Method

The Automated Multilevel Substructuring method for solving frequency response problems may be thought of as having five phases. The process is outlined in the flowchart in Figure 1.1. In the first phase, the finite element mass and stiffness matrices are formed as well as any other model matrices (such as damping) or vectors (such as loads). Though any finite element software package could be used, we have chosen to use MSC.Nastran to generate the finite element data. Because this phase does not fall within the scope of our research, it will not be discussed more than to mention what output is required. In the second phase, the matrices are automatically reordered using a nested dissection algorithm and partitioned into substructures. This phase also falls outside the scope of this dissertation, but will be discussed in some detail because it has some impact on the later phases. The third phase projects the original finite element model onto a more efficient substructure modal subspace. This transformation, based on a multilevel implementation of component mode synthesis, represents a model reduction since the substructure modal subspace is truncated. The fourth phase finds the eigenpairs of the reduced model that approximate the global eigensolution and are necessary for efficiently computing the frequency response. The eigensolver in Phase 4 is designed specifically for this application. The fifth phase uses the transformed model from Phase 3 and the eigensolution from Phase 4 to compute the frequency response. Two implementations of Phase 5 will be discussed: one using mode superposition, the other using an iterative technique. The iterative technique, like the eigensolver in Phase 4, is designed to take advantage of the particular properties of the AMLS transformation. 

## 1.3 Review of Competing Methods

Many methods have been used to solve the frequency response problem. For very large problems, most of these methods can only be used on the most powerful computers. In this section, the leading methods will be briefly described along with their advantages and disadvantages for addressing frequency response problems for very large structure models. Each of these methods is described in greater detail in Chapter 2. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-04/81444ff4-0f7a-4a8b-b5ad-3675b5990a09/6eef52246aa38be6bd4c0615725ec006c492b96f9c1f07a4eabb0d9defa4a9ae.jpg)



Figure 1.1: Flowchart of AMLS method


The direct method for solving a linear system of equations requires a factorization of the matrix $A(\omega)$ , where A is given by 

$$
A (\omega) = \left[ K (1 + i \gamma) + i \omega B + i \Gamma - \omega^ {2} M \right]\tag{1.3.1}
$$

For example, $A$ may be factored as $A = LDL^T$ , where $D$ is a diagonal matrix and $L$ is a lower triangular matrix, if $A$ and all of its leading principal minor submatrices are nonsingular [9]. Given this factorization, the solution can be obtained by solving the series of equations 

$$
\begin{array}{c} L Z = F \\ D Y = Z \\ L ^ {T} X = Y \end{array}\tag{1.3.2}
$$

The advantage of this method is its simplicity. A matrix reordering can be done to reduce the computational effort required. Sparsity can be exploited. As long as a stable solver is used, this method can be expected to provide the greatest accuracy possible. This method is also likely to be the most computationally expensive method because a factorization is required at every frequency at which a solution is desired. In industry, this method is implemented for massively parallel computer architectures. It scales well in parallel and can actually compete with some of the other methods in this section. On a single processor, this method is not a realistic option for solving large frequency response problems. One of the most powerful direct solvers is currently available as part of Boeing Computer Services' library of mathematical software BCSLIB-EXT. 

Domain decomposition represents a broad category of methods for solving systems of equations $[10]$ . These methods are intended for parallel implementation and include both direct and iterative algorithms. In domain decomposition, the model is divided into many subdomains. These subdomains may be overlapping or nonoverlapping. Direct methods use nonoverlapping subdomains and are equivalent to the sparse direct methods mentioned in the last paragraph. The strengths of these methods are in their ability to take advantage of sparsity and their scalability in parallel implementation. For large frequency response problems, these methods have similar liabilities to direct methods. Domain decomposition methods must do almost the same amount of work for every frequency at which a solution is required. 

The mode superposition method depends on the assumption that the frequency response of the structure can be computed as a linear combination of its natural modes of vibration and that a subset of those modes can provide an adequate approximation of the response. First, the generalized eigenvalue problem 

$$
K \Phi = M \Phi \Lambda\tag{1.3.3}
$$

is solved, where $\Phi$ is the matrix of eigenvectors and $\Lambda$ is the diagonal matrix of eigenvalues. The eigenvectors are truncated based on some cutoff frequency. The system is then projected onto the modal subspace as 

$$
\Phi^ {T} A (\omega) \Phi Z (\omega) = \Phi^ {T} F (\omega)\tag{1.3.4}
$$

where $X$ is approximated by $\Phi Z$ . If $B$ and $\Gamma$ are null, the problem is diagonalized and can be easily solved. For large problems, $\Phi$ may represent less than 1% of the total possible eigenvectors. Typically, a Lanczos method [11] is used to solve the eigenvalue problem. One of the most widely used implementations, the shifted block Lanczos method, can be found in the Boeing Computer Services' software library [12]. The main advantage of the mode superposition method is that the problem is projected onto a much smaller subspace. Another benefit of this method is that the modes of vibration are often of interest in structural dynamic analysis. They are obtained as a by-product of this method. This method can be implemented in parallel, though it is not capable of the scalability of the direct or domain decomposition methods. One disadvantage is that the accuracy of the solution cannot be established a priori. In other words, one cannot know what level of accuracy can be achieved with a certain truncated set of modes until after the modes are computed. Thus, the cutoff frequency on the eigenvalue problem is set heuristically. A greater problem may be that the projection of the sparse matrix $A$ onto the modal subspace creates a fully populated matrix in the case of general damping. Another disadvantage is the sheer size of the data involved. The eigenvectors alone could occupy tens of gigabytes of disk space. If those vectors must be read in multiple times for every frequency, it could create a bottleneck in I/O. Despite these issues, mode superposition is the most commonly used method for solving very large frequency response problems. 