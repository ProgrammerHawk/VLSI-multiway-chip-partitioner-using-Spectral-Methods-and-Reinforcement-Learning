# VLSI-multiway-chip-partitioner-using-Spectral-Methods-and-Reinforcement-Learning
_This is a k-way partitioner with constraints to be used in a multilevel partitioning clustered level (Initial Partitioning Phase). To make it deterministic, spectral partitioning along with other methods like FM, reinforcement learning etc are used._  

_Acknowledgements_: I would like to thank __Cadence Design Systems__ for providing me mentorship on this project along with some sample clustered hypergraphs and sample FPGA base board connected system configuration.  
#  

__1. Coarsening Phase__

• __WHY:__ Consecutively reduce the size of the initial large netlist.  
• __HOW:__ Merge clusters of highly connected vertices (cells) into single macro-vertices based on grouping metrics like Edge Coarsening (EC) or Hyperedge Coarsening (HEC).  
• __OUTPUT:__ A sequence of progressively smaller, simpler hypergraphs that preserve the fundamental global structure of the original netlist.  

#    

__2. Initial Partitioning Phase (My Contribution)__    

• __WHY:__ Create a baseline partition on the smallest, most condensed hypergraph.  
• __HOW:__ Apply faster spectral methods that naturally forms clusters of highly connected components in the euclidean space. Improve it with FM based refinement, RL methods etc. Must be comparable to random seed based partition. Direct k-way is required. Given a max "k", use optimal number of partitions less than "k" based on capacity constraints.    
• __OUTPUT:__ A balanced, high-quality direct k-way on the coarsest graph, establishing a starting point with low routing cut sizes satisfying constraints. HAS TO BE DETERMINISTIC.  

#  

__3. Uncoarsening & Refinement Phase__  

• __WHY:__ Progressively project the partitions back onto the original full-scale netlist while improving the cut quality.  
• __HOW:__ Undo the vertex clustering step-by-step. At each intermediate level, local vertex-swapping refinement algorithms like the Fiduccia-Mattheyses (FM) variant ia deployed to fix up cell boundary cut lines.  
• __OUTPUT:__ The fully optimized, multiway partitioned netlist matching the target hardware block or multi-chip constraints.  

#  
#  
__COMPILE AND RUN__  

To compile the basic version: Use basic_compile.sh. 
This will create an executable called "spectral_solve". 


