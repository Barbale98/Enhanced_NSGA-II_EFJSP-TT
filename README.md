# Enhanced-NSGA-II-for-EFJSP-TT

This repository provides all the data from my master thesis "Enhanced GA for the Energy-efficient FJSP with transportation times"

The thesis can be found at the following link: 
https://www.researchgate.net/publication/382329155_Enhanced_Genetic_Algorithm_to_Solve_the_Energy-efficient_Flexible_Job_Shop_Scheduling_Problem

MyInstance folder contains information about Brandimarte's dataset and other instances used in my thesis

MILP.py contains the MILP formulation of the problem solved with GUROBI
NSGAII.py contains the original NSGA-II multi-objective algorithm
VNS-NSGAII.py algorithm extend NSGA-II algorithm with a VNS block
MDR algorithm both for makespan and energy is a novel algorithm developed to initialise the population mixing different dispatching rules
EVNS-NSGAII.py algorithm uses MDR to initialise part of the population and VNS+NSGA

