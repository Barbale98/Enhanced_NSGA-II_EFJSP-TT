# Enhanced-NSGA-II-for-EFJSP-TT

This repository provides all the data from my master thesis "Enhanced GA for the Energy-efficient FJSP with transportation times"

MyInstance folder contains information about
>Brandimarte's dataset
>Instances used in my Thesis

MILP.py contains the MILP formulation of the problem solved with GUROBI
NSGAII.py contains the original NSGA-II multi-objective algorithm
VNSNSGAII.py algorithm extend NSGA-II algorithm with a VNS block
MDR algorithm both for makespan and energy is a novel algorithm developed to solve FJSP mixing different dispatching rules, SPT, NOR and DBA
EVNSNSGAII.py algorithm uses MDR to initialise part of the population and optain near optimal solutions
