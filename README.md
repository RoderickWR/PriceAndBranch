# PriceAndBranch
Implements a Dantzig-Wolfe Decomposition to solve a generic #m machines #n jobs Flow Shop scheduling problem. 
Pricing problems are defined per machine. Patterns either store the start and completion times for the jobs (branch offsetVariables) or the precendence information (integerPatterns). The master problem is initialized with arbitrary, sub-optimal patterns. The price-and-branch method only generates new patterns at the root node. 
