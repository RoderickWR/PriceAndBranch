\ Model Bin Packing Price and Branch
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  makespan
Subject To
 convexityOnMachine(0): lambda(0,0) + lambda(1,0) + lambda(2,0))
   + lambda(3,0)) + lambda(4,0)) = 1
 convexityOnMachine(1): lambda(0,1) + lambda(1,1) + lambda(2,1))
   + lambda(3,1)) + lambda(4,1)) = 1
 R2: 10 lambda(1,0) - start(0,0) + 7 lambda(2,0)) + lambda(3,0))
   + 99 lambda(4,0)) = 0
 R3: 7 lambda(0,0) + 11 lambda(1,0) - finish(0,0) + 8 lambda(2,0))
   + 8 lambda(3,0)) + 100 lambda(4,0)) = 0
 R4: 7 lambda(0,0) + 12 lambda(1,0) - start(0,1) = 0
 R5: 8 lambda(0,0) + 19 lambda(1,0) - finish(0,1) + 7 lambda(2,0))
   + lambda(3,0)) + 7 lambda(4,0)) = 0
 interMachine(0,0): finish(0,0) - start(1,0) <= 0
 interMachine(0,1): finish(0,1) - start(1,1) <= 0
 startFinish(0,0): start(0,0) - finish(0,0) <= -7
 startFinish(0,1): start(0,1) - finish(0,1) <= -1
 R10: 10 lambda(1,1) - start(1,0) + 7 lambda(2,1)) + lambda(3,1))
   + 99 lambda(4,1)) = 0
 R11: 7 lambda(0,1) + 11 lambda(1,1) - finish(1,0) + 8 lambda(2,1))
   + 8 lambda(3,1)) + 100 lambda(4,1)) = 0
 R12: 7 lambda(0,1) + 12 lambda(1,1) - start(1,1) = 0
 R13: 8 lambda(0,1) + 19 lambda(1,1) - finish(1,1) + 7 lambda(2,1))
   + lambda(3,1)) + 7 lambda(4,1)) = 0
 startFinish(1,0): start(1,0) - finish(1,0) <= -1
 startFinish(1,1): start(1,1) - finish(1,1) <= -7
 makespanConstrMachine(0): - finish(1,0) + makespan >= 0
 makespanConstrMachine(1): - finish(1,1) + makespan >= 0
Bounds
 lambda(0,0) <= 1
 lambda(0,1) <= 1
 lambda(1,0) <= 1
 lambda(1,1) <= 1
 start(0,0) <= 100
 finish(0,0) <= 100
 start(0,1) <= 100
 finish(0,1) <= 100
 start(1,0) <= 100
 finish(1,0) <= 100
 start(1,1) <= 100
 finish(1,1) <= 100
 lambda(2,0)) <= 1
 lambda(2,1)) <= 1
 lambda(3,0)) <= 1
 lambda(3,1)) <= 1
 lambda(4,0)) <= 1
 lambda(4,1)) <= 1
End
