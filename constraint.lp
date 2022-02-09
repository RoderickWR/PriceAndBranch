\ Model Bin Packing Price and Branch
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  makespan
Subject To
 convexityOnMachine(0): lambda(0,0) + lambda(1,0) + lambda(2,0)) = 1
 convexityOnMachine(1): lambda(0,1) + lambda(1,1) + lambda(2,1)) = 1
 finishStart(0,0,1): 100 lambda(0,0) + finish(0,0) - start(0,1)
   + lambda(2,0)) <= 100
 finishStart(0,0,2): finish(0,0) - start(0,2) <= 100
 finishStart(0,0,3): 100 lambda(1,0) + finish(0,0) - start(0,3) <= 100
 finishStart(0,1,0): - start(0,0) + finish(0,1) <= 100
 finishStart(0,1,2): 100 lambda(0,0) + 100 lambda(1,0) + finish(0,1)
   - start(0,2) <= 100
 finishStart(0,1,3): finish(0,1) - start(0,3) <= 100
 finishStart(0,2,0): 100 lambda(1,0) - start(0,0) + finish(0,2)
   + lambda(2,0)) <= 100
 finishStart(0,2,1): - start(0,1) + finish(0,2) + lambda(2,0)) <= 100
 finishStart(0,2,3): 100 lambda(0,0) + finish(0,2) - start(0,3) <= 100
 finishStart(0,3,0): 100 lambda(0,0) - start(0,0) + finish(0,3)
   + lambda(2,0)) <= 100
 finishStart(0,3,1): 100 lambda(1,0) - start(0,1) + finish(0,3)
   + lambda(2,0)) <= 100
 finishStart(0,3,2): - start(0,2) + finish(0,3) + lambda(2,0)) <= 100
 finishStart(1,0,1): 100 lambda(0,1) + finish(1,0) - start(1,1)
   + lambda(2,1)) <= 100
 finishStart(1,0,2): finish(1,0) - start(1,2) <= 100
 finishStart(1,0,3): 100 lambda(1,1) + finish(1,0) - start(1,3) <= 100
 finishStart(1,1,0): - start(1,0) + finish(1,1) <= 100
 finishStart(1,1,2): 100 lambda(0,1) + 100 lambda(1,1) + finish(1,1)
   - start(1,2) <= 100
 finishStart(1,1,3): finish(1,1) - start(1,3) <= 100
 finishStart(1,2,0): 100 lambda(1,1) - start(1,0) + finish(1,2)
   + lambda(2,1)) <= 100
 finishStart(1,2,1): - start(1,1) + finish(1,2) + lambda(2,1)) <= 100
 finishStart(1,2,3): 100 lambda(0,1) + finish(1,2) - start(1,3) <= 100
 finishStart(1,3,0): 100 lambda(0,1) - start(1,0) + finish(1,3)
   + lambda(2,1)) <= 100
 finishStart(1,3,1): 100 lambda(1,1) - start(1,1) + finish(1,3)
   + lambda(2,1)) <= 100
 finishStart(1,3,2): - start(1,2) + finish(1,3) + lambda(2,1)) <= 100
 interMachine(0,0): finish(0,0) - start(1,0) <= 0
 interMachine(0,1): finish(0,1) - start(1,1) <= 0
 interMachine(0,2): finish(0,2) - start(1,2) <= 0
 interMachine(0,3): finish(0,3) - start(1,3) <= 0
 startFinish(0,0): start(0,0) - finish(0,0) = -7
 startFinish(0,1): start(0,1) - finish(0,1) = -2
 startFinish(0,2): start(0,2) - finish(0,2) = -1
 startFinish(0,3): start(0,3) - finish(0,3) = -6
 startFinish(1,0): start(1,0) - finish(1,0) = -1
 startFinish(1,1): start(1,1) - finish(1,1) = -3
 startFinish(1,2): start(1,2) - finish(1,2) = -7
 startFinish(1,3): start(1,3) - finish(1,3) = -1
 makespanConstrMachine(0): - finish(1,0) + makespan >= 0
 makespanConstrMachine(1): - finish(1,1) + makespan >= 0
 makespanConstrMachine(2): - finish(1,2) + makespan >= 0
 makespanConstrMachine(3): - finish(1,3) + makespan >= 0
Bounds
 start(0,0) <= 100
 finish(0,0) <= 100
 start(0,1) <= 100
 finish(0,1) <= 100
 start(0,2) <= 100
 finish(0,2) <= 100
 start(0,3) <= 100
 finish(0,3) <= 100
 start(1,0) <= 100
 finish(1,0) <= 100
 start(1,1) <= 100
 finish(1,1) <= 100
 start(1,2) <= 100
 finish(1,2) <= 100
 start(1,3) <= 100
 finish(1,3) <= 100
Binaries
 lambda(0,0) lambda(0,1) lambda(1,0) lambda(1,1) lambda(2,0)) lambda(2,1))
End
