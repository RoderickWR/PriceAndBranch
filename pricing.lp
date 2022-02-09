\ Model Pricing
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 startFinish(0): start(0) - finish(0) = -1
 startFinish(1): start(1) - finish(1) = -3
 startFinish(2): start(2) - finish(2) = -7
 startFinish(3): start(3) - finish(3) = -10
 precedence(0): x(0,1) + x(1,0) = 1
 precedence(0): x(0,2) + x(2,0) = 1
 precedence(0): x(0,3) + x(3,0) = 1
 precedence(1): x(0,1) + x(1,0) = 1
 precedence(1): x(1,2) + x(2,1) = 1
 precedence(1): x(1,3) + x(3,1) = 1
 precedence(2): x(0,2) + x(2,0) = 1
 precedence(2): x(1,2) + x(2,1) = 1
 precedence(2): x(2,3) + x(3,2) = 1
 precedence(3): x(0,3) + x(3,0) = 1
 precedence(3): x(1,3) + x(3,1) = 1
 precedence(3): x(2,3) + x(3,2) = 1
 finishStart(0): - start(0) + finish(0) + 100 x(0,0) <= 100
 finishStart(0): finish(0) - start(1) + 100 x(0,1) <= 100
 finishStart(0): finish(0) - start(2) + 100 x(0,2) <= 100
 finishStart(0): finish(0) - start(3) + 100 x(0,3) <= 100
 finishStart(1): - start(0) + finish(1) + 100 x(1,0) <= 100
 finishStart(1): - start(1) + finish(1) + 100 x(1,1) <= 100
 finishStart(1): finish(1) - start(2) + 100 x(1,2) <= 100
 finishStart(1): finish(1) - start(3) + 100 x(1,3) <= 100
 finishStart(2): - start(0) + finish(2) + 100 x(2,0) <= 100
 finishStart(2): - start(1) + finish(2) + 100 x(2,1) <= 100
 finishStart(2): - start(2) + finish(2) + 100 x(2,2) <= 100
 finishStart(2): finish(2) - start(3) + 100 x(2,3) <= 100
 finishStart(3): - start(0) + finish(3) + 100 x(3,0) <= 100
 finishStart(3): - start(1) + finish(3) + 100 x(3,1) <= 100
 finishStart(3): - start(2) + finish(3) + 100 x(3,2) <= 100
 finishStart(3): - start(3) + finish(3) + 100 x(3,3) <= 100
Bounds
 start(0) <= 100
 finish(0) <= 100
 start(1) <= 100
 finish(1) <= 100
 start(2) <= 100
 finish(2) <= 100
 start(3) <= 100
 finish(3) <= 100
Binaries
 x(0,0) x(0,1) x(0,2) x(0,3) x(1,0) x(1,1) x(1,2) x(1,3) x(2,0) x(2,1)
 x(2,2) x(2,3) x(3,0) x(3,1) x(3,2) x(3,3)
End
