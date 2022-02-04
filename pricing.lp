\ Model Pricing
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 0.125 start(0) + 0.125 finish(0) - 0.125 start(1) + 0.125 finish(1)
Subject To
 startFinish(0): start(0) - finish(0) = -1
 startFinish(1): start(1) - finish(1) = -7
 precedence(0): x(0,1) + x(1,0) = 1
 precedence(1): x(0,1) + x(1,0) = 1
 finishStart(0): - start(0) + finish(0) + 100 x(0,0) <= 100
 finishStart(0): finish(0) - start(1) + 100 x(0,1) <= 100
 finishStart(1): - start(0) + finish(1) + 100 x(1,0) <= 100
 finishStart(1): - start(1) + finish(1) + 100 x(1,1) <= 100
Bounds
 start(0) <= 100
 finish(0) <= 100
 start(1) <= 100
 finish(1) <= 100
Binaries
 x(0,0) x(0,1) x(1,0) x(1,1)
End
