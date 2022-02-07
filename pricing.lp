\ Model Pricing
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 startFinish(0): start(0) - finish(0) = -1
 startFinish(1): start(1) - finish(1) = -7
 precedence(0): x(0,1) + x(1,0) = 1
 fixAtZero(0): start(0) + 50 x(0,1) <= 50
 precedence(1): x(0,1) + x(1,0) = 1
 fixAtZero(1): start(1) + 50 x(1,0) <= 50
 finishStart(0): - start(0) + finish(0) + 100 x(0,0) <= 100
 finishStart(0): finish(0) - start(1) + 100 x(0,1) <= 100
 finishStart(1): - start(0) + finish(1) + 100 x(1,0) <= 100
 finishStart(1): - start(1) + finish(1) + 100 x(1,1) <= 100
 R10: finish(0) <= 16
 R11: finish(1) <= 16
 R12: finish(0) <= 15
 R13: finish(1) <= 15
 R14: finish(0) <= 8
 R15: finish(1) <= 8
 R16: finish(0) <= 8
 R17: finish(1) <= 8
Bounds
 start(0) <= 100
 finish(0) <= 100
 start(1) <= 100
 finish(1) <= 100
Binaries
 x(0,0) x(0,1) x(1,0) x(1,1)
End
