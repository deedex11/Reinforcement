class QUnitTest(unittest.TestCase):
    def test_1(self):
        qlearner = QLearning([2,2], 2, 0.9) # stateSpaceShape=Assignment7Support.MountainCarStateSpaceShape(), numActions=numActions, discountRate=discountRate

        print("action 1")
        qlearner.ObserveAction([0,0], 1, [0,1], 1, learningRateScale = 1.0) #oldState, action, newState, reward, learningRateScale=learningRateScale
        print(qlearner.qTable)
        print("visit")
        print(qlearner.visitCount) 

        print("action 2")
        qlearner.ObserveAction([0,0], 1, [0,1], 1, learningRateScale = 1.0)
        print(qlearner.qTable)
        print("visit")
        print(qlearner.visitCount)