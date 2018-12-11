import torch
import Assignment7Support
import math
import random
from collections import defaultdict

discount = 0.98          # Controls the discount rate for future rewards -- this is gamma from 13.10
actionBase = math.e  # This is k from the P(a_i|s) expression from section 13.3.5 and influences how random exploration is
randomRate = 0.01      # Percent of time the next action selected by GetAction is totally random
learningRateScale = 0.01     # Should be multiplied by visits_n from 13.11.
trainingIterations = 20000

actions = [0, 1, 2]

#random.seed(100)

class QLearning(torch.nn.Module):
    def __init__(self, stateSpaceShape=Assignment7Support.MountainCarStateSpaceShape(), numActions=2, discountRate=discount):
        super(QLearning, self).__init__()
        self.Q = {}
        self.Q = defaultdict(lambda:[0 for i in range(numActions)], self.Q)
        self.visits = {}
        self.visits = defaultdict(lambda:[0 for i in range(numActions)], self.visits)
        #print("Discount rate set to: " + str(discountRate))
        self.discountRate = discountRate

    def GetAction(self, currentState, learningMode=True, randomActionRate=randomRate, actionProbabilityBase=actionBase):
        #print("Get action")
        #print("Action: " + str(actionProbabilityBase))
        currentState = tuple(currentState)
        probabilities = self.getProbability(currentState, actionProbabilityBase)

        if learningMode:
            if random.uniform(0, 1) < randomActionRate:
                action = random.choice(actions)
            else:
                action = random.choices(actions, probabilities)[0]
                #action = int(np.random.choice(actions, 1, probabilities))
        else:
            action = probabilities.index(max(probabilities))

        #print("This is the action being taken: " + str(action))
        #print((self.visits[currentState]))
        #print(str(self.visits[currentState][action]))
        self.visits[currentState][action] = self.visits[currentState][action] + 1
       
        return action

#Uses the formula in section 13.3.5 to decide which action to take P(a_i | s). 
#Support the parameter k to modify this expression (k=e is a good start; also consider values in the range 1.01 - 1.5).

    def getProbability(self, state, actionProbabilityBase):
        totalSumOfActions = sum([actionProbabilityBase**(self.Q[state][actions[i]]) for i in range(len(actions))])
        #print("totalSumOfActions: " + str(totalSumOfActions))
        return [(actionProbabilityBase**(self.Q[state][actions[i]])) / totalSumOfActions for i in range(len(actions))]


    def ObserveAction(self, oldState, action, newState, reward, learningRateScale=learningRateScale):
        oldState = tuple(oldState)
        newState = tuple(newState)
        alpha = self.alpha_n(oldState, action, learningRateScale)
        #print("This is alpha: " + str(alpha))
        #print("Q[state][action]: " + str(self.Q[newState]) + "with max " + str(max(self.Q[newState])))
        #print("Discount rate: " + str(self.discountRate))
        self.Q[oldState][action] = ((1 - alpha) * self.Q[oldState][action]) + (alpha * (reward + self.discountRate * max(self.Q[newState])))

    def alpha_n(self, state, action, learningRateScale=learningRateScale):
        return 1 / (1 + (self.visits[state][action] * learningRateScale))
