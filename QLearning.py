import torch
import Assignment7Support
import math
import random
from collections import defaultdict

discountRate = 0.98          # Controls the discount rate for future rewards -- this is gamma from 13.10
actionProbabilityBase = 1.8  # This is k from the P(a_i|s) expression from section 13.3.5 and influences how random exploration is
randomActionRate = 0.01      # Percent of time the next action selected by GetAction is totally random
learningRateScale = 0.01     # Should be multiplied by visits_n from 13.11.
trainingIterations = 20000

observations = 4 # That means that a state looks like [x, x, x, x] where the numbers can be anything
actions = [0, 1]

random.seed(100)

class QLearning(torch.nn.Module):
    def __init__(self, stateSpaceShape=Assignment7Support.MountainCarStateSpaceShape(), numActions=2, discountRate=discountRate):
        super(QLearning, self).__init__()
        self.Q = {}
        self.Q = defaultdict(lambda:[0, 0], self.Q)
        self.visits = {}
        self.visits = defaultdict(lambda:[0, 0], self.visits)

    def GetAction(self, currentState, learningMode=True, randomActionRate=randomActionRate, actionProbabilityBase=actionProbabilityBase):
        #print("Get action")
        currentState = tuple(currentState)
        action = 0
        if random.uniform(0, 1) < randomActionRate:
            #print("Taking random action")
            action = random.choice(actions)
        else:
            action = self.getProbabilityActionGivenState(currentState, actionProbabilityBase) # Exploit learned values

        #print("This is the action being taken: " + str(action))
        #print((self.visits[currentState]))
        #print(str(self.visits[currentState][action]))
        self.visits[currentState][action] = self.visits[currentState][action] + 1
       
        return action

#Uses the formula in section 13.3.5 to decide which action to take P(a_i | s). 
#Support the parameter k to modify this expression (k=e is a good start; also consider values in the range 1.01 - 1.5).

    def getProbabilityActionGivenState(self, state, k = math.e):
        probabalities = [0 for i in range(len(actions))]

        #print("This is the Q[state]: " + str(self.Q[state]))
        totalSumOfActions = sum(math.exp(self.Q[state][actions[i]]) for i in range(len(actions)))
        #print("totalSumOfActions: " + str(totalSumOfActions))
        probabilities = [(math.exp(self.Q[state][actions[i]])) / totalSumOfActions for i in range(len(actions))]

        #print("These are the probabilities: " + str(probabilities))
        #print("This is the argmax probabiltiy taken: " + str(np.argmax(probabilities)))
        return probabilities.index(max(probabilities))

    def ObserveAction(self, oldState, action, newState, reward, learningRateScale=learningRateScale):
        oldState = tuple(oldState)
        newState = tuple(newState)
        alpha = self.alpha_n(oldState, action, learningRateScale)
        #print("This is alpha: " + str(alpha))
        #print("Q[state][action]: " + str(self.Q[newState]) + "with max " + str(np.max(self.Q[newState])))

        self.Q[oldState][action] = ((1 - alpha) * self.Q[oldState][action]) + (alpha * (reward + (discountRate * max(self.Q[newState]))))

    def alpha_n(self, state, action, learningRateScale=learningRateScale):
        return 1 / (1 + (self.visits[state][action] * learningRateScale))  
