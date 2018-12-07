import torch
import Assignment7Support

discountRate = 0.98          # Controls the discount rate for future rewards -- this is gamma from 13.10
actionProbabilityBase = 1.8  # This is k from the P(a_i|s) expression from section 13.3.5 and influences how random exploration is
randomActionRate = 0.01      # Percent of time the next action selected by GetAction is totally random
learningRateScale = 0.01     # Should be multiplied by visits_n from 13.11.
trainingIterations = 20000
numActions = 5

observations = 4
actions = [0, 1]

class QLearning(torch.nn.Module):
    def __init__(self, stateSpaceShape=Assignment7Support.MountainCarStateSpaceShape(), numActions=numActions, discountRate=discountRate):
        super(QLearning, self).__init__()

        self.rewards, self.Q = np.zeros([observations, actions.size()]) # we have 4 observations and 2 actions [[0, 0], [0, 0], [0, 0], [0, 0]]

    def GetAction(self, currentState, learningMode=True, randomActionRate=randomActionRate, actionProbabilityBase=actionProbabilityBase):
        if random.uniform(0, 1) < randomActionRate:
            return random.choice(actions)
        else:
            return getProbabilityActionGivenState(currentState, k) # Exploit learned values

#Uses the formula in section 13.3.5 to decide which action to take P(a_i | s). 
#Support the parameter k to modify this expression (k=e is a good start; also consider values in the range 1.01 - 1.5).

    def getProbabilityActionGivenState(self, state, k = math(e)):
        probabalities = np.zeros([actions.size()])

        totalSumOfActions = sum(exp(k, Q[action[i]]) for i in probabalities)
        probabilities = [(exp(k, Q[state, action[i]])) / totalSumOfActions for i in probabalities]

        return np.argmax(probabilities)

    def ObserveAction(self, oldState, action, newState, reward, learningRateScale=learningRateScale):
        Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])

        Q[s, a] = (1 - alpha_n) * Q[oldState,previousAction] + alpha_n * (learningRateScale + argmax(Q[newState, nextAction]))

    def alpha_n(self, learningRateScale=learningRateScale):
        return 1 / (1 + visit_n * learningRateScale)  
