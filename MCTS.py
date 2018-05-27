import math
import numpy as np

EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.QValue = {}  # stores Q values for s,a (as defined in the paper)
        self.NStateAction = {}  # stores #times edge s,a was visited
        self.NState = {}  # stores #times board s was visited
        self.nnetPolicy = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.validMoves = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        boardState = self.game.stringRepresentation(canonicalBoard)
        counts = []
        for action in range(self.game.getActionSize()):
            if(boardState, action) in self.NStateAction:
                counts.append(self.NStateAction[(boardState, action)])
            else:
                counts.append(0)

        if temp == 0:
            bestAction = np.argmax(counts)
            moveProbabilities = [0] * len(counts)
            moveProbabilities[bestAction] = 1
            return moveProbabilities

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts))for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        boardState = self.game.stringRepresentation(canonicalBoard)

        # check if the boardstate has been seen before
        if boardState not in self.Es:
            # note we've seen this board before
            self.Es[boardState] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[boardState] != 0:  # game has ended
            return -self.Es[boardState]

        if boardState not in self.nnetPolicy:
            # leaf node
            self.nnetPolicy[boardState], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.nnetPolicy[boardState] = self.nnetPolicy[boardState] * valids  # only want valid moves
            # check if move vector is normalised

            self.nnetPolicy[boardState] /= np.sum(self.nnetPolicy[boardState])  # renormalize
            self.validMoves[boardState] = valids
            self.NState[boardState] = 0
            return -v

        validMoves = self.validMoves[boardState]
        curBest = -float('inf')
        bestAction = 1

        # find which action is best
        for action in range(self.game.getActionSize()):
            if validMoves[action]:
                if (boardState, action) in self.QValue:
                    u = self.QValue[(boardState, action)] + self.args.cpuct * self.nnetPolicy[boardState][
                        action] * math.sqrt(self.NState[boardState]) / (1 + self.NStateAction[(boardState, action)])
                else:
                    u = self.args.cpuct * self.nnetPolicy[boardState][action] * math.sqrt(self.NState[boardState] + EPS)
                if u > curBest:
                    curBest = u
                    bestAction = action

        nextState, nextPlayer = self.game.getNextState(canonicalBoard, 1, bestAction)
        nextState = self.game.getCanonicalForm(nextState, nextPlayer)

        v = self.search(nextState)

        if (boardState, bestAction) in self.QValue:
            self.QValue[(boardState, bestAction)] = (self.NStateAction[(boardState, bestAction)] * self.QValue[
                (boardState, bestAction)] + v) / (self.NStateAction[(boardState, bestAction)] + 1)
            self.NStateAction[(boardState, bestAction)] += 1
        else:
            self.QValue[(boardState, bestAction)] = v
            self.NStateAction[(boardState, bestAction)] = 1

        self.NState[boardState] += 1
        return -v
