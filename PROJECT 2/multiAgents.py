# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]not using this approach

        "*** YOUR CODE HERE ***"
        #The approach I want to follow is to check if PACman position co-incindises with ghost, if it bad will mark as bad state
        #Will check the Closest Ghost and Food distances by Manhattan method to avoid ghosts and move towards food.
        #To stop running out of time, dynamic score is added to hval 
        #One thing I have observed is that PacMan wastes time in doing Stop action, I have tried to avoid this action in this updated code.
        
        #Avoiding STOP action to PACMAN from wasting time staying still
        if action == 'Stop':
            return -9999

        #Checking the Bad states
        for deathpos in newGhostStates:
            if newPos == deathpos.configuration.pos:
                return -9999 # returning negative infinity to indicae death state
        
        closestGhost= 9999 #Set to large value
        for ghost in newGhostStates:
            dist= util.manhattanDistance(newPos, ghost.configuration.pos)
            #print("distghost",dist)
            if dist< closestGhost:
                closestGhost= dist
        
        closestFood= 9999 #set to large value
        for food in newFood.asList():
            dist= util.manhattanDistance(newPos, food)
            #print("distfood",dist)
            if dist< closestGhost:
                closestFood= dist
            
        #return (closestGhost/closestFood+1e-6)
        #Returning the best value, closes the food higher the value, farther the food lesser value
        hval= closestGhost/ (closestFood+ 1e-6) #adding a small value to prevent division by zero(took help with this error)
        current_score= successorGameState.getScore() #to ensure Maximum Score dynamically
        #print("hval",hval)
        #print("score",current_score)
        return (current_score+hval)
    

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #My approach is to define 3 functions:
        
        #The MinMax Search involves with nodeValues whereas the Nodeaction wrt Nodevalue must be returned to the Evaluation Function.
        #Actions and Values are set as tuple to remember the states.
        #maxPAC() to find max value among its action, choice in the MAX layer
        #minGhost() to iterate over each ghost's legal action, choices in MIN layer
        #minmax() to search for calling best maxPAC or worst minGhost according to nodelayer and return to Evaluation Function when win/lose/terminal depth
        
        #Since max starts the game, the first action is sent to PACman
        MinMaxSearch= self.maxPAC(gameState, agentIndex=0, depth=0)
        #We just need the action to be sent to Evaluation Function and Value can be neglected
        Game= MinMaxSearch[0]
        return Game
    
    def maxPAC(self, gameState, agentIndex, depth):
        #the max action is set to a very low value
        maxAction= ("max",-9999)
        
        #list of all pacman legal actions
        legalActions= gameState.getLegalActions(0)
        #all the possible pacman moves
        for action in legalActions:
            #expanding towards the minlayer node for specific pacman action
            nextNode= gameState.generateSuccessor(agentIndex, action)
            #which ghost agent to consider to this action 
            ghost= (agentIndex+1)% gameState.getNumAgents()
            #Calling minmax search to calculate value
            nextValue= self.minmax(nextNode, ghost, depth+1)
            #Making a tuple of action-value, since just sending the actions to getAction() is always failing by the auto-grader
            #I have observed that my tree while backtracking, nodes with same value in the minlayer while comparing worst value when not linked with action is not giving results as expected
            #Tuple of actions and the value of state
            nextAction= (action, nextValue)
            #print("max",nextAction)
            #Updating the Maximum value of the nodes to be the Max action
            if nextAction[1] > maxAction[1]:
                maxAction= nextAction
        #returning the action related to the maximum value in MAX layer to minmax()
        return maxAction
    
    def minGhost(self, gameState, agentIndex, depth):
        #the min action is set to a very high value
        minAction= ("min", 9999)
        
        #list of possible actions by ghosts
        legalActions= gameState.getLegalActions(agentIndex)
        #all the ghostagent possible moves over different ghost agents
        for action in legalActions:
            #expanding the max layer wrt the present ghost move
            nextNode= gameState.generateSuccessor(agentIndex, action)
            #min layer iteration for other ghost agents
            nextghost= (agentIndex+1)% gameState.getNumAgents()
            #Calling minmax search to calculate value of state
            nextValue= self.minmax(nextNode, nextghost, depth+1)
            #Action-Value pairs
            nextAction= (action, nextValue)
            #print("min",nextAction)
            #Updating the Maximum value of the nodes to be the Min action
            if nextAction[1] < minAction[1]:
                minAction= nextAction
        #returning the action related to the minimum value in Min layer to minmax()
        return minAction
    
    def minmax(self, gameState, agentIndex, depth):
        #Check if we've reached the maximum depth or a terminal state
        #print("depth",depth)
        #The max depth is no.of agents remaining * depth
        maxDepth= self.depth*gameState.getNumAgents()
        if depth == (maxDepth) or gameState.isWin() or gameState.isLose():
            #Game results are sent to evaluation function
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            #Pacman is always index 0, so we call the maxPAC()
            return self.maxPAC(gameState,agentIndex,depth)[1]
        else:
            #Any other agent is a ghost, so we call the minGhost()
            return self.minGhost(gameState,agentIndex,depth)[1]        
        
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #approach is similar to minmax search
        #alphaBeta() to search for calling best maxPAC or worst minGhost according to the node layer and return to Eval Func
        #Since max starts the game, the first action is sent to PACman
        AlphaBetaSearch= self.maxPAC(gameState, agentIndex=0, depth=0, alpha= -float("inf"),beta= float("inf"))
        #We just need the action to be sent to Evaluation Function and Value can be neglected
        Game= AlphaBetaSearch[0]
        return Game
    
    def maxPAC(self, gameState, agentIndex, depth, alpha, beta):
        #the max action is set to a very low value
        maxAction= ("max",-9999)
        
        #list of all pacman legal actions
        legalActions= gameState.getLegalActions(0)
        #all the possible pacman moves
        for action in legalActions:
            #expanding towards the minlayer node for specific pacman action
            nextNode= gameState.generateSuccessor(agentIndex, action)
            #which ghost agent to consider to this action 
            ghost= (agentIndex+1)% gameState.getNumAgents()
            #Calling minmax search to calculate value
            nextValue= self.alphabeta(nextNode, ghost, depth+1,alpha,beta)
            #Making a tuple of action-value, since just sending the actions to getAction() is always failing by the auto-grader
            #I have observed that my tree while backtracking, nodes with same value in the minlayer while comparing worst value when not linked with action is not giving results as expected
            #Tuple of actions and the value of state
            nextAction= (action, nextValue)
            #Updating the Maximum value of the nodes to be the Max action
            if nextAction[1] > maxAction[1]:
                maxAction= nextAction
                
            #Alpha-Beta Pruning
            #If maxAction is more than beta, we can stop expanding further
            if maxAction[1] > beta:
                #returning the action related to the maximum value in MAX layer to minmax()
                #print("Prune max", action)
                return maxAction
            #update alpha
            else: alpha= max(alpha,maxAction[1])
        
        #returning the action related to the maximum value in MAX layer to minmax()
        return maxAction
    
    def minGhost(self, gameState, agentIndex, depth, alpha, beta):
        #the min action is set to a very high value
        minAction= ("min", 9999)
        
        #list of possible actions by ghosts
        legalActions= gameState.getLegalActions(agentIndex)
        #all the ghostagent possible moves over different ghost agents
        for action in legalActions:
            #expanding the max layer wrt the present ghost move
            nextNode= gameState.generateSuccessor(agentIndex, action)
            #min layer iteration for other ghost agents
            nextghost= (agentIndex+1)% gameState.getNumAgents()
            #Calling minmax search to calculate value of state
            nextValue= self.alphabeta(nextNode, nextghost, depth+1,alpha,beta)
            #Action-Value pairs
            nextAction= (action, nextValue)
            #Updating the Maximum value of the nodes to be the Min action
            if nextAction[1] < minAction[1]:
                minAction= nextAction
                
            #Alpha-Beta Pruning
            #If maxAction is less than alpha, we can stop expanding further
            if minAction[1] < alpha:
                #returning the action related to the maximum value in MAX layer to minmax()
                #print("Prune min", action)
                return minAction
            else:
                #update beta
                beta= min(beta, minAction[1])       
            
        #returning the action related to the minimum value in Min layer to minmax()
        return minAction
    
    def alphabeta(self, gameState, agentIndex, depth,alpha,beta):
        #Check if we've reached the maximum depth or a terminal state
        #The max depth is no.of agents remaining * depth
        maxDepth= self.depth*gameState.getNumAgents()
        if depth == (maxDepth) or gameState.isWin() or gameState.isLose():
            #Game results are sent to evaluation function
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            #Pacman is always index 0, so we call the maxPAC()
            return self.maxPAC(gameState,agentIndex,depth,alpha,beta)[1]
        else:
            #Any other agent is a ghost, so we call the minGhost()
            return self.minGhost(gameState,agentIndex,depth,alpha,beta)[1] 
        
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    #I see that we are making decisions based at present state
    #Next state is not considered to make a betterEvaluationFunction
    #I have tried my basic EvaluationFunction() in this method without future actions.
    #My approach is to follow the basic manhattan distance to closest food and closest ghost    
    "*** YOUR CODE HERE ***" 
    currentPos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    remainingFood= Food.asList()
    ghostPos= currentGameState.getGhostPositions()
    #Using my basic EvaluationFunction() approach
    
    closestFood= 9999 #Set to large value
    for foodnear in remainingFood:
        dist= util.manhattanDistance(currentPos, foodnear)
        closestFood= min(closestFood,dist)
    
    closestGhost= 9999 #Set to large value
    for ghost in ghostPos:
        dist= util.manhattanDistance(currentPos, ghost)
        closestGhost= min(closestGhost, dist)
        
    #Passing the heuristic conditions
    #Adding 1e-6 to make sure the divisor is not 0
    #This ghost:food ratio letts me how good the state is
    hval= closestGhost/ (closestFood+ 1e-6)
    score= currentGameState.getScore()
    return hval+score
    

# Abbreviation
better = betterEvaluationFunction
