# search.py
# ---------
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
"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """
    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def genericSearch(problem, fringe):

    visited = set()
    totalPath = list()
    fringe.push((problem.getStartState(), list(), 0))
    while not fringe.isEmpty():
        currentState = fringe.pop()
        if problem.isGoalState(currentState[0]) == True:
            return currentState[1]
        if currentState[0] not in visited:
            for childNode, action, childCost in problem.getSuccessors(currentState[0]):
                    totalPath = currentState[1].copy()
                    totalPath.append(action)
                    totalCost = currentState[2] + childCost
                    fringe.push((childNode, totalPath, totalCost))
        visited.add(currentState[0])

    return None


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #I have mostly tried to implement the topics that were discussed in class lectures in the problem
    #I have searched online for function code template for 'Recursive Computation' (had this approach from lecture video) to include in CornersHeuristic to attain best path and reduce number of node expansions for extra credit.
    #I have tried to indicate the lines I took reference from if I have for any, on top of their implementations.
    #I have tried various data structure implementations for Search Algorithms and based on their total costs and node expansions, I have decided to go with the one that performs better
    
    #Priority queue initialization for DFS as a tree.
    #following the lecture where DFS should prioritize nodes with higher '-g' value in priority queue.
    #If using Stack: dfs_stack= util.Stack().
    #"lambda node: -len(node[1])" is the function that prioritizes nodes with higher '-depth' 
    dfs_queue= util.PriorityQueueWithFunction(lambda node: -len(node[1]))
    visited_nodes= []
    root_node= problem.getStartState()
    
    #Push start state and empty action into queue.
    dfs_queue.push((root_node,[]))
    
    #Loop as long as the Priority Queue is not empty.
    while not dfs_queue.isEmpty():
        current_node, current_action= dfs_queue.pop()
        #skip if state is already visited.
        if current_node in visited_nodes:
            continue
        
        #Add the current node as visited.
        visited_nodes.append(current_node)
        
        #Is Current node the Goal node?
        if problem.isGoalState(current_node):
            return current_action
        
        #Expand the child.
        children= problem.getSuccessors(current_node)
        
        for child_node in children:
            if child_node[0] not in visited_nodes:
                new_action= list(current_action)
                new_action.append(child_node[1])
                dfs_queue.push((child_node[0], new_action))
    
    #When no goal is found, we return empty.
    util.raiseNotDefined()
    
    #I have also used Stack as the data structure and the results were as follows:
    #The Priority Queue-based DFS had slightly better total costs for the tiny and medium maze while big maze had same cost with stack DFS.


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #Queue Initialization for BFS as a tree.
    #following the lecture where BFS should prioritize nodes with lower '+g' value in priority queue.
    #If using Queue :  bfs_queue= util.Queue()
    bfs_queue= util.PriorityQueueWithFunction(lambda node: len(node[1]))
    visited_nodes= []
    root_node= problem.getStartState()
    
    #Push start state and empty action into queue.
    bfs_queue.push((root_node,[]))
    
    #Loop as long as the Queue is not empty.
    while not bfs_queue.isEmpty():
        current_node, current_action= bfs_queue.pop()
        
        #skip if state is already visited.
        if current_node in visited_nodes:
            continue
                
        #Add the current node as visited.
        visited_nodes.append(current_node)
        
        #Is Current node the Goal node?
        if problem.isGoalState(current_node):
            return current_action
        
        #Expand the child.
        children= problem.getSuccessors(current_node)
        
        for child_node in children:
            if child_node[0] not in visited_nodes:
                new_action= list(current_action)
                new_action.append(child_node[1])
                bfs_queue.push((child_node[0], new_action))
    
    #When no goal is found, we return empty.
    util.raiseNotDefined()
    
    #I have also used Normal Queue and the results were as follows:
    #The total cost and nodes expanded were the same for Medium and Big mazez. Essentially yeilding same results.
    

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #Priority Queue Initialization for UCS as a tree.
    ucs_queue= util.PriorityQueue()
    visited_nodes= []
    root_node= problem.getStartState()
    
    #Push start state and empty action and initial cost of 0 and priority in queue into queue.
    ucs_queue.push((root_node,[],0),problem.getCostOfActions([]))
    
    #Loop as long as the Priority Queue is not empty.
    while not ucs_queue.isEmpty():
        current_node, current_action, current_cost= ucs_queue.pop()
        
        #skip if state is already visited.
        if current_node in visited_nodes:
            continue
                
        #Add the current node as visited.
        visited_nodes.append(current_node)
        
        #Is Current node the Goal node?
        if problem.isGoalState(current_node):
            return current_action
        
        #Expand the child.
        children= problem.getSuccessors(current_node)
        
        #child_node array stores 'node' 'action' and 'cost' for 0,1,2.
        for child_node in children:
            if child_node[0] not in visited_nodes:
                new_action= list(current_action)
                new_action.append(child_node[1])
                new_cost= current_cost+ child_node[2]
                #priority is just the cost from current node to child node.
                ucs_queue.push((child_node[0], new_action, new_cost), problem.getCostOfActions(new_action))
    
    #When no goal is found, we return empty.
    util.raiseNotDefined()  
    
    #We can see the UCS give priority to the Food-rich areas and avoid Ghost areas from MediumDotted(Very Low total cost) and MediumScary(Very high total cost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #Priority Queue Initialization for A* as a tree.
    astar_queue= util.PriorityQueue()
    visited_nodes= []
    root_node= problem.getStartState()
    
    #Push start state and empty action and initial cost of 0 and priority in queue into root node.
    astar_queue.push((root_node,[],0),(problem.getCostOfActions([])+ heuristic(root_node, problem)))
    
    #Loop as long as the Priority Queue is not empty.
    while not astar_queue.isEmpty():
        current_node, current_action, current_cost= astar_queue.pop()
        
        #skip if state is already visited.
        if current_node in visited_nodes:
            continue
                
        #Add the current node as visited.
        visited_nodes.append(current_node)
        
        #Is Current node the Goal node?
        if problem.isGoalState(current_node):
            return current_action
        
        #Expand the child.
        children= problem.getSuccessors(current_node)
        
        #child_node array stores 'node' 'action' and 'cost' for 0,1,2.
        for child_node in children:
            if child_node[0] not in visited_nodes:
                new_action= list(current_action)
                new_action.append(child_node[1])
                new_cost= current_cost+ child_node[2]
                #priority is: f(n)=g(n)+h(n)
                #g(n) is cost so far to reach child_node
                #h(n) is cost to goal from child_node
                astar_queue.push((child_node[0], new_action, new_cost), (new_cost + heuristic(child_node[0], problem)))
    
    #When no goal is found, we return empty.
    util.raiseNotDefined()  
    
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
