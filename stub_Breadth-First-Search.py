# Imports.
import numpy as np
import numpy.random as npr
import copy

from SwingyMonkey import SwingyMonkey
import scipy.stats
import pandas

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        #macro definition
        self.N_STEP = 8# consider the next 20 steps;
        self.N_ITERATION = 100
        self.DISCOUNT = 0.9
        self.Gravitity = None
        Eta = 0.1#

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.Gravitity = None
        print("reset")
    def action_callback(self, state):
        # not to take action in the first step, we need one more step to estimate gravitiy
        if(None == self.last_state):
            self.last_action = 0
            self.last_state = state
            return 0
        elif None == self.Gravitity:
            Gravitity = self.last_state['monkey']['vel'] - state['monkey']['vel']


        # Breadth-First-Search --- slow!!!
        # current state List
        TempState = copy.deepcopy(state)
        TempState['TimeStep'] = 0
        TempState['Reward'] = self.reward_callback_new(TempState)
        TempState['FirstAction'] = None
        StateList = list()
        StateList.append(TempState)
        for i in range(self.N_STEP):
            for j in range(StateList.__len__()):
                # for a given state in current state list, search both the action and inaction options
                Temp_State_1 = copy.deepcopy(StateList[0])
                Temp_State_0 = copy.deepcopy(StateList[0])

                Temp_State_1['monkey']['vel'] = 15
                Temp_State_1['monkey']['top'] = Temp_State_1['monkey']['top'] + Temp_State_1['monkey']['vel']
                Temp_State_1['monkey']['bot'] = Temp_State_1['monkey']['bot'] + Temp_State_1['monkey']['vel']
                Temp_State_1['monkey']['vel'] = Temp_State_1['monkey']['vel'] - Gravitity
                Temp_State_1['tree']['dist'] = Temp_State_1['tree']['dist'] - 25
                Temp_State_1['TimeStep'] = i+1
                if 0 == i:
                    Temp_State_1['FirstAction'] = 1
                CurrentReward = self.reward_callback_new(Temp_State_1)
                if CurrentReward>=0:
                    Temp_State_1['Reward'] = Temp_State_1['Reward'] + CurrentReward
                    StateList.append(Temp_State_1)


                Temp_State_0['monkey']['top'] = Temp_State_0['monkey']['top'] + Temp_State_0['monkey']['vel']
                Temp_State_0['monkey']['bot'] = Temp_State_0['monkey']['bot'] + Temp_State_0['monkey']['vel']
                Temp_State_0['monkey']['vel'] = Temp_State_0['monkey']['vel'] - Gravitity
                Temp_State_0['tree']['dist'] = Temp_State_0['tree']['dist'] - 25
                Temp_State_0['TimeStep'] = i+1
                if 0 == i:
                    Temp_State_0['FirstAction'] = 0
                CurrentReward = self.reward_callback_new(Temp_State_0)
                if CurrentReward>=0:
                    Temp_State_0['Reward'] = Temp_State_0['Reward'] + CurrentReward
                    StateList.append(Temp_State_0)

                # update list
                StateList.remove(StateList[0])
            Max_Reward = -float("inf")
        # find the final state with maximal reward
        Max_Reward = -float("inf")
        Max_Action = 0#npr.rand() < 0.1
        for j in range(StateList.__len__()):
            if Max_Reward<StateList[j]['Reward']:
                Max_Reward = StateList[j]['Reward']
                Max_Action = StateList[j]['FirstAction']

        # new_action = npr.rand() < 0.1
        #if Reward!=0:
        #print(state)

        self.last_action = Max_Action
        self.last_state  = state

        return self.last_action
    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward
        return reward

    # def reward_callback_new(self,state):
    #     reward = 0
    #     # hit the tree
    #     if((state['tree']['dist'] <= 210 and state['tree']['dist']>=60) and (state['monkey']['top'] > state['tree']['top'] or state['monkey']['bot'] < state['tree']['bot'])):
    #         reward = - 5
    #     # hit the edge
    #     if state['monkey']['bot'] <0 or state['monkey']['top'] > 400:
    #         reward = - 10
    #     # pass the tree
    #     if(state['tree']['dist'] == 60 and state['monkey']['top'] < state['tree']['top'] and state['monkey']['bot'] > state['tree']['bot']):
    #         reward = +1
    #     return reward
    def reward_callback_new(self,state):
        #reward = (400 - state['monkey']['top'])*(state['monkey']['bot'] - 0)/171.5/171.5 # measure how the monkey deviate from the center
        reward = scipy.stats.norm(200, 50).pdf((state['monkey']['top']+state['monkey']['bot'])/2)
        # hit the tree
        if((state['tree']['dist'] <= 210 and state['tree']['dist']>=60) and (state['monkey']['top'] > state['tree']['top'] or state['monkey']['bot'] < state['tree']['bot'])):
            reward = - 5
        # hit the edge
        if state['monkey']['bot'] <0 or state['monkey']['top'] > 400:
            reward = - 10
        # pass the tree
        if(state['tree']['dist'] == 60 and state['monkey']['top'] < state['tree']['top'] and state['monkey']['bot'] > state['tree']['bot']):
            #reward = 10*(state['tree']['top'] - state['monkey']['top'])*(state['monkey']['bot'] - state['tree']['bot'])/np.power((state['tree']['top'] - state['monkey']['top'] +  state['monkey']['bot'] - state['tree']['bot'])/2,2)
            reward = 1*scipy.stats.norm((state['tree']['top']+state['tree']['bot'])/2, 25).pdf((state['monkey']['top'] + state['monkey']['bot']) / 2)
            #reward = +1
        return reward

def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             #text="reward %d" % (ii),
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 20, 10)

	# Save history. 
	np.save('hist',np.array(hist))
    pandas.DataFrame(hist).to_csv('scores.csv')

