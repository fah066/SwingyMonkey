# Imports.
import numpy as np
import numpy.random as npr
import copy
import scipy.stats
import pandas as pd

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        #macro definition
        self.N_STEP = 18# consider the next 20 steps;
        self.N_ITERATION = 5
        self.DISCOUNT = 0.9
        self.Gravitity = None
        Eta = 0.1#
    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.Gravitity = None
    def action_callback(self, state):
        # not to take action in the first step, we need one more step to estimate gravitiy
        if(None == self.last_state):
            self.last_action = 0
            self.last_state = state
            return 0
        if None == self.Gravitity:
            self.Gravitity = self.last_state['monkey']['vel'] - state['monkey']['vel']

        # regular depth-First-Search --- it is working!
        # current state List
        TempState = copy.deepcopy(state)
        TempState['TimeStep'] = 0
        TempState['Reward'] = self.reward_callback_new(TempState)
        TempState['Action'] = 0
        StateList = list()
        StateList.append(TempState)

        Max_Depth = -1 # the maximized depth the tree has been searched --- also the maximal time step the monkey can survive;
        Max_Reward = -float("inf")
        Max_Action = 0
        Max_Num = 0

        while True:
            if StateList.__len__() == 0:
                break
            if Max_Num>self.N_ITERATION:
                break

            if StateList.__len__() == self.N_STEP:# an ideal solution
                Max_Num = Max_Num+1
                if StateList.__len__() > Max_Depth and TempState['Reward']>Max_Reward:
                    Max_Action = StateList[0]["Action"]
                    Max_Reward = TempState['Reward']
                StateList.remove(StateList[StateList.__len__() - 1])
                TempState = StateList[StateList.__len__() - 1]
                TempState['Action'] = TempState['Action'] + 1


            TempState = StateList[StateList.__len__() - 1]  # get the most recent node
            # if action == 0, search the #0 child
            if 0 == TempState['Action']:
                Temp_State_0 = copy.deepcopy(TempState)

                Temp_State_0['monkey']['top'] = Temp_State_0['monkey']['top'] + Temp_State_0['monkey']['vel']
                Temp_State_0['monkey']['bot'] = Temp_State_0['monkey']['bot'] + Temp_State_0['monkey']['vel']
                Temp_State_0['monkey']['vel'] = Temp_State_0['monkey']['vel'] - self.Gravitity
                Temp_State_0['tree']['dist'] = Temp_State_0['tree']['dist'] - 25
                Temp_State_0['Action'] = 0
                Temp_State_0['TimeStep'] = TempState['TimeStep'] + 1
                CurrentReward = self.reward_callback_new(Temp_State_0)
                if CurrentReward >= 0:
                    Temp_State_0['Reward'] = Temp_State_0['Reward'] + CurrentReward*np.power(self.DISCOUNT,TempState['TimeStep'])
                    StateList.append(Temp_State_0)
                    if StateList.__len__()>Max_Depth and Max_Reward>Temp_State_0['Reward']:
                        Max_Action = StateList[0]["Action"]
                        Max_Reward = Temp_State_0['Reward']
                else:
                    TempState['Action'] = TempState['Action'] + 1  # switch to the other node
            if 1 == TempState['Action']:
                Temp_State_0 = copy.deepcopy(TempState)

                Temp_State_0['monkey']['vel'] = 15
                Temp_State_0['monkey']['top'] = Temp_State_0['monkey']['top'] + Temp_State_0['monkey']['vel']
                Temp_State_0['monkey']['bot'] = Temp_State_0['monkey']['bot'] + Temp_State_0['monkey']['vel']
                Temp_State_0['monkey']['vel'] = Temp_State_0['monkey']['vel'] - self.Gravitity
                Temp_State_0['tree']['dist'] = Temp_State_0['tree']['dist'] - 25
                Temp_State_0['Action'] = 0
                Temp_State_0['TimeStep'] = TempState['TimeStep'] + 1
                CurrentReward = self.reward_callback_new(Temp_State_0)
                if CurrentReward >= 0:
                    Temp_State_0['Reward'] = Temp_State_0['Reward'] + CurrentReward*np.power(self.DISCOUNT,TempState['TimeStep'])
                    StateList.append(Temp_State_0)
                    if StateList.__len__()>Max_Depth and Max_Reward>Temp_State_0['Reward']:
                        Max_Action = StateList[0]["Action"]
                        Max_Reward = Temp_State_0['Reward']
                else:
                    TempState['Action'] = TempState['Action'] + 1  # switch to the other node

            if 2 == TempState['Action']:
                StateList.remove(TempState)
                try:
                    TempState = StateList[StateList.__len__() - 1]
                    TempState['Action'] = TempState['Action'] + 1  # switch to the other node
                except:
                    print("Oops!Reward:"+str(Max_Reward)+"Reward_depth:"+str(Max_Depth))

        # new_action = npr.rand() < 0.1
        #if Reward!=0:
        #print(state)
        #print(str(self.reward_callback_new(state)) + ";" +str(self.Gravitity))
        self.last_action = Max_Action
        self.last_state  = state

        return self.last_action
    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward
        return reward

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
	run_games(agent, hist, 200, 10)

	# Save history.
	pd.DataFrame(hist).to_csv('scores_eps0.1_2.csv', header=False, index=True)#np.save('hist',np.array(hist))

	# Save history. 
	#np.save('hist',np.array(hist))

