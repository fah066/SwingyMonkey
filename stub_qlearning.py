# Imports.
import numpy as np
import numpy.random as npr
import os
import pandas as pd
from SwingyMonkey import SwingyMonkey
os.chdir('C:\\Users\\user\\Desktop\\Harvard\\COMPSCI181\\cs181-s17-qiuhaozhang\\p4\\code')

class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        #initialize parameters
        #self.eta=0.1
        self.eps=0.01
        self.gamma=0.9
        self.iteration=0
        self.grav=1

        #initialize Q matrix
        self.xbin=50
        self.ybin=25
        self.vbin=20
        self.Q=np.zeros((600/self.xbin+1, 300*2/self.ybin+1, 80/self.vbin+1, 5, 2))
        self.learn=np.zeros((600/self.xbin+1, 300*2/self.ybin+1, 80/self.vbin+1, 5, 2))

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        #self.iteration+=1

        ##assign new state
        self.xstate = int(state['tree']['dist']/self.xbin) if state['tree']['dist']>0 else 0
        self.ystate = int((state['tree']['top']-state['monkey']['top']+200)/self.ybin)
        self.vstate = int(state['monkey']['vel']/self.vbin)
        #self.cur_monkey= state['monkey']['top']

        if self.last_state == None:
            self.action = 0

        else:
            if self.iteration==2:
                self.grav = int(abs(state['monkey']['vel'])==1)

            ##update Q matrix
            self.derivative = self.Q[self.last_state[0], self.last_state[1], self.last_state[2], self.last_state[3], self.last_action] \
            - (self.last_reward + self.gamma*np.max(self.Q[self.xstate, self.ystate, self.vstate, self.grav, :]))

            ##calculate eta
            self.eta = 1/self.learn[self.last_state[0], self.last_state[1], self.last_state[2], self.last_state[3], self.last_action]

            self.Q[self.last_state[0], self.last_state[1], self.last_state[2],self.last_state[3], self.last_action] -= self.eta*self.derivative
            
            ##determine action
            self.action = int(np.argmax(self.Q[self.xstate, self.ystate, self.vstate, self.grav,:]))
            
            ##epsilon greedy
            if self.learn[self.xstate, self.ystate, self.vstate, self.grav, self.action]>0:
                eps=self.eps/self.learn[self.xstate, self.ystate, self.vstate, self.grav, self.action]
            else:
                eps=self.eps

            if npr.rand()<eps:
                self.action = npr.choice([0,1])

        self.last_state = [self.xstate, self.ystate, self.vstate, self.grav]
        self.last_action= self.action
        self.learn[self.xstate, self.ystate, self.vstate, self.grav, self.action] += 1

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

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
    #         reward = 1
    #     return reward

def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
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
	run_games(agent, hist, 200, 2)

	# Save history. 
	pd.DataFrame(hist).to_csv('scores_norand_3.csv', header=False, index=True)#np.save('hist',np.array(hist))
    #print(hist)
    #pd.DataFrame(hist)#.to_csv('scores', header=False, index=True)