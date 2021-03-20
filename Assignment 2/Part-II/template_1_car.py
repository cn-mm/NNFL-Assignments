'''
Sample file for your own custom car, this car is used in the training templates as
members of the GA / PSO population
'''

from base_car import Car
import numpy as np

def sigmoid(z):
    return 1/1+np.exp(-z)

def tanh(z):
    return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

class MyCar(Car):
    ''' Template car '''

    def move(self, weights, params):
        '''
        Need to implement this method for your car to move effectively

        We have provided this sample implementation but you are free to implement
        this method as you wish - no restrictions at all
        '''

        '''
        Parameters
        +--------------------------------------------------------------------------------------------------------------------------+
        | Parameter        |             Meaning                              | Range   | Remarks                                  |
        |------------------|--------------------------------------------------|---------|------------------------------------------|
        | x,y              | it's current position                            | 0 - 1   | x=0 represents the start of the track,   |
        | prev_x, prev_y   | it's previous position                           | 0 - 1   | x=1 represents the end of the track      |
        |------------------|--------------------------------------------------|---------|------------------------------------------|
        | vx,vy            | it's current velocity                            | 0 - 1   | max_vx = 0.015, max_vy = 0.01            |
        | prev_vx, prev_vy | it's previous velocity                           | 0 - 1   |                                          |
        |------------------|--------------------------------------------------|---------|------------------------------------------|
        | dist_left        | Distance b/w the car's left side and the track   | 0 - 0.2 | The car can see the track only if it is  |
        | dist_right       | Distance b/w the car's right side and the track  | 0 - 0.2 | at a distance of 0.2 or less, otherwise  |
        | dist_front_left  | Distance at a 45 degree angle to the car's left  | 0 - 0.2 | these distances assume the value 0.2     |
        | dist_front_right | Distance at a 45 degree angle to the car's right | 0 - 0.2 |                                          |
        +--------------------------------------------------------------------------------------------------------------------------+

        The car needs to make a decision on how to move (what acceleration to give)
        based on these parameters
        '''

        max_vx, max_vy = self.max_vel
        max_ax, max_ay = self.max_acc

        '''
        Think carefully of what the features below should be

        Generally you have a Neural Network automatically, creating these features for you,
        but in this task you have to manually create useful features from the given parameters

        For example our thinking in creating our (rather simple) features was:
        ax is how fast the car moves forward, therefore it should depend on:
        * The car's current x velocity (vx)
        * how far the track is in the forward direction (dist_front_left, dist_front_right)

        ax is how fast the car moves sideways, therefore it should depend on:
        * how far the track is on it's either side (dist_left - dist_right)
        * The car's current y co-ordinate (y)
        * The car's current y velocity (vy)

        Your features can be as simple / as complicated as you want.
        '''

        features_x = np.array([params['vx'], params['dist_front_left'], params['dist_front_right']])
        features_y = np.array([params['dist_left'] - params['dist_right'], params['y'], params['vy']])

        '''
        Weights (W) are of shape (8,1) they are divided as below:

        Suppose W = [p0 p1 p2 p3 p4 p5 p6 p7], then:

        W1 = [p0 p1 p2]
        b1 = [p3]
        W2 = [p4 p5 p6]
        b2 = [p7]
        '''

        W1 = weights[0:3]
        b1 = weights[3]
        W2 = weights[4:7]
        b2 = weights[7]

        '''
        The weights are essentially acting like the coefficients (weights) in a
        linear sum of your features

        This is just our implementation, you have complete control of how you
        compute ax and ay using the weights and the feautures

        Suppose features_x = [fx0 fx1 fx2], features_y = [fy0 fy1 fy2], then:

        ax = tanh( p0*fx0 + p1*fx1 + p2*fx2  + p3) (scaled to correct acceleration range)
        ay = tanh( p4*fx4 + p5*fx5 + p6*fx6  + p7) (scaled to correct acceleration range)

        Can you see the similiraties to the Neural Network you implemented in the
        last assignment? (Can you use this to your advantage?)
        '''
        ax = max_ax * tanh( np.dot(W1,features_x.T) + b1)
        ay = max_ay * tanh( np.dot(W2,features_y.T) + b2)

        return [ax, ay] # Return the acceleration
