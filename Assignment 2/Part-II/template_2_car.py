'''
Sample file for your own custom car, this car does not require any training.
This is an example of a completely hard-coded solution
'''

from base_car import Car
import numpy as np

class MyCar(Car):
    ''' Template  car '''

    def move(self, weights, params):
        '''
        Need to implement this method for your car to move effectively

        We have provided this sample implementation but you are free to implement
        this method as you wish - no restrictions at all

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
        For this hardcoded car, we use 2 simple rules:

        1. If the car is very close to either side of the track, move away from the track
           i.e try to stay in the centre as much as possible

        2. Try to maintain a constant forward speed of 0.1 m/s
        '''

        ax, ay = 0, 0
        thresh_vel = 0.1 # Car will try to maintain this velocity in forward direction
        thresh_dist = 0.18 # if car is closer than this distance, apply acceleration in opposite direction

        if params['vx'] > thresh_vel:
            ax -= self.max_acc[0] # Give as much acceleration as possible
        elif params['vx'] < thresh_vel:
            ax += self.max_acc[0] # Give as much acceleration as possible

        if params['dist_left'] <= thresh_dist:
            ay -= self.max_acc[1]/2 # Give half of max possible acceleration as tracks turn a lot

        if params['dist_right'] <= thresh_dist:
            ay += self.max_acc[1]/2

        return [ax, ay] # Return the acceleration
