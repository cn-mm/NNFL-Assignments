'''
This is the base car that implements most of the functionalities you will require.
You don't need to see / modify this code, but you need to import this module in your custom car
(See the templates for an example)

Do not change this file
'''

from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from skimage import filters, morphology, transform
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class Car(ABC):
    '''
    An abstract class that represents the car
    '''

    def __init__(self, id, weights):
        ''' Initialize the car '''
        self.surr_hist = []
        self.id = id
        self.weights = weights

        self.state = { # Defines the cars current state
            'pos': np.array([0., 0.5]),
            'vel': np.array([0., 0.]),
        }

        self.prev_state = self.state.copy() # The car's previous state

        # distance vectors directions
        r'''
        =========== <-- upper track
         0  2
         | /
         |/
         * <-- car
         |\
         | \
         1  3
         ========= <-- lower track
        '''
        self.eyes = np.array([[0, 1], [0, -1], [1, 1], [1, -1]])

        self.max_dist = 0.2
        self.max_vel = np.array([0.015, 0.01])
        self.max_acc = np.array([0.002, 0.002])
        self.max_time = 100

    def get_surrounding(self, track):
        pos = self.state['pos']
        vel = self.state['vel']

        normalize = lambda v: v if np.linalg.norm(v) == 0 else v/np.linalg.norm(v)

        if np.linalg.norm(vel) != 0:
            v1 = normalize(vel)
            rot_matrix = np.array([v1, [-v1[1], v1[0]]]).T
        else:
            rot_matrix = np.eye(2) # 2 is the num of dimensions
        rotated_eyes = self.eyes.dot(rot_matrix.T)
        dists = np.zeros(self.eyes.shape[0])
        for i in range(self.eyes.shape[0]):
            idx = 0
            if rotated_eyes[i][0] == 0: # parallel to y
                idx = np.abs(pos[0]-track['x']).argmin()
                if rotated_eyes[i][1]>0:
                    dists[i] = np.abs(pos[1]-track['y_up'])[idx]
                else:
                    dists[i] = np.abs(pos[1]-track['y_down'])[idx]
                dists[i] = min(self.max_dist, dists[i])
                continue

            # calculate m and c
            rotated_eyes_slope = rotated_eyes[i][1]/rotated_eyes[i][0]
            rotated_eyes_intercept = rotated_eyes[i][1]+pos[1]-rotated_eyes_slope*(rotated_eyes[i][0]+pos[0])
            line_along_eyes = rotated_eyes_slope*track['x']+rotated_eyes_intercept
            # print("line_along_eyes {},{}".format(line_along_eyes[0:5], track['x'][0:5]))
            # print("pos of car {}, {}".format(pos[0], pos[1]))


            # take upper track if line of vision is above 0
            if rotated_eyes[i][1] > 0:
                # calculate m and c
                idx = np.argwhere(
                    np.diff(np.sign(line_along_eyes - track['y_up']))
                ).flatten()

            # else take lower track
            else:
                # find point on line drawn from eyes closest to track
                idx = np.argwhere(
                    np.diff(np.sign(line_along_eyes - track['y_down']))
                ).flatten()
            # print("intesection at ")
            # print(idx)
            # plt.plot(track['x'],line_along_eyes)
            # plt.plot(track['x'][idx], line_along_eyes[idx],'ro')
            try:
                dists[i] = np.sqrt((line_along_eyes[idx]-pos[1])**2 + (track['x'][idx]-pos[0])**2)
            except ValueError:
                dists[i] = self.max_dist
            #dists[i] = min(self.max_dist, dists[i])

        # plt.plot(track['x'],track['y_up'])
        # plt.plot(track['x'],track['y_down'])
        # plt.show()

        return dists


    def is_legal(self, track):
        ''' check if the current car position is within the track boundary '''
        eps = 0
        track_at = lambda x: (
            np.interp(x, track['x'], track['y_down']),
            np.interp(x, track['x'], track['y_up']),
        )
        t = track_at(self.state['pos'][0])
        return (t[0] + eps < self.state['pos'][1] < t[1] - eps) and (0 <= self.state['pos'][0] <= track['x'][-1])

    @abstractmethod
    def move(self, weights, params):
        ''' This method will be filled in by the students '''
        raise NotImplentedError()

    def draw_frame(self, frame_num, track, images):
        eps = 0.05

        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        #Image from plot
        ax.axis('off')
        fig.tight_layout(pad=0)
        # To remove the huge white borders
        ax.margins(0)

        ax.plot(track['x'], track['y_up'], c='k') # Upper track
        ax.plot(track['x'], track['y_down'], c='k') # lower track
        show_info = f'time: {frame_num}s | Fitness: {round(self.state["pos"][0]/frame_num,3)}'
        ax.scatter(self.state['pos'][0], self.state['pos'][1], marker='>', c='b', label=show_info) # Car
        ax.vlines(track['x'][0], 0, 1, color='r', label=f'Start') # Car
        ax.vlines(track['x'][-1], 0, 1, color='g', label=f'End') # Car
        r = self.max_dist
        for angle in self.eyes:
            x,y = self.state['pos']
            ax.plot([x, x + r*angle[0]], [y, y + r*angle[1]], c='r', alpha=0.3)

        ax.set_xlim([0 - eps, 1 + eps])
        ax.set_ylim([0 - eps, 1 + eps])
        ax.legend(loc='upper left')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))

        #_, (w, h) = canvas.print_to_buffer()
        #img = np.fromstring(canvas.tostring_rgb(), dtype=np.float64).reshape(h, w, 4)
        images.append(image)

    def run(self, track_img, save=None):
        '''
        This method will be used to move the car according to the given track
        Returns the fitness values ( betweeen 0 and 1 )
        The show flag toggles visualization
        '''
        track = self.get_track(track_img)
        self.state['pos'] = np.array([0, .5*track['y_up'][0] + .5*track['y_down'][0]])
        images = []

        t_final = self.max_time
        rng = range(1,self.max_time+1)
        if save:
            rng = tqdm(range(1,self.max_time+1), desc=f'Visualizing fitness for car {self.id}')

        for t in rng: # Run for fixed time
            if save:
                self.draw_frame(t, track, images)

            self.surr_hist.append(self.surr_hist)
            surr = self.get_surrounding(track)

            params = {
                'dist_left': surr[0],
                'dist_right': surr[1],
                'dist_front_left': surr[2],
                'dist_front_right': surr[3],
                'x': self.state['pos'][0],
                'y': self.state['pos'][1],
                'vx': self.state['vel'][0],
                'vy': self.state['vel'][1],
                'prev_x': self.prev_state['pos'][0],
                'prev_y': self.prev_state['pos'][1],
                'prev_vx': self.prev_state['vel'][0],
                'prev_vy': self.prev_state['vel'][1],
            }

            acc = np.array(self.move(self.weights, params)) # get acc from students func
            acc = np.maximum(np.minimum(acc, self.max_acc), -1*self.max_acc) # Clip acceleration

            self.state['vel'] += acc
            self.state['vel'] = np.maximum(np.minimum(self.state['vel'], self.max_vel), -1*self.max_vel) # Clip the velocity

            pos = self.state['pos'] + self.state['vel']

            if self.is_legal(track): # If within track boundary
                self.prev_state = self.state.copy()
                self.state['pos'] = pos
            else:
                self.state['vel'] = np.array([0.,0.]) # stop the car if it hits the wall
                y = self.state['pos'][1]
                for y_ in [1,-1,2,-2,3,-3,4,-4,5,-5,7,-7,10,-10]:
                    self.state['pos'][1] = y + y_/100
                    if self.is_legal(track):
                        break
                else:
                    self.state['pos'] = self.prev_state['pos']

            if self.state['pos'][0] >= track['x'][-1]:
                t_final = t
                break

        if save: # build gif
            head, tail = os.path.splitext(save)
            save = head + '.gif'

            with imageio.get_writer(save, mode='I') as writer:
                for image in images:
                    writer.append_data(image)

        return self.max_time * max(0,self.state['pos'][0]) / max(1,t_final) # is between 0 and 1

    def get_track(self, track_img):
        '''
        Convert an image of a track to a track compatible with our interface (for the run method)
        '''

        # lambda function to scale an array to [min,max]
        scale = lambda arr,min,max: min + (max-min)*(arr-np.min(arr))/(np.max(arr)-np.min(arr))

        # lambda function get a PIL image from an array
        get = lambda arr: Image.fromarray(scale(arr,0,255))

        # lambda function to get running mean of a given array
        running_mean = lambda x,N: (np.cumsum(np.insert(x, 0, 0))[N:] - np.cumsum(np.insert(x, 0, 0))[:-N]) / float(N)

        im = Image.open(track_img).convert('L') # open image and convert to Black and white (0-255)
        im = 255 - np.array(im) # invert image colors for better results and convert to an array for easier manipulation

        im = filters.median(im) # apply median filter to remove salt and pepper noise
        im = filters.gaussian(im,sigma=2.2) # use gaussian filter to smooth out the image
        im = scale(im,0,1)

        # Reduce thick lines to single pixel lines for easier analysis
        # Returns a thresholded image i.e All values are either 255 or 0
        final_im =np.array(morphology.skeletonize_3d(im)).astype(int)

        y,x = np.where(final_im==255) # Get the x and y co-ordinates (numpy and PIL have transpose conventions)
        y = y[np.argsort(x)]
        x = np.sort(x)
        x,y = scale(x,0,1), scale(y,0,1)

        # Divide the y values into upper and lower parts of the track
        x_final = []
        y_up = []
        y_down = []
        for pt in np.sort(x):
            try:
                y1,y2 = y[np.where(x==pt)]
                x_final.append(pt)
                y_up.append(max(y1,y2))
                y_down.append(min(y1,y2))
            except:
                pass

        # Use a moving average to smooth out the array to remove any minor discontinuities
        # Window size is 1% of track length to minimize the amount of data lost
        win = int(0.01*len(x_final))
        y_up = running_mean(y_up,win)
        y_down = running_mean(y_down,win)
        x_final = x_final[:-win+1]
        return {
            'x': np.array(x_final),
            'y_down': 1 - np.array(y_up),
            'y_up': 1 - np.array(y_down),
        }

    def save(self, file='weights.npy'):
        '''
        Save the car's weights in the provided file
        Default file is weights.npy
        '''

        head, tail = os.path.splitext(file)
        file = head + '.npy'

        print(f'saving weights of car {self.id} to {file}')
        np.save(file, self.weights)

    def load(self, file='weights.npy'):
        '''
        Load the car's weights from the provided file
        Default file is weights.npy
        '''

        head, tail = os.path.splitext(file)
        file = head + '.npy'
        print(f'loading weights of car {self.id} from {file}')
        self.weights = np.load(file)
