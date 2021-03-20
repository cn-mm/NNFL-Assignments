'''
A template to see how template_car_2 is performing
No training is required since the car is completely hardcoded
'''

from template_2_car import MyCar
import numpy as np

track1 = 'tracks/sample_track_1.jpg'
track2 = 'tracks/sample_track_2.jpg'
track3 = 'tracks/sample_track_3.jpg'

car = MyCar(id=0, weights=np.zeros(6,)) # weights are dummy as we are not using them in our move function at all

# Visualize on different tracks
f1 = car.run(track1, save='template_2_track1.gif')
f2 = car.run(track2, save='template_2_track2.gif')
f3 = car.run(track3, save='template_2_track3.gif')
print(f'Overall fitness: {f1+f2+f3} = {f1} + {f2} + {f3}')

# save (dummy) weights
car.save(file='template_2_weights')
