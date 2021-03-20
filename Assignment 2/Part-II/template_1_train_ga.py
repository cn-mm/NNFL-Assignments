'''
A template for training your cars using a Genetic Algorithm
'''

from template_1_car import MyCar
import numpy as np
from copy import deepcopy

def encode(weights, w_range, num_bits):
    '''
    Encode weights as an array of 1s and 0s
    '''
    encoded_weights = []
    int2binary = lambda x,bits: list(np.binary_repr(x).zfill(bits))
    min_w, max_w = w_range
    for w in weights:
        w = round((pow(2,num_bits)-1)*(w-min_w)/(max_w-min_w))
        encoded_weights.extend(int2binary(w, num_bits))
    return [int(i) for i in encoded_weights]

def decode(enc_weights,w_range,num_bits):
    '''
    Decode an array of 0s and 1s to the weights array
    '''
    n = len(enc_weights)
    bin2int = lambda x,bits: sum([np.power(2,bits-1-i)*x[i] for i in range(len(x))])
    min_w, max_w = w_range
    weights = []
    for i in range(0,n,num_bits):
        w = bin2int(enc_weights[i:i+num_bits], num_bits)
        w = min_w + (max_w-min_w)*w/(pow(2,num_bits)-1)
        weights.append(w)
    return np.array(weights)

def create_population(weight_size, hyper_params):
    '''
    create the GA population (essentially a list of cars)
    '''
    lo,hi = hyper_params['weight_range']
    return [MyCar(id=i, weights=np.random.uniform(lo,hi,weight_size)) for i in range(hyper_params['population_size'])]

def selection(population, track, hyper_params):
    '''
    Implement roulette wheel based selection for the entire population
    '''
    pop_size = hyper_params['population_size']
    num_elites = round(hyper_params['elite']*pop_size)
    sorted_pop = sorted(population, key = lambda p: p.run(track), reverse=True)
    elites = deepcopy(sorted_pop[:num_elites])
    pop_fitness = [p.run(track) + 1e-6 for p in population]
    new_pop = []

    new_pop.extend(elites) # put elites in the population

    while len(new_pop) < len(population):
        selected_member = np.random.choice(pop, p=np.array(pop_fitness)/sum(pop_fitness))
        new_pop.append(selected_member)

    return new_pop, elites

def crossover_util(w1, w2):
    '''
    Implement single point crossover for one pair of (encoded) weights
    '''
    crossover_point = np.random.choice(range(1,len(w1)-1))
    new_w = w1[:crossover_point] + w2[crossover_point:]
    return new_w

def crossover(population, hyper_params):
    '''
    Implement single point crossover for the entire population
    (parents are selected randomly)
    '''
    new_pop = []
    for i in range(hyper_params['population_size']):
        parent1 = np.random.choice(population)
        parent2 = np.random.choice(population)

        w1 = encode(parent1.weights, hyper_params['weight_range'], hyper_params['weight_bits'])
        w2 = encode(parent2.weights, hyper_params['weight_range'], hyper_params['weight_bits'])
        new_w = decode(crossover_util(w1,w2), hyper_params['weight_range'], hyper_params['weight_bits'])
        child = MyCar(id=i,weights=new_w)
        new_pop.append(child)

    return new_pop

def mutate(weights, mutation_prob):
    '''
    Mutate the given (enocoded) weights
    '''
    prob = hyper_params['mutation_prob']
    flip = lambda bit,p: 1-bit if np.random.random() < p else bit
    return [flip(w,prob) for w in weights]

def mutation(population, hyper_params):
    '''
    Implement uniform mutation for the entire population
    '''
    for i,member in enumerate(population):
        enc_weights = encode(member.weights, hyper_params['weight_range'], hyper_params['weight_bits']) # encode given weights
        mutated_weights = mutate(enc_weights,hyper_params['mutation_prob']) # mutate the encoded weights
        population[i].weights = decode(mutated_weights, hyper_params['weight_range'], hyper_params['weight_bits']) # decode the weights
    return population

def GA(population, track, hyper_params, print_every=10):
    pop = population
    f = [p.run(track) for p in pop]
    print(f'Iteration 0: [avg: {round(np.mean(f),3)} | best: {round(np.max(f),3)}]')

    for i in range(1,hyper_params['iterations']+1):
        pop,elites = selection(pop, track, hyper_params) # create parent population using selection (store the elites seperately)
        pop = crossover(pop, hyper_params) # Crossover
        pop = mutation(pop, hyper_params)

        num_elites = round(hyper_params['elite']*hyper_params['population_size'])
        pop = sorted(pop, key = lambda p: p.run(track))
        pop[:num_elites] = elites # propogate elite population without any changes

        if i%print_every == 0:
            f = [p.run(track) for p in pop]
            print(f'Iteration {i}: [avg: {round(np.mean(f),3)} | best: {round(np.max(f),3)}]')
            #print([p.run(track) for p in elites])
    return pop

### ----------------------------------------------------------------------- ###

# These can be freely changed
hyper_params = {
    'weight_range': (-3,3), # Range in which the wieghts can vary
    'weight_bits': 5, # Number of bits to use while encoding the weights
    'iterations': 15, # Number of iterations
    'population_size': 30, # Size of the population
    'elite': 0.15, # as a percentage of the population (between 0 and 1)
    'mutation_prob': 0.1 # mutation probability (between 0 and  1)
}

'''
Although weight_size can also be considered as a hyper parameters, it can't be
freely changed as changing this would require a corresponding change in your
car's move() function
'''
weight_size = 8

# create a population of cars
print('Creating the population ...')
pop = create_population(weight_size, hyper_params)

# train the Genetic algorithm on different tracks
track1 = 'tracks/sample_track_1.jpg'
print('Training on track 1 ...')
pop = GA(pop, track1, hyper_params, print_every=5)

track2 = 'tracks/sample_track_2.jpg'
print('Training on track 2 ...')
pop = GA(pop, track2, hyper_params, print_every=5)

# save the car that runs best on track 2
best_car = max(pop, key=lambda x: x.run(track2))
best_car.run(track2, save='best_car_ga_template_track_2.gif')
best_car.save(file='ga_best_weights')

# continue training on track 3
# car.load('ga_best_weights') # load the saved weights

print('Training on track 3 ...')
track3 = 'tracks/sample_track_3.jpg'
pop = GA(pop, track3, hyper_params, print_every=5)

# save the car that runs best on all tracks combined
best_car = max(pop, key=lambda x: x.run(track1) + x.run(track2) + x.run(track3))
f1 = best_car.run(track1, save='best_car_ga_template_track_1.gif')
f2 = best_car.run(track2, save='best_car_ga_template_track_2.gif')
f3 = best_car.run(track3, save='best_car_ga_template_track_3.gif')
print(f'Overall fitness: {f1+f2+f3} = {f1} + {f2} + {f3}')
best_car.save(file='ga_best_weights')
