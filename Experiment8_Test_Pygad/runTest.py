import pygad
import numpy
import matplotlib.pyplot as plt

from mpipool import MPIPool
from mpi4py import MPI
"""
Given the following function:
    y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.
"""

def fitness_func(solution, solution_idx):
    function_inputs = [4,-2,3.5,5,-11,-4.7] # Function inputs.
    desired_output = 44 # Function output.

    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

class PooledGA(pygad.GA):

    def cal_pop_fitness(self):
        global pool,hyperParam
        pop_fitness = pool.starmap(fitness_func, [(individual,i) for i,individual in enumerate(self.population)])
        #print(pop_fitness)
        pop_fitness = numpy.array(pop_fitness)
        return pop_fitness

# run optimization 
with MPIPool() as pool:
    pool.workers_exit() ## Only master process will proceed
    
    num_generations = 100 # Number of generations.
    num_parents_mating = 2 # Number of solutions to be selected as parents in the mating pool.

    sol_per_pop = 39 # Number of solutions in the population.
    num_genes = 6

 
    
    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_num_genes = 2
    
    ga_instance = PooledGA(num_generations=num_generations,
                num_parents_mating=num_parents_mating,
                fitness_func=fitness_func, # Actually this is not used.
                sol_per_pop=sol_per_pop,
                num_genes=num_genes,
                parent_selection_type=parent_selection_type,
                keep_parents=keep_parents,
                crossover_type=crossover_type,
                mutation_type=mutation_type,
                mutation_num_genes=mutation_num_genes
                )
    # Running the GA to optimize the parameters of the function.
    ga_instance.run()

    #ga_instance.plot_result()
    #plt.plot(ga_instance.best_solutions_fitness)
    print("Fitness_versus_generation")
    print(ga_instance.best_solutions_fitness)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))




