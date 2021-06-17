import pygad
import numpy
from multiprocessing import Pool

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
if __name__ == "__main__":
    numpy.random.seed(1234)
    with Pool(processes=20) as pool:
        num_generations = 100 # Number of generations.
        num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.

        sol_per_pop = 20 # Number of solutions in the population.
        num_genes = 6

        last_fitness = 0
        
        
        ga_instance = PooledGA(num_generations=num_generations,
                            initial_population =  numpy.random.randint(-10,10,(20,6)),
                            num_parents_mating=num_parents_mating,
                            sol_per_pop=sol_per_pop,
                            num_genes=num_genes,
                            fitness_func=fitness_func)
        # Running the GA to optimize the parameters of the function.
        ga_instance.run()

        #ga_instance.plot_result()

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))




