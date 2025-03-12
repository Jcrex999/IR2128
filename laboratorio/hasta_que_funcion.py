# Resolucion de la tarea 1
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators
from pyevolve import Crossovers
from pyevolve import Consts
import math

def fitness(chromosome):
    params = [
        chromosome[0] * (1.8 - 0.8) + 0.8,
        chromosome[1] * (0.5 - 0) + 0,
        chromosome[2] * (0.5 - (-0.5)) + (-0.5),
        chromosome[3] * (3.14 - (-3.14)) + (-3.14),
        chromosome[4] * (0.4 - 0.2) + 0.2,
        chromosome[5] * (0.1 - (-0.1)) + (-0.1),
        chromosome[6] * (3.14 - (-3.14)) + (-3.14),
        chromosome[7] * (0.4 - 0) + 0,
        chromosome[8] * (0 - (-1.2)) + (-1.2),
        chromosome[9] * (3.14 - (-3.14)) + (-3.14),
        chromosome[10] * (0.2 - 0) + 0,
        chromosome[11] * (0.5 - 0) + 0,
        chromosome[12] * (3.14 - (-3.14)) + (-3.14),
        chromosome[13] * (0.1 - 0) + 0,
        chromosome[14] * (0.2 - 0) + 0,
        chromosome[15] * (3.14 - (-3.14)) + (-3.14),
        chromosome[16] * (0.2 - 0) + 0,
        chromosome[17] * (1.9 - 1.6) + 1.6,
        chromosome[18] * (3.14 - (-3.14)) + (-3.14)
    ]
    return nao.crawl(params, seconds=5)

def eval_func(chromosome):
    return fitness(chromosome),

def main():
    genome = G1DList.G1DList(19)
    genome.setParams(rangemin=0.0, rangemax=1.0)
    genome.evaluator.set(eval_func)
    genome.mutator.set(Mutators.RealGaussianMutator)
    genome.crossover.set(Crossovers.G1DListCrossoverUniform)

    ga = GSimpleGA.GSimpleGA(genome)
    ga.setGenerations(10)
    ga.setPopulationSize(10)
    ga.selector.set(Selectors.GRouletteWheel)
    ga.setMutationRate(0.5)
    ga.setCrossoverRate(0.8)
    ga.evolve(freq_stats=1)

    best = ga.bestIndividual()
    print(fitness(best))

    return best

def eval_func(chromosome):
    param_min = [0.8, 0, -0.5, -3.14, 0.2, -0.1, -3.14, 0, -1.2, -3.14, 0, 0, -3.14, 0, 0, -3.14, 0, 1.6, -3.14]
    param_max = [1.8, 0.5, 0.5, 3.14, 0.4, 0.1, 3.14, 0.4, 0, 3.14, 0.2, 0.5, 3.14, 0.1, 0.2, 3.14, 0.2, 1.9, 3.14]

    robot_parameters = [0] * len(chromosome)
    for i in range(len(chromosome)):
        robot_parameters[i] = chromosome[i] * (param_max[i] - param_min[i]) + param_min[i]

    print(f"Robot parameters: {robot_parameters}")

    score = nao.crawl(robot_parameters, seconds=5)
    print(f"Score: {score}")
    return score

best = main()