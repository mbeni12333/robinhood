import algos
import utils
import numpy as np

nb_tasks = 10

instance, erreur = utils.generateInstance(nb_tasks)
colors = np.random.rand(nb_tasks, 3)

solution = algos.PREDICT(instance)
obj = utils.evaluateObjective(instance, solution)
utils.plotGant(solution, nb_tasks, colors, obj)

solution = algos.SPT(instance)
obj = utils.evaluateObjective(instance, solution)
utils.plotGant(solution, nb_tasks, colors, obj)

solution = algos.ROUND_ROBIN(instance)
obj = utils.evaluateObjective(instance, solution)
utils.plotGant(solution, nb_tasks, colors, obj)