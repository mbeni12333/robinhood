import algos
import utils
import numpy as np

nb_tasks = 10

instance, erreur = utils.generateInstance(nb_tasks, v_generator=utils.poisson_generator)
colors = np.random.rand(nb_tasks, 3)

# solution = algos.PREDICT(instance)
# obj = utils.evaluateObjective(instance, solution)
# utils.plotGant(solution, nb_tasks, colors, obj)

# solution = algos.SPT(instance)
# obj = utils.evaluateObjective(instance, solution)
# utils.plotGant(solution, nb_tasks, colors, obj)


solution = algos.PREDICT_PART2(instance)
obj = utils.evaluateObjective(instance, solution)
utils.plotGant(solution, nb_tasks, colors, obj)

solution = algos.PREDICT_PART2(instance, lambda task: task.remainingTime_estimated)
obj = utils.evaluateObjective(instance, solution)
utils.plotGant(solution, nb_tasks, colors, obj)


# solution = algos.ROUND_ROBIN(instance)
# obj = utils.evaluateObjective(instance, solution)
# utils.plotGant(solution, nb_tasks, colors, obj)