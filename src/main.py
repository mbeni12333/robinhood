import algos
import utils
import numpy as np

nb_tasks = 20

instance, erreur = utils.generateInstance(nb_tasks)
colors = np.random.rand(nb_tasks, 3)

solution = algos.PREDICT(instance)
utils.plotGant(solution, nb_tasks, colors)


solution = algos.SPT(instance)
utils.plotGant(solution, nb_tasks, colors)