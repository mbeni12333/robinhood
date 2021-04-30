import algos
import utils
import numpy as np
import matplotlib.pyplot as plt

def simulate(numberInstances=1000, tasksPerInstance=50):
    
    points = np.zeros((numberInstances, 2))
    
    for i in range(numberInstances):
    
        instance, erreur = utils.generateInstance(tasksPerInstance,
                                                  eps_generator=lambda n:np.random.randn(n),
                                                  v_generator=utils.poisson_generator)
        
        solution = algos.PREDICT_PART2(instance)
        obj_optimale = utils.evaluateObjective(instance, solution)
    
        solution = algos.PREDICT_PART2(instance, lambda task: task.remainingTime_estimated)
        obj = utils.evaluateObjective(instance, solution)
    
        rapportDeCompettetivite = obj/obj_optimale
        
        points[i, 0] = erreur
        points[i, 1] = rapportDeCompettetivite
        
    plt.scatter(points[:, 0], points[:, 1])
    plt.ylabel("Rapport de compettetivite")
    plt.xlabel("Erreur")
    plt.show()

simulate()



# nb_tasks = 10

# instance, erreur = utils.generateInstance(nb_tasks, v_generator=utils.poisson_generator)
# colors = np.random.rand(nb_tasks, 3)

# # solution = algos.PREDICT(instance)
# # obj = utils.evaluateObjective(instance, solution)
# # utils.plotGant(solution, nb_tasks, colors, obj)

# # solution = algos.SPT(instance)
# # obj = utils.evaluateObjective(instance, solution)
# # utils.plotGant(solution, nb_tasks, colors, obj)


# solution = algos.PREDICT_PART2(instance)
# obj = utils.evaluateObjective(instance, solution)
# utils.plotGant(solution, nb_tasks, colors, obj)

# solution = algos.PREDICT_PART2(instance, lambda task: task.remainingTime_estimated)
# obj = utils.evaluateObjective(instance, solution)
# utils.plotGant(solution, nb_tasks, colors, obj)


# # solution = algos.ROUND_ROBIN(instance)
# # obj = utils.evaluateObjective(instance, solution)
# # utils.plotGant(solution, nb_tasks, colors, obj)