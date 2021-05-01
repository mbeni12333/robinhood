import algos
import utils
import numpy as np
import matplotlib.pyplot as plt

def simulate(numberInstances=500, tasksPerInstance=50):
    

    bars = []
    sigmas = [0.1, 0.5, 1, 5]
    params = [1.1, 1.5, 2.0, 10.0, 50.0]
    
    for a in params:
        
        #eps_generator = lambda n: np.random.randn(n)
        x_generator = lambda n: np.random.pareto(a, (n, ))
        
        points = np.zeros((numberInstances, 3))
        print(a)
        
        for i in range(numberInstances):
            #print(f"instance {i}")
            instance, erreur = utils.generateInstance(tasksPerInstance,
                                                      x_generator=x_generator)
            
            solution = algos.PREDICT_PART2(instance)
            obj1 = utils.evaluateObjective(instance, solution)
            #print("inished spt")
            solution = algos.PREDICT_PART2(instance, lambda task: task.time_estimated)
            obj2 = utils.evaluateObjective(instance, solution)
            #print("inished predict")
            solution = algos.ROUND_ROBIN(instance)
            obj3 = utils.evaluateObjective(instance, solution)
            #print("inished round robin")
            #rapportDeCompettetivite = obj/obj_optimale
            
            points[i] = np.array([1, obj2/obj1, obj3/obj1])

        
        bars.append(points.max(0))
    
    
    X = np.arange(len(params))
    bars = np.array(bars)
    colors = ["green", "blue", "red"]
    labels = ["SPT", "PREDICT", "ROUND_ROBIN"] 
    
    for i in range(3):
        plt.bar(X+0.25*i, bars[:, i], width=0.25, color=colors[i], label=labels[i])
    
    plt.ylabel("Rapport de compettetivite")
    plt.xlabel("pareto param")
    plt.xticks(X+0.25, params)
    plt.legend()
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