import numpy as np
import matplotlib.pyplot as plt
    
class Task(object):
    """
    Task that has id, time, start time
    """
    
    def __init__(self, id=0, time=1, time_estimated=1, start=0):
        """
        """
        self.id = id
        self.time = time
        self.remainingTime = time
        
        self.time_estimated = time_estimated
        self.remainingTime_estimated = time_estimated
        
        self.start = start
        self.hasFinished = False
    
    def getWaitingTime(self, timestamp):
        """
        """
        return timestamp - self.start
    
    def reset(self):
        self.remainingTime = self.time
        self.remainingTime_estimated = self.time_estimated
        self.hasFinished = False
     
    def run(self, time):
        """
        run this task for time units
        """
        
        if(self.remainingTime <= time):
            tmp = self.remainingTime
            self.remainingTime = 0
            self.hasFinished = True
            return tmp
        
        self.remainingTime -= time 
        self.remainingTime_estimated -= time
        return time
    
def evaluateObjective(instance, solution):
    """
    solution: list(tuple(task_id, time))
    """
    
    # reset tasks
    for task in instance:
        task.reset()
        
    # sum end dates
    timestep = 0
    obj = 0
    for chunk in solution:
        runtime = instance[chunk[0]].run(chunk[1])
        
        if instance[chunk[0]].remainingTime == 0:
            obj += timestep
        timestep += chunk[1]
    
    return obj

normal_generator = lambda n: np.random.randn(n)
uniform_generator = lambda n: np.random.rand(n)
pareto_generator = lambda n: np.random.pareto(1.1, n)
poisson_generator = lambda n: np.random.poisson(1.0, (n, ))
zero_generator = lambda n: np.zeros((n, ))

def generateInstance(nb_tasks=10,
                     x_generator=pareto_generator,
                     eps_generator=normal_generator,
                     v_generator=zero_generator):
    
    X = x_generator(nb_tasks)
    eps = eps_generator(nb_tasks)
    Y = X + eps
    V = v_generator(nb_tasks)
    
    erreur = eps.sum()
    
    return [Task(i, xi, yi, vi) for i, (xi, yi, vi) in enumerate(zip(X, Y, V))], erreur
    
    
def plotGant(solution, nb_tasks=10, colors=None, title=None):
#
#    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if colors is None:
        colors = np.random.rand(nb_tasks, 3)
        
    fig, ax = plt.subplots(figsize=(nb_tasks, 20))

      
    # Setting labels for x-axis and y-axis
    ax.set_xlabel('time')
    ax.set_ylabel('task')
    
    ax.set_yticks(np.arange(nb_tasks))
    
    ax.grid()
    
    timestep = 0
    for chunk in solution:
        ax.broken_barh([(timestep, chunk[1])],
                        (chunk[0]-0.25, 0.5),
                        facecolors=colors[chunk[0]])
        timestep += chunk[1]

    if title is not None:
        plt.title(f"Objective : {title:0.2f}", fontsize=32)
    plt.show()
    