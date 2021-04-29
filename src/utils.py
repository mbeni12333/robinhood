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
        self.time_estimated = time_estimated
        self.remainingtime = time
        self.start = start
    
    def getWaitingTime(self, timestamp):
        """
        """
        return timestamp - self.start
    
    def reset(self):
        self.remainingtime = self.time
     
    def run(self, time):
        """
        run this task for time units
        """
        
        if(self.remainingTime < time):
            tmp = time - self.remainingTime
            self.remainingTime = 0
            return tmp
        
        self.remainingTime -= time        
        return 0
    
def evaluateObjective(solution):
    """
    solution: list(tuple(task_id, ))
    """
    
    return

normal_generator = lambda n: np.random.randn(n)*0.1
uniform_generator = lambda n: np.random.rand(n)
pareto_generator = lambda n: np.random.pareto(1.1, n)
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
    
    
def plotGant(solution, nb_tasks=10):

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig, ax = plt.subplots()

      
    # Setting labels for x-axis and y-axis
    ax.set_xlabel('time')
    ax.set_ylabel('task')
    
    ax.set_yticks(np.arange(nb_tasks))
    
    ax.grid()
    
    timestep = 0
    for chunk in solution:
        ax.broken_barh([(timestep, chunk[1])],
                        (chunk[0]-0.25, 0.5),
                        facecolors=colors[len(colors)%(chunk[0]+1)])
        timestep += chunk[1]

    plt.show()

if __name__ == "__main__":
    instance, erreur = generateInstance()
    instance_sorted = sorted(instance, key=lambda task: task.time)
    
    solution = [(task.id, task.time) for task in instance_sorted]
    plotGant(solution)
    