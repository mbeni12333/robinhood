import numpy as np

def PREDICT(instance):
    """
    """
    instance_sorted = sorted(instance, key=lambda task: task.time_estimated)
    
    solution = [(task.id, task.time) for task in instance_sorted]
    
    return solution

def SPT(instance):
    """
    """
    instance_sorted = sorted(instance, key=lambda task: task.time)
    
    solution = [(task.id, task.time) for task in instance_sorted]
    
    return solution

def ROUND_ROBIN(instance):
    """
    """
    
    for task in instance:
        task.reset()
    
    queue = [task for task in instance]
    solution = []
    timestep = 0
    while queue != []:
        quantum = 1/len(queue)
        
        current_task = queue.pop(0)
        runtime = current_task.run(quantum)
        timestep += runtime
        
        solution.append((current_task.id, runtime))
        
        # task ended , remove from queue
        if runtime == quantum:
            queue.append(current_task)
            
    return solution

def PRED_ROUND_ROBIN():
    return

