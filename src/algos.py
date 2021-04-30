import numpy as np
import utils


def SPT_PART2(instance, sortingFunction=lambda task: task.remainingTime):
    """
    """
    
    for task in instance:
        task.reset()
    
    instance_sorted = sorted(instance, key=lambda task: task.start)
    queue = sorted([task for task in instance_sorted if task.start == 0],
                   key=sortingFunction)
    i = len(queue)

    timestep = 0
    time_till_next_insert = 0
    solution = []
    
    
    # total of "len(instances)" insertion
    while i <= len(instance):
        
        if(i == len(instance)):
            # fake task just to handle last case scenario
            next_insertion_task = utils.Task(0, 0, 0, 1000000)
        else:
            # get the next task to be inserted
            next_insertion_task = instance_sorted[i]
            
        while True: 
            
            # how much time we have till the insert
            # when it reaches 0, we insert the next task
            time_till_next_insert = next_insertion_task.start - timestep

            if queue != []:
                # run the current task for maximum time_till_next_insert time
                current_task = queue.pop(0)
                runtime = current_task.run(time_till_next_insert)
                # keep track of the solution
                solution.append((current_task.id, runtime))
                # the real time passed
                timestep += runtime
                
                # if the task did not finish add it to the queue
                if(current_task.hasFinished == False):
                    queue.append(current_task)
                
            else:
                # if the queue is empty
                # move time direclty till the next insert
                timestep += time_till_next_insert
            
            if next_insertion_task.start <= timestep:
                # insert the task
                queue.append(next_insertion_task)
                i += 1
                # sort the queue
                queue = sorted(queue, key=sortingFunction)
                
                break
    
    return solution

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

