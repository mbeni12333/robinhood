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

def ROUND_ROBIN():
    return

def PRED_ROUND_ROBIN():
    return

