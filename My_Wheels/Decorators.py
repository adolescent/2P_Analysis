# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:19:13 2021

@author: ZR
A timer decorator.

"""
import time

def Timer(func):
    def core_func(*args,**kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('%s Cost: %.3f s' %(func.__name__,(end_time-start_time)))
        return result
    return core_func


def Log(func):
    pass