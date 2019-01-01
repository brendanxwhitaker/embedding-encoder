import multiprocessing
import numpy
import time
import sys

def ayyy():
    
    name = multiprocessing.current_process().name
    print name, 'Starting'
    print("Ayyyyyyyyyy")
    print name, 'Exiting'
    return

def oooh():
    
    name = multiprocessing.current_process().name
    print name, 'Starting'
    print("Ooooooooohh")
    print name, 'Exiting'
    return

ayyy_process = multiprocessing.Process(name="ayyy",target=ayyy)
oooh_process = multiprocessing.Process(name="oooh",target=oooh)

ayyy_process.start()
oooh_process.start()

