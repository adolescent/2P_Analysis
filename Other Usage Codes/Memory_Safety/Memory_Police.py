# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 14:53:47 2019

@author: ZR

This Code will supervise Memory usage of Operation System. If less than 5% memory usable, this code will force logout current user.
"""


from psutil import virtual_memory
import time
import os

mem = virtual_memory()
usage_percent = mem.percent
f = open('Log_File.txt','w')

while 1:
    f = open('Log_File.txt','a')
    print('Current Used Memory Percent is:'+str(usage_percent)+'%')
    f.write(str(usage_percent)+'\n')
    time.sleep(1)
    if usage_percent>95:
        now = int(round(time.time()*1000))
        now_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
        print('Memory Out!! Logging User Out at '+now_time)
        f.close()
        os.system("shutdown -l")
    else:
        f.close()