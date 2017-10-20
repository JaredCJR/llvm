#!/usr/bin/python3
import inspect
import os
import sys
import shlex
import subprocess as sp
import time
from time import gmtime, strftime

#Should never print to stdout, "lit" will get unexpected output
class Logger:
    def out(self, msg):
        print(msg)

    def err(self, msg):
        print(msg)

class Executer:
    Args = None
    def __init__(self, args):
        self.Args = args

    def run(self):
        log = Logger()
        elfPath = inspect.getfile(inspect.currentframe())
        #Remove the postfix ".py"
        elfPath = elfPath[:-3]
        RealElfPath = elfPath + ".OriElf"
        Cmd = RealElfPath + " " + self.Args
        TimeList = []
        Repeat = 1
        try:
            for i in range(Repeat):
                err = None
                os.system("/home/jrchang/workspace/llvm/DSOAO/random_select/LLVMTestSuiteScript/DropCache/drop")
                StartTime = time.perf_counter()
                p = sp.Popen(shlex.split(Cmd), stdout = sp.PIPE, stderr= sp.PIPE)
                out, err = p.communicate()
                p.wait()
                EndTime = time.perf_counter()
                TimeList.append(EndTime - StartTime)
        except:
            if err is not None:
                log.err(err.decode('utf-8'))
            else:
                log.err("Why exception happend, and err is None?\n")
            return

        TimeList.sort()

        #Output for "lit"
        p = sp.Popen(shlex.split(Cmd))
        ReturnCode = p.wait()
        with open("./ReturnValue", "w") as file:
            file.write(str(ReturnCode))
            file.close()

        #TODO
        LogTime = TimeList[len(TimeList)//2]


if __name__ == '__main__':
    exec = Executer(' '.join(str(arg) for arg in sys.argv[1:]))
    exec.run()
