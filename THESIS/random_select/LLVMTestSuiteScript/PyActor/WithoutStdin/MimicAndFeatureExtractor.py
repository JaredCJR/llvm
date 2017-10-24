#!/usr/bin/python3
import inspect
import os
import sys
import shlex
import subprocess as sp
import time
from time import gmtime, strftime
import ServiceLib as sv

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(sv.LogService):
    #Should never print to stdout, "lit" will get unexpected output
    def out(self, msg):
        pass

class Executer:
    Args = None
    def __init__(self, args):
        self.Args = args

    def run(self):
        Log = Logger()
        elfPath = inspect.getfile(inspect.currentframe())
        #Remove the postfix ".py"
        elfPath = elfPath[:-3]
        RealElfPath = elfPath + ".OriElf"
        Cmd = RealElfPath + " " + self.Args
        TimeList = []
        Repeat = 5
        try:
            for i in range(Repeat):
                err = None
                DropLoc = os.getenv('LLVM_THESIS_RandomHome')
                os.system(DropLoc + "/LLVMTestSuiteScript/DropCache/drop")
                StartTime = time.perf_counter()
                p = sp.Popen(shlex.split(Cmd), stdout = sp.PIPE, stderr= sp.PIPE)
                out, err = p.communicate()
                p.wait()
                EndTime = time.perf_counter()
                TimeList.append(EndTime - StartTime)
        except Exception as ex:
            if err is not None:
                Log.err(err.decode('utf-8'))
            else:
                Log.err("Why exception happend, and err is None?\n")
                Log.err(str(ex) + "\n")
            return

        TimeList.sort()

        #Output for "lit"
        p = sp.Popen(shlex.split(Cmd))
        ReturnCode = p.wait()
        with open("./ReturnValue", "w") as file:
            file.write(str(ReturnCode))
            file.close()

        RandomSetLoc = os.getenv('LLVM_THESIS_RandomHome') + "/InputSet"
        with open(RandomSetLoc, "r") as file:
            RandomSet = file.read().rstrip()
            file.close()

        BenchmarkName = sv.BenchmarkNameService()
        BenchmarkName = BenchmarkName.GetFormalName(elfPath)
        LogTime = TimeList[len(TimeList)//2]
        log_msg = BenchmarkName + ", " + RandomSet + ", " + str(LogTime) + "\n"
        Log.record(log_msg)


if __name__ == '__main__':
    exec = Executer(' '.join(str(arg) for arg in sys.argv[1:]))
    exec.run()
