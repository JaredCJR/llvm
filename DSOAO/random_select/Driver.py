#!/usr/bin/python3
import RandomGenerator as RG
import shlex
import subprocess as sp
import os
import time
from time import gmtime, strftime

class Executer:
    # numbers of random pass set execution
    repeat = 60
    #path is relative to "llvm_source/DSOAO/random_select"
    # ["directory path", "build command", [available benchmarks commands], "clean command"]
    benchmark_build_run_list = [
        ["benchmark/helloworld" ,"clang -O1 -o hello hello.c", ["./hello", "ls"], "rm hello"],
        #["benchmark/botan" ,"make -j10", ["./botan-test", ], "make clean"],
    ]
    def run(self):
        TestingStart = time.perf_counter()
        localtime = time.strftime("%m%d-%H:%M", time.localtime())
        prev_cwd = os.getcwd()
        Result_FileLoc = prev_cwd + "/results_time/result_" + localtime
        Result_File = open(Result_FileLoc, "w")
        Result_File.close()
        for i in range(self.repeat):
            print("Iteration Start-----------------------------------------")
            #generate pass set
            rg_driver = RG.Driver()
            #mean should between 0~1
            mean = (i + 1) / (self.repeat + 1)
            rg_driver.run(mean)

            #build benchmarks for this pass set
            for build_bench in self.benchmark_build_run_list:
                cwd = prev_cwd + "/" + build_bench[0]
                os.chdir(cwd)
                print("Build Start*******************************************")
                print("clean=\"{}\"".format(build_bench[3]))
                try:
                    p = sp.Popen(shlex.split(build_bench[3]))
                    p.wait()
                except:
                    print("Clean the previous built failed. Why?")
                print("build=\"{}\"".format(build_bench[1]))
                try:
                    p = sp.Popen(shlex.split(build_bench[1]))
                    p.wait()
                except:
                    print("Build with this combination error.")
                    file_loc = "/home/jrchang/workspace/llvm/DSOAO/random_select/InputSet"
                    target_file = open(file_loc, "r")
                    print(target_file.read())
                    print("-------------------------")
                    target_file.close()
                    continue
                print("Build End*******************************************")
                #Run built benchmarks
                for run_single in build_bench[2]:
                    #run command
                    print("<<<<<<  RUN=\"{}\"  >>>>>>".format(run_single))
                    start_time = time.perf_counter()
                    try:
                        p = sp.Popen(shlex.split(run_single))
                        p.wait()
                    except:
                        print("Run the benchmark={} failed".format(run_single))
                    end_time = time.perf_counter()
                    period = end_time - start_time
                    Result_File = open(Result_FileLoc, "a")
                    Result_File.write("{}\n".format(period))
                    Result_File.close()
                os.chdir(prev_cwd)
            print("Iteration End-----------------------------------------")
        #Print Finishing Results
        os.chdir(prev_cwd)
        print("Done. Results written to \"{}\"".format(Result_FileLoc))
        TestingEnd = time.perf_counter()
        TestingTime = TestingEnd-TestingStart
        hours, seconds =  TestingTime // 3600, TestingTime % 3600
        minutes, seconds = seconds // 60, seconds % 60
        print("Testing Total Time = {}:{}:{}".format(int(hours), int(minutes), int(seconds)))



if __name__ == '__main__':
    execute = Executer()
    execute.run()
