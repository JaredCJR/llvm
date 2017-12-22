#!/usr/bin/python3
import RandomGenerator as RG
import shlex
import subprocess as sp
import os
import time
from time import gmtime, strftime

class Executer:
    # numbers of random pass set execution
    repeat = 30
    """ 
    path is relative to "llvm_source/DSOAO/random_select"
     ["directory path", "build command", [available benchmarks commands], "clean command"]
     [available benchmarks commands] can be split into three parts:
                                    ["label name", "commands", "expected last outputs"]
    """
    benchmark_build_run_list = [
        ["benchmark/helloworld" ,"clang -O1 -o hello hello.c", [["hello", "./hello", "function 2"], ["ls","ls", "hello.c"]], "rm hello"],
        #["benchmark/botan" ,"make -j14", [["botan-all", "./botan-test", "all tests ok"], ], "make clean"],
    ]
    def run(self):
        TestingStart = time.perf_counter()
        localtime = time.strftime("%m%d-%H:%M", time.localtime())
        prev_cwd = os.getcwd()
        os.system("mkdir -p results_time")
        Result_FileLoc = prev_cwd + "/results_time/result_" + localtime
        ErrorResult_FileLoc = Result_FileLoc + "_Error"
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
                cwd = prev_cwd + "/" + build_bench[0].rstrip()
                os.chdir(cwd)
                print("Build Start*******************************************")
                print("clean=\"{}\"".format(build_bench[3]))
                try:
                    p = sp.Popen(shlex.split(build_bench[3]))
                    p.wait()
                except:
                    print("Clean the previous built failed. Why?")
                    Result_File.write("----------------------------------\n")
                    Result_File = open(ErrorResult_FileLoc, "a")
                    Result_File.write(
                            "Clean Error:\"{}\"".format(build_bench[3]))
                    Result_File.write("----------------------------------\n")
                    Result_File.close()
                    
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
                    Result_File.write("----------------------------------\n")
                    Result_File = open(ErrorResult_FileLoc, "a")
                    Result_File.write(
                            "Build Error:\"{}\"".format(build_bench[1]))
                    Result_File.write(
                            "Build with this combination error:\"{}\"".format(target_file.read()))
                    Result_File.write("----------------------------------\n")
                    Result_File.close()
                    target_file.close()
                    continue
                print("Build End*******************************************")
                #Run built benchmarks
                for run_single in build_bench[2]:
                    #run command
                    print("<<<<<<  RUN=\"{}\"  >>>>>>".format(run_single))
                    period = 0.0
                    try:
                        start_time = time.perf_counter()
                        p = sp.Popen(shlex.split(run_single[1]),stdout = sp.PIPE, stderr= sp.PIPE)
                        out, err = p.communicate()
                        p.wait()
                        end_time = time.perf_counter()
                        period = end_time - start_time
                        #Verify expected output, strip out the newline characters
                        if out.rstrip().endswith(run_single[2].rstrip().encode('utf-8')):
                            print("<<<<<<  Verify Success  >>>>>>")
                        else:
                            print("Verify Failed:")
                            print("stdout=\"{}\"\nstderr=\"{}\"".format(
                                out.decode('utf-8'),err.decode('utf-8')))
                            print("Expected Last stdout={}".format(run_single[2]))
                            raise Exception
                    except:
                        print("Run the benchmark={} failed".format(run_single))
                        Result_File = open(ErrorResult_FileLoc, "a")
                        Result_File.write("----------------------------------\n")
                        Result_File.write("Run the benchmark={} failed".format(run_single))
                        Result_File.write("stdout=\"{}\"\n\n stderr=\"{}\"".format(
                                out.decode('utf-8'),err.decode('utf-8')))
                        Result_File.write("----------------------------------\n")
                        Result_File.close()
                        continue
                    Result_File = open(Result_FileLoc, "a")
                    Result_File.write("{},{}\n".format(run_single[0].rstrip(), period))
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
