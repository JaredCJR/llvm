#!/usr/bin/python3

class BenchmarkList:
    # numbers of random pass set execution
    repeat = 20

    """
    path is relative to "llvm_source/DSOAO/random_select"
     ["directory path", "build command", [available benchmarks commands], "clean command"]
     [available benchmarks commands] can be split into three parts:
                                    ["label name", "commands", "expected last outputs"]
    """
    benchmark_build_run_list = [
        ["../benchmark/helloworld" ,"clang -O1 -o hello hello.c", [["hello", "./hello", "function 2"], ["ls","ls", "hello.c"]], "rm hello"],
        #["../benchmark/botan" ,"make -j14", [["botan-all", "./botan-test", "all tests ok"], ], "make clean"],
    ]
    def genList(self):
        return self.benchmark_build_run_list

    def getRepeat(self):
        return self.repeat

