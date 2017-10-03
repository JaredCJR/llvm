#!/usr/bin/python3
import RandomGenerator as RG
import os

class Executer:
    def run(self):
        for i in range(10):
            #generate pass set
            rg_driver = RG.Driver()
            rg_driver.run()
            #run command
            print("---%d---" % i)
            os.system("clang -O1 -o benchmark/hello benchmark/hello.c")



if __name__ == '__main__':
    execute = Executer()
    execute.run()
