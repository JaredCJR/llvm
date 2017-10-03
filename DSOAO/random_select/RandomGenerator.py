#!/usr/bin/python3
from random import SystemRandom
import random

class Candidate:
    def generator(self):
        return list(range(1, 34+1))

class RandomPick:
    def run(self, list):
        res_list = []
        res_list.append(list[0])
        for num in list[1:]:
            if random.uniform(0, 1) > 0.5:
                res_list.append(num)
        return res_list

class Driver:
    def run(self):
        cand = Candidate()
        #l is the ordered candidates
        l = cand.generator() 
        cryptorand = SystemRandom()
        #shuffle it to create randomness
        for i in range(10):
            cryptorand.shuffle(l)
            rand = RandomPick() 
            #pick at least one as candidate
            target_list = rand.run(l)
            #shuffle to keep first does not always the same
            cryptorand.shuffle(target_list)
            """
            print(target_list)
            """
            #open file with buffering, text is not able to get rid of buffering
            target_file = open("/home/jrchang/workspace/llvm/DSOAO/random_select/InputSet", "w")
            for ps in target_list:
                target_file.write("%s " % ps)
            target_file.write("\n")
            target_file.close()
            """
            target_file = open("/home/jrchang/workspace/llvm/experiment/random_select/InputSet", "r")
            print(target_file.read())
            target_file.close()
            """

if __name__ == '__main__':
    drv = Driver()
    drv.run()
