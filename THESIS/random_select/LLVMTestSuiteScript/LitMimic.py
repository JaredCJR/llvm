#!/usr/bin/python3
"""
This file will replace the real elf with PyActor to extract features
"""
import os
import shutil


class TargetBenchmarks:
    #The last character in BuiltPath must be '/'
    BuiltPath = "/home/jrchang/workspace/llvm/test-suite/build/"
    #Target dir lists, format: ["First level dir in BuiltPath", ["List of second level dir"]]
    SingleSource = ["SingleSource", ["Benchmarks", ]]
    MultiSource = ["MultiSource", ["Applications", "Benchmarks", ]]
    #Add all source together
    TargetDirLists = [SingleSource, MultiSource, ]

class Logger:
    def out(self, msg):
        print(msg)

    def err(self, msg):
        print(msg)

class LitMimic:
    BuiltPath = None
    TargetDirLists = None
    #Make sure that the elf already exists
    PyActorLoc_withStdin = "./PyActor/WithStdin/MimicAndFeatureExtractor.py"
    PyActorLoc_withoutStdin = "./PyActor/WithoutStdin/MimicAndFeatureExtractor.py"
    PyCallerLoc_withStdin = "./PyActor/WithStdin/PyCaller"
    PyCallerLoc_withoutStdin = "./PyActor/WithoutStdin/PyCaller"

    def __init__(self):
        target = TargetBenchmarks()
        self.BuiltPath = target.BuiltPath
        self.TargetDirLists = target.TargetDirLists

    def run(self):
        CombinedPath = []
        BenchmarkPath = []
        log = Logger()
        for Dir in self.TargetDirLists:
            for SubDir in Dir[1]:
                path = self.BuiltPath + Dir[0] + "/" + SubDir + "/"
                CombinedPath.append(path)
        for RootPath in CombinedPath:
            for root, dirs, files in os.walk(RootPath):
                for file in files:
                    test_pattern = '.test'
                    if file.endswith(test_pattern):
                        #Does this benchmark need stdin?
                        NeedStdin = False
                        TestFilePath = os.path.join(os.path.abspath(root), file)
                        with open(TestFilePath, "r") as TestFile:
                            for line in TestFile:
                                if line.startswith("RUN:"):
                                    if line.find(" < ") != -1:
                                        NeedStdin = True
                        #Do what we want: rename elf and copy actor
                        ElfName = file.replace(test_pattern, '')
                        ElfPath = os.path.join(root, ElfName)
                        NewElfName = ElfName + ".OriElf"
                        NewElfPath = os.path.join(root, NewElfName)
                        #based on "stdin" for to copy the right ones
                        if NeedStdin == True:
                            PyCallerLoc = self.PyCallerLoc_withStdin
                            PyActorLoc = self.PyActorLoc_withStdin
                        else:
                            PyCallerLoc = self.PyCallerLoc_withoutStdin
                            PyActorLoc = self.PyActorLoc_withoutStdin
                        #copy it
                        if os.path.exists(ElfPath) == True:
                            #rename the real elf
                            shutil.move(ElfPath, NewElfPath)
                            #copy the feature-extractor
                            shutil.copy2(PyActorLoc, ElfPath + ".py")
                            #copy the PyCaller
                            if os.path.exists(PyCallerLoc) == True:
                                shutil.copy2(PyCallerLoc, ElfPath)
                            else:
                                log.err("Please \"$ make\" to get PyCaller in {}\n".format(PyCallerLoc))
                                return
                        else:
                            log.err("This elf={} filed to build?\n".format(ElfPath))



if __name__ == '__main__':
    actor = LitMimic()
    actor.run()


