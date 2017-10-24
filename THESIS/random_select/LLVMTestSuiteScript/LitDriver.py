#!/usr/bin/python3
import LitMimic as lm
import ServiceLib as sv
import os
import sys
import multiprocessing
import shlex
import subprocess as sp
import progressbar
import smtplib

class LitRunner:
    def ExecCmd(self, cmd, ShellMode=False, NeedPrintStdout=False,
            NeedPrintStderr=True):
        Log = sv.LogService()
        try:
            #Execute cmd
            p = sp.Popen(shlex.split(cmd), shell=ShellMode, stdout=sp.PIPE, stderr= sp.PIPE)
            out, err = p.communicate()
            p.wait()
            if NeedPrintStdout and out is not None:
                Log.out(out.decode('utf-8'))
            if NeedPrintStderr and err is not None:
                Log.out(err.decode('utf-8'))
        except Exception as e:
            Log.err("----------------------------------------------------------\n")
            Log.err("Exception= {}".format(str(e)) + "\n")
            Log.err("Command error: {}\n".format(cmd))
            if err is not None:
                Log.err("Error Msg= {}\n".format(err.decode('utf-8')))
            Log.err("----------------------------------------------------------\n")

    def run(self):
        timeFile = os.getenv('LLVM_THESIS_Random_LLVMTestSuite_Results') + "/TimeStamp"
        if os.path.isfile(timeFile):
            os.remove(timeFile)

        time = sv.TimeService()
        StartDateTime = time.GetCurrentLocalTime()
        Target = lm.TargetBenchmarks()
        Log = sv.LogService()

        #build target tests
        pwd = os.getcwd()
        CoreNum = str(multiprocessing.cpu_count())
        for RootPath in Target.TargetPathList:
            os.chdir(RootPath)
            self.ExecCmd("make clean", ShellMode=True)
            self.ExecCmd("make -j" + CoreNum, ShellMode=True,
                    NeedPrintStdout=True, NeedPrintStderr=True)
        os.chdir(pwd)

        #place the corresponding feature extractor
        actor = lm.LitMimic()
        SuccessBuiltPath = actor.run()

        #execute it one after another with "lit"
        lit = os.getenv('LLVM_THESIS_lit', "Error")
        if lit == "Error":
            Log.err("Please setup \"lit\" environment variable.\n")
            sys.exit("lit is unknown\n")
        bar = progressbar.ProgressBar(redirect_stdout=True)
        for idx, LitTargetDir in enumerate(SuccessBuiltPath):
            os.chdir(LitTargetDir)
            cmd = lit + " -j1 -q ./"
            Log.out("Run: {}\n".format(LitTargetDir))
            bar.update((idx / len(SuccessBuiltPath)) * 100)
            self.ExecCmd(cmd, ShellMode=False, NeedPrintStdout=True, NeedPrintStderr=True)
        os.chdir(pwd)


        #calculate used time
        EndDateTime = time.GetCurrentLocalTime()
        DeltaDateTime = time.GetDeltaTimeInDate(StartDateTime, EndDateTime)

        #Send notification
        mail = sv.EmailService()
        MailSubject = "LitDriver Done."
        Content = "Start date time: " + StartDateTime + "\n"
        Content += "Finish date time: " + EndDateTime + "\n"
        Content += "Whole procedure takes \"{}\"\n".format(DeltaDateTime)
        Content += "-------------------------------------------------------\n"
        Content += "Error Msg:\n"
        try:
            with open(Log.ErrorFilePath, 'r') as file:
                Content += file.read()
                file.close()
        except Exception as e:
            Content += "Nothing wrong!\n"

        mail.send(To="jaredcjr.tw@gmail.com", Subject=MailSubject, Msg=Content)


if __name__ == '__main__':
    lit = LitRunner()
    lit.run()
