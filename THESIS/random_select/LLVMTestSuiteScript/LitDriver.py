#!/usr/bin/python3
import LitMimic as lm
import os
import sys
import multiprocessing
import shlex
import subprocess as sp
import progressbar
import smtplib
from time import gmtime, strftime, localtime

class EmailService:
    def send(self, To, Subject, Msg):
        TO = To
        SUBJECT = Subject
        TEXT = Msg
        # Gmail Sign In
        gmail_sender = 'sslab.cs.nctu@gmail.com'
        gmail_passwd = 'sslabasdgh'

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_sender, gmail_passwd)

        BODY = '\r\n'.join(['To: %s' % TO,
                            'From: %s' % gmail_sender,
                            'Subject: %s' % SUBJECT,
                            '', TEXT])
        log = lm.Logger()
        try:
            server.sendmail(gmail_sender, [TO], BODY)
            log.out('Email sent!\n')
        except:
            log.out('Error sending mail\n')
            log.err('Error sending mail\n')
        server.quit()


class LitRunner:
    def ExecCmd(self, cmd, ShellMode=False, NeedPrintStdout=False,
            NeedPrintStderr=True):
        log = lm.Logger()
        try:
            #Execute cmd
            p = sp.Popen(shlex.split(cmd), shell=ShellMode, stdout=sp.PIPE, stderr= sp.PIPE)
            out, err = p.communicate()
            p.wait()
            if NeedPrintStdout and out is not None:
                log.out(out.decode('utf-8'))
            if NeedPrintStderr and err is not None:
                log.out(err.decode('utf-8'))
        except Exception as e:
            log.err("Exception= {}".format(str(e)) + "\n")
            log.err("Command error: {}\n".format(cmd))
            if err is not None:
                log.err("Error Msg= {}\n".format(err.decode('utf-8')))
            log.err("----------------------------------------------------------\n")

    def run(self):
        Target = lm.TargetBenchmarks()
        Log = lm.Logger()

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
            Log.out("Run: {}".format(LitTargetDir))
            bar.update((idx / len(SuccessBuiltPath)) * 100)
            self.ExecCmd(cmd, ShellMode=False, NeedPrintStdout=True, NeedPrintStderr=True)
        os.chdir(pwd)

        #Send notification
        mail = EmailService()
        MailSubject = "LitDriver Done."
        Now = strftime("%Y-%m-%d %H:%M:%S", localtime())
        Content = "Finish time: " + Now + "\n"
        mail.send(To="jaredcjr.tw@gmail.com", Subject=MailSubject, Msg=Content)


if __name__ == '__main__':
    lit = LitRunner()
    lit.run()
