#!/usr/bin/python3
import os
import sys
from time import gmtime, strftime, localtime
from datetime import datetime, date, timedelta
import LitMimic as lm
import smtplib

class TimeService:
    DateTimeFormat = "%Y%m%d_%H-%M-%S"
    def GetCurrentLocalTime(self):
        return strftime(self.DateTimeFormat, localtime())

    def GetDeltaTimeInDate(self, prev, post):
        t1 = datetime.strptime(prev, self.DateTimeFormat)
        t2 = datetime.strptime(post, self.DateTimeFormat)
        delta = t2 - t1
        return delta

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

#Singletonn
class LogService(metaclass=Singleton):
    StderrFilePath = None
    StdoutFilePath = None
    RecordFilePath = None
    time = None
    def __init__(self):
        timeFile = os.getenv('LLVM_THESIS_Random_LLVMTestSuite_Results') + "/TimeStamp"
        if self.time is None:
            if os.path.isfile(timeFile):
                with open(timeFile, 'r') as file:
                    self.time = file.read()
                    file.close()
            else:
                time = TimeService()
                self.time = time.GetCurrentLocalTime()
                with open(timeFile, 'w') as file:
                    file.write(self.time)
                    file.close()

        Loc = os.getenv('LLVM_THESIS_Random_LLVMTestSuite_Results', "/tmp")
        if(Loc == "/tmp"):
            mail = drv.EmailService()
            mail.SignificantNotification(Msg="Log dir=\"{}\"\n".format(Loc))
        else:
            os.system("mkdir -p "+ Loc)

        self.StdoutFilePath = Loc + '/' + self.time + "_STDOUT"
        self.out("Record Stdout to {}\n".format(self.StdoutFilePath))
        self.StderrFilePath = Loc + '/' + self.time + "_STDERR"
        self.out("Record Stderr to {}\n".format(self.StderrFilePath))
        self.RecordFilePath = Loc + '/' + self.time + "_Time"
        self.out("Record Results to {}\n".format(self.RecordFilePath))

    def out(self, msg):
        print(msg, end="")
        #save to same file for every instance
        with open(self.StdoutFilePath, "a") as file:
            file.write(msg)
            file.close()

    def err(self, msg):
        #save to same error file for every instance
        with open(self.StderrFilePath, "a") as file:
            file.write(msg)
            file.close()

    def record(self, msg):
        #save to same error file for every instance
        with open(self.RecordFilePath, "a") as file:
            file.write(msg)
            file.close()

class EmailService:
    def send(self, Subject, Msg, To="jaredcjr.tw@gmail.com"):
        TO = To
        SUBJECT = Subject
        TEXT = Msg
        # Gmail Sign In
        gmail_sender = 'sslab.cs.nctu@gmail.com'
        gmail_passwd = '2018graduate'

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_sender, gmail_passwd)

        BODY = '\r\n'.join(['To: %s' % TO,
                            'From: %s' % gmail_sender,
                            'Subject: %s' % SUBJECT,
                            '', TEXT])
        Log = LogService()
        try:
            server.sendmail(gmail_sender, [TO], BODY)
            Log.out('Email sent!\n')
        except:
            Log.out('Error sending mail\n')
            Log.err('Error sending mail\n')
        server.quit()

    def SignificantNotification(self, To, Msg):
        MailSubject = "LitDriver Notification."
        self.send(To=To, Subject=MailSubject, Msg=Msg)

class BenchmarkNameService:
    def ReplaceWithDash(self, str):
        ret = ""
        for c in str:
            if c != '/':
                ret += c
            else:
                ret += '-'
        return ret

    def GetFormalName(self, full_path):
        tests = lm.TargetBenchmarks()
        ret = full_path
        for RemoveWords in tests.TargetPathList:
            if ret.startswith(RemoveWords):
                ret = ret[len(RemoveWords):]
                break
        if ret.startswith("./"):
            ret = ret["./"]
        return self.ReplaceWithDash(ret)

    def RemoveFailureRecords(self, StdoutFile, RecordFile):
        Log = LogService()
        FailList = []
        TargetPrefix = "    test-suite :: "
        with open(StdoutFile, 'r') as file:
            for line in file:
                if line.startswith(TargetPrefix) :
                    FailList.append(line[len(TargetPrefix):])
            file.close()
        #get the targets
        FailNameList = []
        for line in FailList:
            PathPrefix = line[:line.rfind('/')]
            PathPostfix = line[line.rfind('/') + 1:]
            NamePostfix = PathPrefix[PathPrefix.rfind('/') + 1:] + '-' + PathPostfix
            #remove .test
            NamePostfix = NamePostfix[:-(len(".test") + 1)]
            FailNameList.append(NamePostfix.rstrip())
        #remove form records
        NewRecord = ""
        with open(RecordFile, 'r') as file:
            for line in file:
                LinePrefix = line[:line.find(',')]
                RmFlag = False
                for RemoveTarget in FailNameList:
                    if LinePrefix.endswith(RemoveTarget):
                        RmFlag = True
                        break
                if RmFlag == False:
                    NewRecord += line
            file.close()

        #replace the original one
        with open(RecordFile, 'w') as file:
            file.write(NewRecord)
            file.close()


