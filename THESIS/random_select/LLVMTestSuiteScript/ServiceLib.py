#!/usr/bin/python3
import os
import sys
from time import gmtime, strftime, localtime
from datetime import datetime, date, timedelta
import LitMimic as lm

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
    ErrorFilePath = None
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
            mail.SignificantNotification(To="jaredcjr.tw@gmail.com", Msg="Log dir=\"{}\"\n".format(Loc))
        else:
            os.system("mkdir -p "+ Loc)

        self.RecordFilePath = Loc + '/' + self.time
        self.out("Record Results to {}\n".format(self.RecordFilePath))
        self.ErrorFilePath = Loc + '/' + self.time + "_Error"
        self.out("Record Error to {}\n".format(self.ErrorFilePath))

    def out(self, msg):
        print(msg, end="")

    def err(self, msg):
        #save to same error file for every instance
        with open(self.ErrorFilePath, "a") as file:
            file.write(msg)
            file.close()

    def record(self, msg):
        #save to same error file for every instance
        with open(self.RecordFilePath, "a") as file:
            file.write(msg)
            file.close()

class EmailService:
    def send(self, To, Subject, Msg):
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
        log = lm.Logger()
        try:
            server.sendmail(gmail_sender, [TO], BODY)
            log.out('Email sent!\n')
        except:
            log.out('Error sending mail\n')
            log.err('Error sending mail\n')
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


