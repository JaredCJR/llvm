#!/usr/bin/python3
import RandomDriver as RD
import time

if __name__ == '__main__':
    round = 5
    print("Remember to set 'clang' and 'clang++' to customized build.")
    print("It will start after 5 secs...")
    time.sleep(5)
    for i in range(round):
        execute = RD.Executer()
        execute.run()
    print("All Rounds={} are done!".format(round))
