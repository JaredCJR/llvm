#!/usr/bin/python3
import RandomDriver as RD
import time
import sys

if __name__ == '__main__':
    round = None
    if len(sys.argv) > 1:
        round = int(sys.argv[1])
    else:
        round = 6 # About two days
    print("Remember to set 'clang' and 'clang++' to customized build.")
    print("Run {} rounds = about {} days".format(round, round*8.0/24.0))
    print("It will start after 5 secs...")
    time.sleep(5)
    for i in range(round):
        execute = RD.Executer()
        execute.run()
    print("All Rounds={} are done!".format(round))
