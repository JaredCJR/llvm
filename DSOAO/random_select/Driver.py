#!/usr/bin/python3
import RandomDriver as RD
import time
import sys

if __name__ == '__main__':
    round = None
    if len(sys.argv) > 1:
        round = int(sys.argv[1])
    else:
        round = 5 # About 1 day
    HoursPerRound = 4.5
    print("Remember to set 'clang' and 'clang++' to \"Customized Release\" build.")
    print("Run {0} rounds = about {1:5.2f} day(s) --> ({2:7.1f} hours)".format(
        round, round*HoursPerRound/24.0, round*HoursPerRound))
    print("It will start after 5 secs...")
    time.sleep(5)
    for i in range(round):
        execute = RD.Executer()
        execute.run()
    print("All Rounds={} are done!".format(round))
