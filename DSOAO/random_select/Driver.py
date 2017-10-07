#!/usr/bin/python3
import RandomDriver as RD


if __name__ == '__main__':
    round = 5
    for i in range(round):
        execute = RD.Executer()
        execute.run()
    print("All Rounds={} are done!".format(round))
