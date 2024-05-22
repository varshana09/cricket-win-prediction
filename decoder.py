import time
import numpy as np
import argparse
parser = argparse.ArgumentParser()

if __name__ == "__main__":
    parser.add_argument("--value-policy", type=str, default="-")
    parser.add_argument("--states", type=str, default="-")
    args = parser.parse_args()
    with open(args.states, "r") as f:
        lines = f.readlines()
    S = len(lines)
    states = []
    actions = [0, 1, 2, 4, 6]
    id = dict()
    idx = 0
    for line in lines:
        balls, runs = int(line.strip()[:2]), int(line.strip()[2:])
        states.append((balls, runs))
        id[(balls, runs, 0)] = idx
        id[(balls, runs, 1)] = S + idx
        idx += 1
    with open(args.value_policy, "r") as f:
        lines = f.readlines()
    lineno = 1
    for line in lines:
        if(lineno > S):
            break
        balls,runs = states[lineno - 1]
        prob,action = line.strip().split()[0],actions[int(line.strip().split()[1])]
        print(str(balls).zfill(2)+str(runs).zfill(2),action,prob)
        lineno+=1
    
