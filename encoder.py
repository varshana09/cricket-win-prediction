import time
import numpy as np
import argparse
parser = argparse.ArgumentParser()

if __name__ == "__main__":
    parser.add_argument("--states", type=str, default="-")
    parser.add_argument("--parameters", type=str, default="-")
    parser.add_argument("--q", type=float, default="-")
    args = parser.parse_args()
    q = args.q
    with open(args.states, "r") as f:
        lines = f.readlines()
    S = len(lines)
    states = []
    id = dict()
    idx = 0
    for line in lines:
        balls, runs = int(line.strip()[:2]), int(line.strip()[2:])
        states.append((balls, runs))
        id[(balls, runs, 0)] = idx
        id[(balls, runs, 1)] = S + idx
        idx += 1
    actions = [0, 1, 2, 4, 6]
    outcomes = [-1, 0, 1, 2, 3, 4, 6]
    actions_idx = {0: 0, 1: 1, 2: 2, 4: 3, 6: 4}
    b_params = {-1: q, 0: (1-q)/2, 1: (1-q)/2}
    with open(args.parameters, "r") as f:
        lines = f.readlines()
    params = []
    for i in range(1, len(lines)):
        line = lines[i]
        params.append(list(map(float, line.strip().split()[1:])))
    Transitions = []
    probs = dict()
    for balls, runs in states:
        s1 = id[(balls, runs, 0)]
        for i in range(len(params)):
            ac = actions_idx[actions[i]]
            for j in range(len(params[i])):
                if(params[i][j] == 0):
                    continue
                prob = params[i][j]
                reward = 0
                if(outcomes[j] == -1):
                    s2 = 2 * S + 1
                else:
                    n_balls, n_runs = balls - 1, runs - outcomes[j]
                    if(n_runs <= 0):
                        s2 = 2 * S
                        reward = 1
                    else:
                        if(n_balls == 0):
                            s2 = 2 * S + 1
                        else:
                            strike_change = int(((outcomes[j] % 2 == 0 and n_balls % 6 == 0)
                                                 or (outcomes[j] % 2 != 0 and n_balls % 6 != 0)))

                            s2 = id[(n_balls, n_runs, strike_change)]
                if(not ((s1,ac,s2) in probs)):
                    probs[(s1,ac,s2)] = 0
                probs[(s1,ac,s2)] += prob
        s1 = id[(balls, runs, 1)]
        ac = 0
        reward = 0
        for run in b_params:
            prob = b_params[run]
            if(prob == 0):
                continue
            if(run == -1):
                s2 = 2 * S + 1
            else:
                n_balls, n_runs = balls - 1, runs - run
                if(n_runs <= 0):
                    s2 = 2 * S
                    reward = 1
                else:
                    if(n_balls == 0):
                        s2 = 2 * S + 1
                    else:
                        strike_change = int(((run % 2 == 0 and n_balls % 6 == 0)
                                             or (run % 2 != 0 and n_balls % 6 != 0)))

                        s2 = id[(n_balls, n_runs, 1 - strike_change)]
            if(not ((s1,ac,s2) in probs)):
                probs[(s1,ac,s2)] = 0
            probs[(s1,ac,s2)] += prob
    for s1,ac,s2 in probs:
        reward = int(s2 == 2 * S)
        Transitions.append((s1,ac,s2,reward,probs[(s1,ac,s2)]))

    print("numStates", 2 * S + 2)
    print("numActions", 5)
    print("end", 2 * S, 2 * S + 1)
    for s1, ac, s2, r, p in Transitions:
        print("transition", s1, ac, s2, r, p)
    print("mdptype episodic\ndiscount 1\n")
