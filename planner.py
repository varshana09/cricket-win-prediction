import time
from pulp import *
import numpy as np
import random
import argparse
import sys
import os
parser = argparse.ArgumentParser()


def read_mdp_file(file_name):
    #absolute_path = os.path.dirname(os.path.abspath())
    #file_name = "./"+file_name
    with open(file_name, "r") as f:
        lines = f.readlines()
    T, R = dict(), dict()
    for line in lines:
        X = line.strip().split()
        if(len(X) == 0):
            break
        if(X[0] == "numStates"):
            S = int(X[-1])
        elif (X[0] == "numActions"):
            A = int(X[-1])
        elif (X[0] == "end"):
            ends = [int(x) for x in X[1:]]
        elif (X[0] == "transition"):
            T[(int(X[1]), int(X[2]), int(X[3]))] = float(X[-1])
            R[(int(X[1]), int(X[2]), int(X[3]))] = float(X[-2])
        elif (X[0] == "mdptype"):
            mdptype = X[-1]
        else:
            assert(X[0] == "discount")
            gamma = float(X[-1])
    return (mdptype, S, A, T, R, gamma)


def read_pol_file(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    pi = [int(line.strip()) for line in lines]
    return pi


def eval_policy(pi, params, print_res=1):
    mdptype, S, A, T, R, gamma = params
    A = np.zeros((S, S))
    y = np.zeros(S)
    for s in range(S):
        y_sum = 0
        for s_ in range(S):
            if((s, pi[s], s_) in T):
                A[s][s_] = -T[(s, pi[s], s_)] * gamma
                y_sum += T[(s, pi[s], s_)] * R[(s, pi[s], s_)]
        y[s] = y_sum
        A[s][s] += 1
    V = np.linalg.solve(A, y)
    if(print_res):
        for s in range(S):
            print(round(V[s], 6), int(pi[s]))
    return V


def value_iteration(params):
    mdptype, S, A, T, R, gamma = params
    V_prev, V_curr = np.zeros(S), np.zeros(S)
    pi = np.zeros(S)
    while(True):
        pi_ = np.zeros(S)
        for s in range(S):
            for a in range(A):
                sum = 0
                for s_ in range(S):
                    if((s, a, s_) in T):
                        sum += T[(s, a, s_)]*(R[(s, a, s_)]+gamma*V_prev[s_])
                if(V_curr[s] < sum):
                    pi_[s] = a
                    V_curr[s] = sum
        max_diff = np.max(np.absolute(V_curr-V_prev))
        if(max_diff < 1e-12):
            pi = pi_
            break
        else:
            V_prev = V_curr
            V_curr = np.zeros(S)

    for s in range(S):
        print(round(V_curr[s], 6), int(pi[s]))


def howards_policy_iteration(params):
    mdptype, S, A, T, R, gamma = params
    pi = np.zeros(S)
    for s in range(S):
        for a in range(A):
            flag = 0
            for s_ in range(S):
                if((s, a, s_) in T):
                    pi[s] = a
                    flag = 1
                    break
            if(flag == 1):
                break
    while(True):
        V = eval_policy(pi, params, 0)
        Q = np.zeros((S, A))
        flag = 0
        for s in range(S):
            for a in range(A):
                if(pi[s] == a):
                    continue
                for s_ in range(S):
                    if((s, a, s_) in T):
                        Q[s][a] += T[(s, a, s_)] * \
                            (R[(s, a, s_)] + gamma * V[s_])
                if(Q[s][a] > V[s]):
                    flag = 1    
                    pi[s] = a
                    V[s] = Q[s][a]
        if(flag == 0):
            break
    for s in range(S):
        print(round(V[s], 6), int(pi[s]))


def linear_programming(params):
    mdptype, S, A, T, R, gamma = params
    states = range(S)
    V = LpVariable.dicts("V", (states))
    prob = LpProblem("MDPs",LpMinimize)
    prob += lpSum([V[i] for i in range(S)])
    for s in range(S):
        for a in range(A):        
            prob += V[s] - gamma * lpSum([T[(s,a,s_)] * V[s_] if (s,a,s_) in T else 0 for s_ in range(S)]) >= lpSum([T[(s,a,s_)]*R[(s,a,s_)] if (s,a,s_) in T else 0 for s_ in range(S)])
    prob.solve(PULP_CBC_CMD(msg=False))
    v = np.zeros(S)
    for i in range(S):
        v[i] = V[i].varValue
    pi = np.zeros(S)
    for s in range(S):
        val = -1e9
        for a in range(A):
            sum = 0
            for s_ in range(S):
                if((s,a,s_) in T):
                    sum += T[(s,a,s_)] * (R[(s,a,s_)] + gamma * v[s_])
            if(val < sum):
                val = sum
                pi[s] = a
    for s in range(S):
        print(round(v[s],6),int(pi[s]))
    
if __name__ == "__main__":
    parser.add_argument("--mdp", type=str, default="-")
    parser.add_argument("--algorithm", type=str, default="lp")
    parser.add_argument("--policy", type=str, default="-")
    args = parser.parse_args()
    mdp_file = args.mdp
    algo = args.algorithm
    pol = args.policy
    params = read_mdp_file(mdp_file)
    if(pol != "-"):
        pi = read_pol_file(pol)
        eval_policy(pi, params)
        sys.exit(0)
    if(algo == "vi"):
        value_iteration(params)
    elif(algo == "hpi"):
        howards_policy_iteration(params)
    else:
        assert(algo == "lp")
        linear_programming(params)
