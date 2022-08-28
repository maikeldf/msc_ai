#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import time
import argparse

from src.Agent import Agent

def remove_path(path):
    try:
        def removeReadOnly(func, path, excinfo):
            os.chmod(path, stat.S_IWRITE)
            func(path)

        shutil.rmtree(path, onerror=removeReadOnly)
        time.sleep(3)
        return True
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='logs', help='Root log dir for tensorboard purposes')
    parser.add_argument('--model', type=str,  default='model', help='Root model dir for tensorboard purposes') 
    parser.add_argument('--input', type=str, default='../data/input.csv', help='CSV file with input data')
    parser.add_argument('--steps', type=int, default=20000, help='Total of steps to train')    
    
    parser.add_argument('--train', action='store_true', help='Training mode')
    parser.add_argument('--eval', action='store_true',  help='Evaluation mode')
    parser.add_argument('--lin', action='store_true', help='Learning rate linear decay')
    parser.add_argument('--newmodel', action='store_true', help='Create a new model rather than evolve the current one')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} does not exist")
        return

    if not args.train and not args.eval:
        print("Error: No mode selected.\nPlease select one of the following modes --train, --eval")
        return 

    count =  lambda root, folder: len([d for d in os.listdir(root) if d.startswith(folder)])

    if args.train:
        if args.newmodel and os.path.isdir(args.model):
            folders = count('.',args.model)
            if folders:
                args.model += f"_{folders}"
        else:
            folders = count('.',args.model) - 1
            if folders and folders > 0:
                args.model += f"_{folders}"


    folders = count('.',args.log)
    if folders:
        args.log += f"_{folders}"

    args.log = os.path.join(args.log,'progress_tensorboard')

    log_eval = os.path.join(args.log,'PPO_EVAL')

    os.makedirs(args.log, exist_ok=True)
    os.makedirs(log_eval, exist_ok=True)
    os.makedirs(args.model, exist_ok=True)    

    agent = Agent(
        args.log,
        log_eval,
        args.model,
        args.input,
        args.steps)

    if args.train:
        agent.train(args.lin)

    if args.eval:
        agent.evaluate()

if __name__=="__main__":
    start = time.perf_counter()
    main()
    elapsed_time = time.perf_counter() - start
    print(f"Elapsed time: {elapsed_time/60:0.2f} minutes")
