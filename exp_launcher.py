from omegaconf import OmegaConf

import importlib
import multiprocessing as mp
import argparse
import random
import time
import subprocess
import os
import csv

from tqdm import tqdm

from setproctitle import setproctitle


def find_available_gpus(idle_threshold: int = 5):
    # Find all gpus with GPU util and MemoryAccess util < threshold

    csv_output = subprocess.check_output(["nvidia-smi",
                                          "--query-gpu=index,utilization.gpu,utilization.memory",
                                          "--format=csv,noheader,nounits"]).decode()
    csv_output = [list(map(int, line)) for line in csv.reader(csv_output.splitlines())]
    return [index for (index, gpu_util, mem_util) in csv_output if (gpu_util < idle_threshold) and (mem_util < idle_threshold)]


def split(a, n):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length

    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


def get_class_from_str(import_name: str, class_name: str):
    if import_name:
        class_inst = getattr(importlib.import_module(import_name), class_name, None)
    else:
        class_inst = globals().get(class_name, None)

    return class_inst


def worker_(env, import_name, method, conf_list):
    # Set environment variables for worker
    os.environ.update(env)

    m = get_class_from_str(import_name, method)

    for conf in conf_list:
        # set proctitle
        name = conf.log_group
        setproctitle(name)

        # info
        tqdm.write("-- {} running --".format(name))
        tqdm.write(OmegaConf.to_yaml(conf))
        tqdm.write("-------------")

        # begin process
        m(conf)


def launch_experiments(launch):
    mp.set_start_method("spawn")

    # generate experiment conf list
    exp_conf_list = []
    seed = random.SystemRandom().randint(0, int(1e8))  # guarantee different seed for each experiment

    for expset_name, expset_conf in launch.experiment_sets.items():
        for task_conf in launch.tasks:
            # allocate seed for each experiment
            for _ in range(launch.launch.seed_per_exp):
                exp_conf_list.append(OmegaConf.merge(task_conf, expset_conf, {"seed": seed, "log_group": expset_name}))
                seed += 1

    # run in random order to balance loads
    random.shuffle(exp_conf_list)

    # single-threaded prepare for running environment
    if "prepare_method" in launch.launch:
        prepare_fn = get_class_from_str(launch.launch.prepare_filename, launch.launch.prepare_method)
        for conf in exp_conf_list:
            prepare_fn(conf)

    # allocate experiments
    all_devices = launch.devices if hasattr(launch, "devices") else find_available_gpus()
    if not len(all_devices):
        print ("No available devices !!!")
        return

    print(f"Available devices: {all_devices}")

    num_processes = launch.launch.runs_per_device * len(all_devices)

    proc_device_id = list(all_devices) * launch.launch.runs_per_device
    proc_conf_list = split(exp_conf_list, num_processes)

    # get env variables
    env = OmegaConf.to_container(launch.get("env", OmegaConf.create({})))
    env = {k: str(v) for k, v in env.items()}  # To string

    # create processes
    processes = []
    for proc_id, conf_list in enumerate(proc_conf_list):
        # Set device id for proc
        proc_env = env.copy()
        proc_env["CUDA_VISIBLE_DEVICES"] = str(proc_device_id[proc_id])

        # start process
        processes.append(mp.Process(target=worker_, kwargs={
            "env": proc_env,

            "import_name": launch.launch.filename,
            "method": launch.launch.method,

            "conf_list": conf_list
        }))

    # run processes
    for proc in processes:
        proc.start()

        # cold start the processes to avoid contention
        time.sleep(launch.launch.cold_start_seconds)

    [proc.join() for proc in processes]


def main():
    conf = OmegaConf.from_cli()
    if hasattr(conf, "include"):
        conf = OmegaConf.merge(
            OmegaConf.load(conf.include),
            conf
        )

    launch_experiments(conf)


if __name__ == "__main__":
    main()
