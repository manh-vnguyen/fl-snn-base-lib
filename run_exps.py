from __lib import FLTrainer
import multiprocessing as mp
import argparse
import copy
import pickle
import os
import torch

GPU_QUOTAS = {0 : 3,
              1 : 3,
              2 : 3,
              3 : 3,
              4 : 3,
              5 : 3,
              6 : 3,
              7 : 3}

class ExpRecord:
    pass

def get_available_gpu(gpu_remained):
    available_gpus = [gpu_id for gpu_id in gpu_remained.keys() 
                        if gpu_remained[gpu_id] > 0]
    if not available_gpus:
        return None
    
    selected_gpu = max(available_gpus, key=lambda gpu_id: gpu_remained[gpu_id])
    gpu_remained[selected_gpu] -= 1

    return selected_gpu

def release_gpu(gpu_id, gpu_remained):
    assert gpu_remained[gpu_id] < GPU_QUOTAS[gpu_id]
    gpu_remained[gpu_id] += 1

def run_experiment(exp_path, gpu_id):
    exp = pickle.load(open(exp_path, 'rb'))
    
    if getattr(exp, 'checkpointed_epoch', None) != None and exp.checkpointed_epoch >= exp.total_epochs:
        return
    try:
        FLTrainer(exp=exp, device=f'cuda:{gpu_id}').run()
    except Exception as e:
        print(exp.exp_num, e)

    torch.cuda.empty_cache()

def pair_experiment_with_gpu(config):
    exp_path, gpu_remained = config
    with lock:
        gpu_id = get_available_gpu(gpu_remained)

    result = run_experiment(exp_path, gpu_id)
    with lock:
        release_gpu(gpu_id, gpu_remained)

    return result

def init_pool_processes(the_lock):
    global lock
    lock = the_lock

def run_parallel_processes(all_experiments):
    num_processes = sum([GPU_QUOTAS[n] for n in GPU_QUOTAS.keys()])
    print(f"Number of experiments: {len(all_experiments)}, {GPU_QUOTAS=}, "
          f"Starting {num_processes} processes")
    
    with mp.Manager() as manager:
        lock = mp.Lock()
        gpu_usage = manager.dict(copy.deepcopy(GPU_QUOTAS))
        exp_gpu_usages = [(exp, gpu_usage) for exp in all_experiments]
        with mp.Pool(
            processes=num_processes,
            initializer=init_pool_processes, initargs=(lock,)
        ) as pool:
            pool.map(pair_experiment_with_gpu, exp_gpu_usages)

def main(run_id):
    exp_paths = []
    i = 0
    while True:
        path = f'/home/combined_everything_FL/run_data/{run_id}/exp_{i}'
        if not os.path.exists(path):
            break
        exp_paths.append(path)
        i += 1

    run_parallel_processes(exp_paths)
    # run_experiment(exp_paths[6], 7)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FL experiments')
    parser.add_argument('--run_id', type=str, required=True, help='Experiment IDs')
    parsed_args = parser.parse_args()
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main(parsed_args.run_id)