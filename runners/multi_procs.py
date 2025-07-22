from .base import ExperimentRunnerBase
import multiprocessing as mp
import copy

class ExperimentRunnerProcess(ExperimentRunnerBase):
    def pair_experiment_with_gpu(self, config):
        exp, gpu_remained = config
        with lock:
            gpu_id = self.get_available_gpu(gpu_remained)
        result = self.run_func(exp, gpu_id)
        with lock:
            self.release_gpu(gpu_id, gpu_remained)

        return result

    def init_pool_processes(self, the_lock):
        global lock
        lock = the_lock

    def run_parallel_experiment(self, all_experiments):
        mp.set_start_method('spawn', force=True)
        num_processes = min(sum([self.GPU_QUOTAS[n] for n in self.GPU_QUOTAS.keys()]), len(all_experiments))
        print(f"Number of experiments: {len(all_experiments)}, {self.GPU_QUOTAS=}, "
              f"Starting {num_processes} processes")
        
        with mp.Manager() as manager:
            lock = mp.Lock()
            gpu_usage = manager.dict(copy.deepcopy(self.GPU_QUOTAS))
            exp_gpu_usages = [(exp, gpu_usage) for exp in all_experiments]
            with mp.Pool(
                processes=num_processes,
                initializer=self.init_pool_processes, initargs=(lock,)
            ) as pool:
                pool.map(self.pair_experiment_with_gpu, exp_gpu_usages)

__all__ = ['ExperimentRunnerProcess']