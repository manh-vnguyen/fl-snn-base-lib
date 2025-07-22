
import copy
import time
import threading
from .base import ExperimentRunnerBase


class ExperimentRunnerThread(ExperimentRunnerBase):
    def run_single_experiment_thread(self, exp, gpu_remained):
        """Run a single experiment in a thread."""
        gpu_id = None
        while gpu_id is None:
            with self.lock:
                gpu_id = self.get_available_gpu(gpu_remained)
            if gpu_id is None:
                time.sleep(2)
        self.run_func(exp, gpu_id)
        with self.lock:
            self.release_gpu(gpu_id, gpu_remained)

    def run_parallel_experiment(self, all_experiments):
        self.lock = threading.Lock()
        num_threads = min(sum([self.GPU_QUOTAS[n] for n in self.GPU_QUOTAS.keys()]), len(all_experiments))
        print(f"Number of experiments: {len(all_experiments)}, {self.GPU_QUOTAS=}, "
              f"Starting {num_threads} threads")
        # Shared GPU usage dictionary (threads share memory)
        gpu_usage = copy.deepcopy(self.GPU_QUOTAS)
        threads = []
        results = []
        # Create a semaphore to limit concurrent threads
        semaphore = threading.Semaphore(num_threads)
        def worker(exp):
            with semaphore:
                result = self.run_single_experiment_thread(exp, gpu_usage)
                results.append(result)
        # Start all threads
        for exp in all_experiments:
            thread = threading.Thread(target=worker, args=(exp,))
            threads.append(thread)
            thread.start()
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        return results

__all__ = ['ExperimentRunnerThread']