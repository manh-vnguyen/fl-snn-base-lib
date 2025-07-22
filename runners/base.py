import torch

from ..trainers import FLTrainer, SoloTrainer
from ..experiments import ExpFed, ExpSolo


class ExperimentRunnerBase:
    def __init__(self, gpu_quotas=None, run_func='run_experiment'):
        self.GPU_QUOTAS = gpu_quotas or {6: 3, 7: 3}
        self.run_func = getattr(self, run_func, None)

    def run_experiment_from_path(self, exp_path, gpu_id):
        exp = ExpFed(exp_path=exp_path)
        return self.run_experiment(exp, gpu_id)

    def run_experiment(self, exp, gpu_id):
        if type(exp) is ExpFed:
            trainer = FLTrainer(exp=exp, device=f'cuda:{gpu_id}')
        elif type(exp) is ExpSolo:
            trainer = SoloTrainer(exp=exp, device=f'cuda:{gpu_id}')
        trainer.run()

    def get_available_gpu(self, gpu_remained):
        available_gpus = [gpu_id for gpu_id in gpu_remained.keys() 
                            if gpu_remained[gpu_id] > 0]
        if not available_gpus:
            return None
        selected_gpu = max(available_gpus, key=lambda gpu_id: gpu_remained[gpu_id])
        gpu_remained[selected_gpu] -= 1
        return selected_gpu

    def release_gpu(self, gpu_id, gpu_remained):
        assert gpu_remained[gpu_id] < self.GPU_QUOTAS[gpu_id]
        gpu_remained[gpu_id] += 1
        torch.cuda.empty_cache()

    def run_parallel_experiment(self, all_experiments):
        pass
