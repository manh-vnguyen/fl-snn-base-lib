import json
from .trainers import FLTrainer
import torch
import multiprocessing as mp
import copy

class ExpBase:
    def __init__(self, 
                 exp_path=None,
                 model=None,
                 optimizer=None,
                 exp_id=None,
                 checkpoint_freq=10,
                 run_path=None,
                 dataset=None,
                 total_epochs=500,
                 test_epochs=None,
                 ):
        if exp_path != None:
            self.from_json_file(exp_path)
        else:
            self.run_path = run_path
            self.seed = 22032025
            self.total_epochs = total_epochs
            self.checkpoint_freq = checkpoint_freq
            self.checkpoint_retain_last_model = False
            self.collect_std_stats = False
            self.perm_checkpoints = []
            self.test_epochs = test_epochs
            self.verbose = True
            
            self.batch_size = 32

            self.model=model
            self.dataset=dataset
            self.optimizer={
                'lr': 0.02,
                'momentum': 0.95,
                'weight_decay': 0.0005,
            } if optimizer is None else optimizer

            self.exp_id = exp_id

    def best_acc(self):
        pass

    def from_dict(self, exp_dict):
        for key, value in exp_dict.items():
            setattr(self, key, value)

    def from_json_file(self, exp_path):
        # Load the JSON data from file
        with open(exp_path, 'r') as f:
            exp_dict = json.load(f)
        
        self.from_dict(exp_dict)

    def save_json(self):
        # List of attributes to exclude from serialization
        exclude_attrs = ['model_obj']
        
        # Gather all available attributes except system defaults and excluded ones
        exp_dict = {}
        for attr_name in dir(self):
            # Skip private/magic methods and callable attributes
            if attr_name.startswith('_') or callable(getattr(self, attr_name)):
                continue
            # Skip excluded attributes
            if attr_name in exclude_attrs:
                continue
            
            attr_value = getattr(self, attr_name)
            
            # Check if the attribute is JSON serializable
            try:
                json.dumps(attr_value)
                exp_dict[attr_name] = attr_value
            except (TypeError, ValueError) as e:
                print(f"Warning: Attribute '{attr_name}' is not JSON serializable and will be excluded. Error: {e}")
                continue

        # Save the dictionary to the JSON file
        with open(self.exp_path, 'w') as f:
            json.dump(exp_dict, f, indent=2)

    def __repr__(self) -> str:
        return json.dumps({attr: getattr(self, attr) for attr in dir(self) 
                if not attr.startswith('__') and not callable(getattr(self, attr))}, indent=4)

class ExpSolo(ExpBase):
    pass

class ExpFed(ExpBase):
    def __init__(self, 
                 exp_path=None,
                 model=None,
                 optimizer = None,
                 exp_id=None,
                 checkpoint_freq=10,
                 dataset=None,
                 run_path=None,
                 total_epochs=500,
                 test_epochs=None,
                 fl_momentum=None,
                 num_clients=None,
                 num_byz=0,
                 attack=None,
                 aggregator=None,
                 ):
        if exp_path != None:
            self.from_json_file(exp_path)
        else:
            super().__init__(
                 exp_path=exp_path,
                 model=model,
                 optimizer=optimizer,
                 exp_id=exp_id,
                 checkpoint_freq=checkpoint_freq,
                 dataset=dataset,
                 run_path=run_path,
                 total_epochs=total_epochs,
                 test_epochs=test_epochs,
                 )
            
            self.fl_momentum=fl_momentum
            self.num_clients = num_clients
            self.num_byz = num_byz
            self.attack={'type': None, 'params': {}} if attack is None else attack
            self.aggregator={'type': 'Mean', 'params': {}} if aggregator is None else aggregator
    
    def best_acc(self):
        if len(self.test_accs) == 0:
            return None
        return max([item[1] for item in self.test_accs])
    

class ExperimentRunner:
    def __init__(self, gpu_quotas=None, run_func='run_experiment'):
        self.GPU_QUOTAS = gpu_quotas or {6: 3, 7: 3}
        self.run_func = getattr(self, run_func, None)

    def run_experiment_from_path(self, exp_path, gpu_id):
        exp = ExpFed(exp_path=exp_path)
        return self.run_experiment(exp, gpu_id)

    def run_experiment(self, exp, gpu_id):
        trainer = FLTrainer(exp=exp, device=f'cuda:{gpu_id}')
        trainer.run()
        return trainer.exp.best_acc()

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

    def run_parallel_processes(self, all_experiments):
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


__all__ = ['ExpSolo', 'ExpFed', 'ExperimentRunner']