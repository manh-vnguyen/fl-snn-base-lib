import json

base_attr_default = {
    'run_path': 'NON_DEFAULT_ATTRIBUTE',
    'seed': 22032025,
    'total_epochs': 'NON_DEFAULT_ATTRIBUTE',
    'checkpoint_freq': 10,
    'checkpoint_retain_last_model': False,
    'collect_std_stats': False,
    'perm_checkpoints': [],
    'test_epochs': [],
    'test_freq': None,
    'verbose': True,
    'batch_size': 32,
    'model': 'NON_DEFAULT_ATTRIBUTE',
    'dataset': 'NON_DEFAULT_ATTRIBUTE',
    'optimizer': 'NON_DEFAULT_ATTRIBUTE',
    'exp_id': 'NON_DEFAULT_ATTRIBUTE',
    'tags': None,
}

fl_attr_default = {
    'fl_momentum': 'NON_DEFAULT_ATTRIBUTE',
    'num_clients': 'NON_DEFAULT_ATTRIBUTE',
    'num_byz': 0,
    'attack': {'type': None, 'params': {}},
    'aggregator': {'type': 'Mean', 'params': {}},
}

class ExpSolo():
    def __init__(self, 
                 **kwargs
                 ):
        if 'exp_path' in kwargs.keys():
            self.from_json_file(kwargs['exp_path'])
            return
        elif 'exp_dict' in kwargs.keys():
            exp_dict = kwargs['exp_dict']
        else:
            exp_dict = kwargs
        self.from_dict(exp_dict)

    def best_acc(self):
        pass

    def from_dict(self, exp_dict):
        for key in base_attr_default.keys():
            if key in exp_dict.keys():
                setattr(self, key, exp_dict[key])
            elif base_attr_default[key] !=  'NON_DEFAULT_ATTRIBUTE':
                setattr(self, key, base_attr_default[key])
            else:
                raise Exception(f"Attribute problems: {key} is non-default")
        
        for key in exp_dict.keys():
            setattr(self, key, exp_dict[key])

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

class ExpFed(ExpSolo):
    def from_dict(self, exp_dict):
        super().from_dict(exp_dict)
        for key in fl_attr_default.keys():
            if key in exp_dict.keys():
                setattr(self, key, exp_dict[key])
            elif fl_attr_default[key] != 'NON_DEFAULT_ATTRIBUTE':
                setattr(self, key, fl_attr_default[key])
            else:
                raise f"Attribute problems: {key} is non-default"
            
    def best_acc(self):
        if len(self.test_accs) == 0:
            return None
        return max([item[1] for item in self.test_accs])
    