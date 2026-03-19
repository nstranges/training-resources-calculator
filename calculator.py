import yaml

class MemoryCalculator():
    def __int__(self):
        self.config = self.load_config('config.yaml')

        # --- Model Parameters ---
        model = self.config.get('model', {})
        self.model_params_b = model.get('model_params_b')
        self.hidden_size = model.get('hidden_size')
        self.layers = model.get('layers')
        self.kv_heads = model.get('kv_heads')
        self.head_dim = model.get('head_dim')
        self.bytes_per_param = model.get('bytes_per_parameter')

        # --- Training Parameters ---
        train = self.config.get('training', {})
        self.train_gpus = train.get('train_gpus')
        self.zero_stage = train.get('zero_stage')
        self.train_batch_size = train.get('train_batch_size')
        self.train_overhead = train.get('estimated_overhead_percent')

        # --- Inference Parameters ---
        infer = self.config.get('inference', {})
        self.infer_gpus = infer.get('infer_gpus')
        self.infer_batch_size = infer.get('infer_batch_size')
        self.infer_overhead = infer.get('estimated_overhead_percent')

    def load_config(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)