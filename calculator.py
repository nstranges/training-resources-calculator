import yaml
import matplotlib.pyplot as plt

class MemoryCalculator():
    def __init__(self):
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
        self.show_training = train.get('show_training')
        self.train_gpus = train.get('train_gpus')
        self.zero_stage = train.get('zero_stage')
        self.train_batch_size = train.get('train_batch_size')
        self.train_overhead = train.get('train_overhead_percent')

        # --- Inference Parameters ---
        infer = self.config.get('inference', {})
        self.show_infer = infer.get('show_inference')
        self.infer_gpus = infer.get('infer_gpus')
        self.infer_batch_size = infer.get('infer_batch_size')
        self.infer_overhead = infer.get('infer_overhead_percent')

        # --- LoRA Parameters ---
        lora = self.config.get('lora', {})
        self.use_lora = lora.get('use_lora')
        self.lora_rank = lora.get('rank')

        # --- Context Sizes ---
        self.context_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]

    def load_config(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
        
    # Full weights for training
    def optimizer_plus_model(self):
        # Weights of the model for forward
        model_weights = self.model_params_b * self.bytes_per_param

        # The activations of the model for backprop. Same space for gradients as they replace the activations
        model_activations_or_gradients = self.model_params_b * self.bytes_per_param

        # Extra info stored for Adam. These params are usually stored in fp32 (4 bytes per param). 2 moments are used. 8 bytes per param
        adam_params = self.model_params_b * 8

        # Only zero stage 3 spreads the weights across every gpu
        if self.zero_stage == 3:
            total_gb = (model_weights + model_activations_or_gradients + adam_params) / self.train_gpus
        else:
            total_gb = model_weights + ((model_activations_or_gradients + adam_params) / self.train_gpus)

        return total_gb
    
    # LoRA weights for training
    def optimizer_plus_model_lora(self):
        # Weight reduction factor. Based on replacing the matrices. Assuming all matricies
        reduction_factor = self.hidden_size / (2 * self.lora_rank)

        total_gb = self.optimizer_plus_model() / reduction_factor

        return total_gb
    
    # Context dependedent VRAM for training. Assuming the use of flash attention
    def train_context_dependent(self, context_len):
        total_gb = (2 * self.train_batch_size * context_len * self.hidden_size * self.layers * self.bytes_per_param) / (1024**3)

        return total_gb
    
    # Context dependedent VRAM for inference
    def infer_context_dependent(self, context_len):
        total_gb = (2 * self.infer_batch_size * context_len * self.hidden_size * self.layers * self.bytes_per_param) / (1024**3)

        return total_gb
    
    # For the model weights alone
    def infer_model_weights(self):
        total_gb = (self.model_params_b * self.bytes_per_param) / self.infer_gpus

        return total_gb
    
    # Total training. Returns list
    def total_training_over_context(self):
        # Using lora or not
        if self.use_lora:
            base = self.optimizer_plus_model_lora()
        else:
            base = self.optimizer_plus_model()

        total_gbs = []

        for context in self.context_sizes:
            total_gb = (base + self.train_context_dependent(context_len=context)) * (1 + (self.train_overhead / 100))
                        
            total_gbs.append(total_gb)

        return total_gbs
    
    # Total infer. Returns list
    def total_infer_over_context(self):
        base = self.infer_model_weights()

        total_gbs = []

        for context in self.context_sizes:
            total_gb = (base + self.infer_context_dependent(context_len=context)) * (1 + (self.infer_overhead / 100))
                        
            total_gbs.append(total_gb)

        return total_gbs
    
    # Plots all of the information
    def plot_vram(self):
        if self.show_infer:
            infer_gbs = self.total_infer_over_context()
            plt.plot(self.context_sizes, infer_gbs, label='Inference')

        if self.show_training:
            train_gbs = self.total_training_over_context()
            plt.plot(self.context_sizes, train_gbs, label='Training')

        if self.show_training or self.show_infer:
            plt.xlabel('Context Length')
            plt.ylabel('VRAM Usage (GB)')
            plt.title('VRAM Vs Context Length')
            plt.legend()
            plt.ylim(bottom=0)
            plt.grid(True)
            plt.show()
        else:
            print("Both training and inference are not selected")

if __name__ == "__main__":
    calc = MemoryCalculator()

    calc.plot_vram()