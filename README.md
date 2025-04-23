# 🚀 Fine-Tuned LLaMA-3 8B Model with Unsloth and Alpaca

Welcome to the **Fine-Tuned LLaMA-3 8B Model** project! This repository contains a high-performance implementation of fine-tuning the LLaMA-3 8B model using **Unsloth** for efficient training and the **Alpaca-cleaned** dataset for instruction-based learning. Optimized for GPU environments, this project leverages 4-bit quantization, LoRA adapters, and modern training techniques to deliver a robust and memory-efficient fine-tuning pipeline.

---

## 🎯 Project Overview

This project fine-tunes the `unsloth/llama-3-8b-bnb-4bit` model using **Unsloth**, a library designed to accelerate and optimize large language model training. The fine-tuning process employs **LoRA** (Low-Rank Adaptation) to reduce memory usage while maintaining performance. The model is trained on the **Alpaca-cleaned** dataset, which provides high-quality instruction-input-output triplets for instruction tuning.

Key features:
- **4-bit quantization** for reduced memory footprint.
- **LoRA adapters** for efficient fine-tuning.
- **Unsloth optimization** for up to 30% less VRAM usage and larger batch sizes.
- **Alpaca prompt formatting** for standardized instruction tuning.
- **GPU compatibility** for both modern (Ampere, Hopper) and older (Tesla T4, V100) architectures.

---

## 🛠️ Installation

To get started, ensure you have a compatible GPU environment (e.g., Google Colab with Tesla T4 or NVIDIA A100). Follow these steps to set up the project:

1. **Install PyTorch with CUDA support**:
   The code automatically detects the GPU's CUDA capability and installs the appropriate dependencies.

2. **Install Unsloth**:
   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```

3. **Install additional dependencies**:
   - For modern GPUs (CUDA capability >= 8.0, e.g., RTX 30xx, A100):
     ```bash
     pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
     ```
   - For older GPUs (e.g., Tesla T4, V100):
     ```bash
     pip install --no-deps xformers trl peft accelerate bitsandbytes
     ```

4. **Verify GPU setup**:
   The code includes a memory stats check to display the GPU name, total memory, and reserved memory. For example:
   ```
   GPU = Tesla T4. Max memory = 14.741 GB.
   7.209 GB of memory reserved.
   ```

---

## 📊 Model and Training Configuration

### Model Setup
- **Base Model**: `unsloth/llama-3-8b-bnb-4bit`
- **Max Sequence Length**: 2048 (with RoPE scaling support)
- **Quantization**: 4-bit quantization enabled to reduce memory usage
- **Data Type**: Auto-detected (Float16 for Tesla T4/V100, Bfloat16 for Ampere+)

### LoRA Configuration
- **Rank (`r`)**: 16
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **LoRA Alpha**: 16
- **LoRA Dropout**: 0 (optimized)
- **Bias**: None (optimized)
- **Gradient Checkpointing**: `unsloth` (reduces VRAM usage by 30%)
- **Random State**: 3407

### Training Arguments
- **Batch Size**: 2 (per device)
- **Gradient Accumulation Steps**: 4
- **Warmup Steps**: 5
- **Max Steps**: 60
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW (8-bit)
- **Weight Decay**: 0.01
- **LR Scheduler**: Linear
- **Precision**: FP16 (or BF16 if supported)
- **Output Directory**: `outputs`

---

## 📈 Training Results

The model was trained for **60 steps**, with the training loss recorded at each step. Below is a summary of the training progress:

| Step | Training Loss |
|------|---------------|
| 1    | 1.5828        |
| 10   | 1.0990        |
| 20   | 0.8843        |
| 30   | 0.8604        |
| 40   | 1.1649        |
| 50   | 1.0706        |
| 60   | 0.8937        |

The loss generally trends downward, indicating successful fine-tuning. The final loss of **0.8937** suggests the model has adapted well to the Alpaca dataset.

---

## 💾 Dataset

The **Alpaca-cleaned** dataset (`yahma/alpaca-cleaned`) is used for training. It contains high-quality instruction-input-output triplets, formatted using the following prompt template:

```plaintext
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

The dataset is preprocessed to include the end-of-sequence (EOS) token for proper generation termination.

---

## 🧑‍💻 Usage

To fine-tune the model, run the provided Python script in a GPU-enabled environment (e.g., Google Colab). The script handles:
1. Dependency installation based on GPU type.
2. Model and tokenizer loading with 4-bit quantization.
3. LoRA adapter setup for efficient fine-tuning.
4. Dataset loading and prompt formatting.
5. Training with the **SFTTrainer** from the `trl` library.

After training, the model is saved to the `outputs` directory. You can load the fine-tuned model for inference as follows:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)
FastLanguageModel.for_inference(model)

# Example inference
inputs = tokenizer("Your prompt here", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🔧 Optimizations

This project incorporates several optimizations for efficiency and performance:
- **Unsloth**: Reduces VRAM usage by 30% and supports larger batch sizes.
- **4-bit Quantization**: Minimizes memory requirements, enabling training on consumer GPUs.
- **LoRA**: Fine-tunes only a small subset of parameters, reducing computational overhead.
- **Gradient Checkpointing**: Trades compute for memory, allowing longer contexts.
- **8-bit AdamW Optimizer**: Optimizes memory usage during training.

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 🙌 Acknowledgments

- **Unsloth**: For providing an efficient fine-tuning framework.
- **Hugging Face**: For the `transformers`, `trl`, and `peft` libraries.
- **Yahma**: For the high-quality Alpaca-cleaned dataset.
- **PyTorch**: For the backbone of the training pipeline.

---

## 📬 Contact

For questions or contributions, feel free to open an issue or submit a pull request on this repository.

Happy fine-tuning! 🚀
