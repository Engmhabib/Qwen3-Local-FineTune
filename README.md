# Qwen3 Local Fine-Tuning (Backend-First)

This project demonstrates how to fine-tune the open-source **Qwen3 (14B)** large language model using your **own data**, fully locally — with complete backend control.

Unlike most cloud-based platforms, this setup gives you:
- Full access to training configuration
- Efficient parameter fine-tuning with LoRA
- Reproducibility and transparency
- No third-party studios or hidden abstractions

## 🔧 Tech Stack

- **Model:** Qwen3 14B from Hugging Face
- **Tokenization:** Hugging Face Transformers
- **Training:** PyTorch + HF `Trainer`
- **Adapters:** LoRA via PEFT
- **Optimization:** 4-bit quantization (`bitsandbytes`)
- **Execution:** `accelerate` for hardware scaling

## ⚙️ Hardware Requirements

For Qwen3-14B:
- Minimum: `1x A100 (40GB)` or `2x RTX 3090 (24GB each)`
- For smaller setups: use `Qwen1.5-7B`

## 🧠 How to Fine-Tune

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

Place your dataset in:
```
data/reasoning_dataset.json
```

Format:
```json
[
  {"prompt": "Your question here", "response": "Your answer here"},
  ...
]
```

### 3. Configure Accelerate
```bash
accelerate config
```

### 4. Train the model
```bash
python src/train.py
```

## 🧪 Where to Tweak for Response Style

Inside `src/prepare_data.py`, we format how the **prompt + response** pairs are constructed. You can add instructions, system prompts, or wrap your inputs like so:

```python
text = f"### Instruction:\n{example['prompt']}\n\n### Think step by step before answering.\n\n### Response:\n{example['response']}"
```

## 📦 Project Structure
```
qwen3-local-finetune/
├── data/
│   └── reasoning_dataset.json
├── src/
│   ├── load_model.py
│   ├── apply_lora.py
│   ├── prepare_data.py
│   ├── train.py
│   └── config.py
├── README.md
├── requirements.txt
├── accelerate_config.yaml
```

## License
Open-source for educational and research use.
