from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType

def apply_lora(model):
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "c_proj", "c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, config)
    return model
