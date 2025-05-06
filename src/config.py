from transformers import TrainingArguments

def get_training_args():
    return TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=20,
        save_steps=100,
        fp16=True,
        report_to="none"
    )
