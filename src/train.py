from transformers import Trainer, DataCollatorForLanguageModeling
from src.load_model import load_qwen
from src.apply_lora import apply_lora
from src.prepare_data import load_and_format
from src.config import get_training_args

model, tokenizer = load_qwen()
model = apply_lora(model)
dataset = load_and_format(tokenizer)
args = get_training_args()

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
