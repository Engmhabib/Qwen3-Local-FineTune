from datasets import load_dataset

def load_and_format(tokenizer):
    dataset = load_dataset("json", data_files="data/reasoning_dataset.json")["train"]

    def tokenize(example):
        text = (
            f"### Instruction:\n{example['prompt']}\n\n"
            f"### Think step by step before answering.\n\n"
            f"### Response:\n{example['response']}"
        )
        return {"input_ids": tokenizer(text, truncation=True, return_tensors="pt").input_ids[0]}

    dataset = dataset.map(tokenize)
    return dataset
