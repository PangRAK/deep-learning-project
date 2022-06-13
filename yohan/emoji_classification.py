import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer,DataCollatorWithPadding,TrainingArguments,Trainer

def preprocess(dataset):
    t = dataset['text']
    t = '@user' if t.startswith('@') and len(t) > 1 else t
    t = 'http' if t.startswith('http') else t
    dataset['text'] = t
    return dataset


dataset = load_dataset("tweet_eval", "emoji",script_version="master")
dataset = dataset.map(preprocess)
file = "roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(file, num_labels=20)
tokenizer = AutoTokenizer.from_pretrained(file)
model.resize_token_embeddings(len(tokenizer)) 
# pip install sentencepiece
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_emoji = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./result_emoji",
    learning_rate=3e-5,
    evaluation_strategy = "epoch",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
)

metric = load_metric("f1")
label_list = ['❤', '😍', '😂', '💕', '🔥', '😊', '😎', '✨', '💙', '😘', '📷', '🇺🇸', '☀', '💜', '😉', '💯', '😁', '🎄', '📸', '😜']


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    results = metric.compute(predictions=predictions, references=labels, average="macro")
    return results

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_emoji["train"],
    eval_dataset=tokenized_emoji["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()