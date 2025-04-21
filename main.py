import transformers
import datasets
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import os

print(f"--- Используется Transformers версия: {transformers.__version__} ---")

# 1. Данные
print("Загрузка датасета SST-2...")
dataset = datasets.load_dataset("glue", "sst2")
MODEL_NAME = "bert-base-uncased"
print(f"Загрузка токенизатора {MODEL_NAME}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

print("Токенизация данных...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
print("Обработка датасета...")
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
print("Данные подготовлены.")

# 2. Модель
print(f"Загрузка модели {MODEL_NAME}...")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Модель перемещена на устройство: {device}")

# 3. Метрики
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'f1': f1,
    }
print("Функция compute_metrics определена.")

# 4. Аргументы обучения
print("Настройка аргументов обучения...")
training_args = TrainingArguments(
    output_dir='./results_sst2',          
    num_train_epochs=3,                   
    per_device_train_batch_size=16,       
    per_device_eval_batch_size=64,        
    warmup_steps=500,                     
    weight_decay=0.01,                   
    logging_dir='./logs_sst2',            
    logging_strategy="steps",             
    logging_steps=100,                    
    eval_strategy="epoch",         
    save_strategy="epoch",               
    load_best_model_at_end=True,          
    metric_for_best_model="f1",           
    report_to="tensorboard",           
    save_total_limit=2,                   
    # fp16=torch.cuda.is_available(),     
)
print("Аргументы обучения настроены.")

# 5. Trainer
print("Создание объекта Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
print("Объект Trainer создан.")

# 6. Обучение
print("Запуск обучения...")
if device == torch.device("cuda"):
    torch.cuda.empty_cache() 
train_result = trainer.train()
print("Обучение завершено.")

# Сохранение метрик обучения
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
print("Метрики обучения сохранены.")

# 7. Сохранение модели
best_model_path = os.path.join(training_args.output_dir, 'best_model_final') 
print(f"Сохранение финальной лучшей модели и токенизатора в {best_model_path}...")
trainer.save_model(best_model_path)
tokenizer.save_pretrained(best_model_path)
print("Финальная модель и токенизатор сохранены.")

# 8. Оценка (на валидационном наборе)
print("Запуск оценки на валидационном наборе...")
eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
print("Метрики на валидационном наборе:")
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", os.path.join(training_args.output_dir, "eval_results.json"))
print("Метрики оценки сохранены.")

print("\n--- Скрипт main.py завершен ---")