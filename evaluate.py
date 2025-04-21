import transformers
import datasets
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

print(f"--- Используется Transformers версия: {transformers.__version__} ---")

# --- КОНФИГУРАЦИЯ ---
# Укажите путь к папке, ГДЕ СОХРАНЕНА лучшая модель и токенизатор после клонирования репо и запуска main.py
MODEL_DIR = './results_sst2/best_model_final' 
EVAL_BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_DIR):
    print(f"Ошибка: Папка с моделью не найдена по пути: {MODEL_DIR}")
    print("Пожалуйста, убедитесь, что скрипт main.py был запущен и модель сохранена.")
    exit()

print(f"Загрузка модели и токенизатора из: {MODEL_DIR}")
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval() 
except Exception as e:
    print(f"Ошибка при загрузке модели или токенизатора: {e}")
    exit()
print("Модель и токенизатор загружены.")


print("Загрузка и подготовка валидационного датасета SST-2...")
dataset = datasets.load_dataset("glue", "sst2")
validation_dataset_raw = dataset["validation"] 

def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_validation_dataset = dataset["validation"].map(tokenize_function, batched=True)
tokenized_validation_dataset = tokenized_validation_dataset.remove_columns(["sentence", "idx"])
tokenized_validation_dataset = tokenized_validation_dataset.rename_column("label", "labels")
tokenized_validation_dataset.set_format("torch")
eval_dataset = tokenized_validation_dataset
print("Валидационный датасет подготовлен.")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    return {
        'accuracy': accuracy,
        'f1': f1,
    }

eval_args = TrainingArguments(
    output_dir='./results_sst2/eval_temp',
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    do_train=False,
    do_eval=True,
    report_to="none", 
)

eval_trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print("Получение предсказаний и оценка...")
predictions_output = eval_trainer.predict(eval_dataset)

# Рассчитаем метрики из предсказаний
y_true = predictions_output.label_ids
y_pred = np.argmax(predictions_output.predictions, axis=-1)

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='binary')

print("\n--- Результаты оценки на валидационном наборе ---")
print(f"Точность (Accuracy): {accuracy:.4f}")
print(f"F1-мера (Binary): {f1:.4f}")

# Расширенный отчет и матрица ошибок
target_names = ['negative (0)', 'positive (1)']
print("\n--- Классификационный отчет ---")
print(classification_report(y_true, y_pred, target_names=target_names))

print("\n--- Матрица ошибок ---")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Предсказано')
plt.ylabel('Истинно')
plt.title('Матрица ошибок (Валидация)')
confusion_matrix_path = 'confusion_matrix_sst2.png'
plt.savefig(confusion_matrix_path)
print(f"Матрица ошибок сохранена в файл: {confusion_matrix_path}")

# --- Анализ ошибок (Примеры) ---
print("\n--- Примеры неверно классифицированных предложений (до 10) ---")
misclassified_indices = np.where(y_pred != y_true)[0]
count = 0
for i in misclassified_indices:
    if count < 10: 
        original_example = validation_dataset_raw[int(i)] 
        print(f"\nПредложение: {original_example['sentence']}")
        print(f"Истинный класс: {target_names[y_true[i]]}")
        print(f"Предсказанный класс: {target_names[y_pred[i]]}")
        print("-" * 20)
        count += 1
    else:
        break
if count == 0:
    print("Неверно классифицированных примеров не найдено.")

print("\n--- Скрипт evaluate.py завершен ---")