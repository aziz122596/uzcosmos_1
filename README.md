# Анализ тональности SST-2 с помощью дообучения BERT

Этот проект дообучает предобученную модель `bert-base-uncased` для задачи бинарной классификации тональности (позитивная/негативная) на датасете SST-2 (Stanford Sentiment Treebank).

## Структура проекта

- `main.py`: Основной скрипт для загрузки данных, дообучения модели и сохранения лучшей версии.
- `evaluate.py`: Скрипт для загрузки сохраненной модели, оценки ее производительности на валидационном наборе и генерации отчетов/графиков.
- `requirements.txt`: Список необходимых Python библиотек.
- `README.md`: Этот файл.
- `.gitignore`: Определяет файлы, которые не должны отслеживаться Git (например, папки с моделями).
- `results_sst2/`: Папка (создается `main.py` после запуска), содержащая чекпоинты, логи и сохраненную лучшую модель (`best_model_final/`).
- `logs_sst2/`: Папка (создается `main.py` после запуска) для логов TensorBoard.
- `confusion_matrix_sst2.png`: Матрица ошибок (создается `evaluate.py`).

*Необходимо иметь минимум 16 ГБ оперативной памяти и GPU с 16 ГБ видеопамяти для комфортного запуска проекта.*

## Установка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone [https://github.com/aziz122596/uzcosmos_1.git](https://github.com/aziz122596/uzcosmos_1.git)
    cd uzcosmos_1
    ```

2.  **Создайте и активируйте виртуальное окружение (рекомендуется):**
    ```bash
    python -m venv venv
    # Linux/macOS:
    source venv/bin/activate
    # Windows:
    # venv\Scripts\activate
    ```

3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```
    *Примечание: Установка `torch` может потребовать специфических команд в зависимости от вашей ОС и наличия GPU (CUDA). См. официальный сайт PyTorch.*

## Использование

1.  **Обучение модели:**
    Запустите скрипт `main.py`. Это начнет процесс загрузки данных, токенизации, дообучения и сохранения модели.
    ```bash
    python main.py
    ```
    - Логи и чекпоинты будут сохраняться в папках `logs_sst2/` и `results_sst2/`.
    - Лучшая модель и токенизатор будут сохранены в `results_sst2/best_model_final/`.

2.  **Оценка модели:**
    После завершения обучения запустите скрипт `evaluate.py` для оценки сохраненной модели.
    ```bash
    python evaluate.py
    ```
    - Скрипт выведет метрики (Accuracy, F1), классификационный отчет и примеры ошибок в консоль.
    - Матрица ошибок будет сохранена в файл `confusion_matrix_sst2.png`.

## Подход

- **Модель:** `bert-base-uncased` из библиотеки Hugging Face Transformers.
- **Датасет:** SST-2 из библиотеки Hugging Face Datasets.
- **Предобработка:** Токенизация с помощью `BertTokenizer`, паддинг до максимальной длины (512), усечение.
- **Обучение:** Использовался `Trainer` API. Модель обучалась 3 эпохи (настраиваемо) с размером батча 16 (настраиваемо). Лучшая модель выбиралась по F1-мере на валидационном наборе.
- **Оценка:** Метрики Accuracy и F1-score, классификационный отчет, матрица ошибок.

## Результаты

*(Заполните этот раздел после получения результатов оценки)*

- **Accuracy (валидация):** ...
- **F1-score (валидация):** ...

*(Сюда можно добавить изображение `confusion_matrix_sst2.png` после его генерации, используя синтаксис `![Описание](путь/к/картинке.png)`)*

## Возможные улучшения

- Тонкая настройка гиперпараметров (learning rate, batch size, scheduler).
- Использование других предобученных моделей (RoBERTa, DeBERTa).
- Применение более сложных техник обучения (кросс-валидация, ансамблирование).
- Более глубокий анализ ошибок классификации.