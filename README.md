
# Seq2Seq Translator — Нейронный перевод с английского на русский

> **Модель Seq2Seq с механизмом внимания для перевода с английского на русский.
> Архитектура: двунаправленный GRU-энкодер, GRU-декодер с Luong-вниманием. Обучена на датасете ManyThings (536k пар).
> Достигает BLEU 0.28 на тестовой выборке.**

---

## Навигация
- [Особенности](#особенности)
- [Архитектура модели](#архитектура-модели)
- [Результаты](#результаты)
- [Установка](#установка)
- [Использование](#использование)
  - [Инференс](#инференс)
  - [Обучение](#обучение)
- [Структура проекта](#структура-проекта)
- [Веса модели](#веса-модели)

---

## Особенности

- **Seq2Seq с вниманием** - двунаправленный GRU-энкодер, GRU-декодер с механизмом Luong-внимания.
- **Кастомная токенизация** - векторизация текста на уровне слов с обработкой специальных токенов `[start]`, `[end]`.
- **Обучение с Teacher Forcing** - постепенное снижение коэффициента с 1.0 до 0.5.
- **Early Stopping** - остановка при отсутствии улучшения validation loss.
- **Оценка BLEU** - расчёт корпусного BLEU на тестовой выборке.
- **Поддержка GPU** - автоматическое использование CUDA при наличии.

---

## Архитектура модели

Модель представляет собой классический Seq2Seq с механизмом внимания:

| Компонент | Описание |
|-----------|----------|
| **Энкодер** | Двунаправленный GRU (embed_dim=256, hidden_dim=512) |
| **Декодер** | Односторонний GRU с вниманием |
| **Внимание** | Luong-style (базовое) |
| **Функция потерь** | CrossEntropyLoss (ignore_index=PAD_IDX) |
| **Оптимизатор** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **Планировщик** | ReduceLROnPlateau (patience=2, factor=0.5) |

**Количество параметров:** ~30.8 млн

---

## Результаты

Модель обучалась 6 эпох до early stopping. Лучшие метрики:

| Метрика | Значение |
|---------|----------|
| **Train Loss** | 1.6907 |
| **Validation Loss** | **1.6561** |
| **Test Loss** | 3.3033 |
| **Corpus BLEU** | **0.2843 (28.43%)** |

### Примеры перевода

| Исходный текст (EN) | Перевод (RU) |
|---------------------|--------------|
| Hello world! | здравствуйте мир |
| I love machine learning. | я люблю изучать английский |
| Where is the nearest bank? | где ближайшая банк |
| She has a dog. | у неё собака |
| How are you today? | как вы сегодня |

> **Примечание:** Качество перевода ограничено размером датасета и вычислительными ресурсами. Для улучшения результатов стоит увеличить выборку для обучения или изменить стиль токенизации

---

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Lol547/Seq2Seq-eng-rus-translator.git
cd Seq2Seq-eng-rus-translator
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

**Требования:**
- Python 3.8+
- PyTorch (с поддержкой CUDA, если есть GPU)
- torchvision, numpy, matplotlib, tqdm, nltk

---

## Использование

### Инференс

**Пример перевода одного предложения:**

```python
from src.translate import load_model, translate

# Загрузка модели
model, eng_tokenizer, rus_tokenizer = load_model("checkpoints/best_model.pt")

# Перевод
result = translate(
    model, eng_tokenizer, rus_tokenizer,
    "How are you today?",
    max_len=20,
    device="cuda"  # или "cpu"
)
print(result)  # как вы сегодня
```

**Командная строка:**

```bash
python src/translate.py --text "Hello world!" --weights checkpoints/best_model.pt
```

### Обучение

```bash
python src/train.py --epochs 20 --batch_size 64 --hidden_dim 512 --embed_dim 256
```

Все гиперпараметры можно настроить через аргументы командной строки или изменить в конфигурационном файле.

---

## Структура проекта

```
seq2seq-translator/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── seq2seq_translator.ipynb    # Исходный ноутбук с экспериментами
│
├── src/
│   ├── __init__.py
│   ├── model.py                     # Seq2SeqWithAttention
│   ├── tokenizer.py                 # TextVectorization
│   ├── dataset.py                   # TranslationDataset
│   ├── train.py                     # Скрипт обучения
│   ├── translate.py                 # Скрипт инференса
│   └── utils.py                     # Вспомогательные функции
│
├── configs/
│   └── default.yaml                 # Конфигурация гиперпараметров
│
├── checkpoints/                     # Папка для весов (в .gitignore)
│   ├── best_model.pt
│   ├── english_tokenizer.pkl
│   ├── russian_tokenizer.pkl
│   └── model_config.pkl
│
└── data/
    └── rus-eng.zip                  # Датасет (скачивается автоматически)
```

---

## Веса модели

Для работы необходимы следующие файлы:
1. **`best_model.pt`** — веса модели.
2. **`english_tokenizer.pkl`** — токенизатор для английского языка.
3. **`russian_tokenizer.pkl`** — токенизатор для русского языка.
4. **`model_config.pkl`** — конфигурация модели.

**Ссылки для скачивания:**
[Все веса, конфиг и токенизаторы](https://drive.google.com/drive/folders/1mUNV3TJInDP81TP8cAu-KKGuwbWlzymx?usp=sharing)

---

## Контакты

По вопросам или найденным ошибкам пишите: sokolovkirill489@gmail.com TG: @qqkiru
