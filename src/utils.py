import torch

PAD_IDX = 0
UNK_IDX = 1


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def tokens_to_words(token_ids, vocabulary, start_idx, end_idx):
    """Преобразует последовательность индексов в список слов."""
    words = []
    for idx in token_ids:
        if idx == PAD_IDX or idx == UNK_IDX or idx == start_idx:
            continue
        if idx == end_idx:
            break
        if 0 <= idx < len(vocabulary):
            words.append(vocabulary[idx])
    return words


def generate_translation(
    model,
    eng_tokenizer,
    rus_tokenizer,
    input_sentence,
    max_len=20,
    device="cuda",
):
    """Генерирует перевод для одного предложения."""
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    source_tensor = eng_tokenizer([input_sentence]).to(device)

    start_idx = rus_tokenizer.word_to_index["[start]"]
    end_idx = rus_tokenizer.word_to_index["[end]"]
    rus_vocab = rus_tokenizer.get_vocabulary()

    pred_token_ids = model.translate(
        source_tensor=source_tensor,
        start_token_idx=start_idx,
        end_token_idx=end_idx,
        max_len=max_len,
    )

    translated_words = tokens_to_words(
        pred_token_ids,
        rus_vocab,
        start_idx,
        end_idx,
    )

    return " ".join(translated_words)
