import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse

from model import Seq2SeqWithAttention
from tokenizer import TextVectorization, PAD_IDX
from dataset import make_dataset
from utils import EarlyStopping

import pathlib
import urllib.request
import zipfile
import re
import string


def download_and_prepare_data():
    """Скачивает и распаковывает датасет rus-eng."""
    zip_path = pathlib.Path("rus-eng.zip")
    extract_dir = pathlib.Path("rus-eng-data")
    text_path = extract_dir / "rus.txt"

    if not text_path.exists():
        if not zip_path.exists():
            print("Скачивание архива...")
            url = "https://www.manythings.org/anki/rus-eng.zip"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=15) as response, open(zip_path, "wb") as out_file:
                chunk_size = 1024 * 1024
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    print(f"Загружено: {downloaded / (1024 * 1024):.1f} MB", end="\r")
            print("\nСкачивание завершено")

        print("Распаковка архива...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Распаковка завершена")

    return text_path


def load_data(text_path):
    """Загружает пары предложений из файла."""
    with open(text_path, encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]

    text_pairs = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 2:
            english, russian = parts[0], parts[1]
            russian = f"[start] {russian} [end]"
            text_pairs.append((english, russian))
    return text_pairs


def custom_standardization(input_string):
    strip_chars = string.punctuation.replace("[", "").replace("]", "")
    lowercase = input_string.lower()
    return re.sub(f"[{re.escape(strip_chars)}]", "", lowercase)


def main():
    parser = argparse.ArgumentParser(description="Обучение Seq2Seq переводчика")
    parser.add_argument("--epochs", type=int, default=20, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=64, help="Размер батча")
    parser.add_argument("--embed_dim", type=int, default=256, help="Размер эмбеддингов")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Размер скрытого слоя")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--sequence_length", type=int, default=20, help="Максимальная длина последовательности")
    parser.add_argument("--vocab_size", type=int, default=25000, help="Размер словаря")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    text_path = download_and_prepare_data()
    text_pairs = load_data(text_path)
    random.shuffle(text_pairs)

    train_len = int(0.85 * len(text_pairs))
    val_len = int(0.075 * len(text_pairs))
    train_pairs = text_pairs[:train_len]
    val_pairs = text_pairs[train_len:train_len + val_len]
    test_pairs = text_pairs[train_len + val_len:]

    print(f"Всего пар: {len(text_pairs)}")
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    eng_tokenizer = TextVectorization(
        max_tokens=args.vocab_size,
        output_sequence_length=args.sequence_length,
    )
    rus_tokenizer = TextVectorization(
        max_tokens=args.vocab_size,
        output_sequence_length=args.sequence_length + 1,
        standardize=custom_standardization,
        special_tokens=["[start]", "[end]"],
    )

    train_eng = [pair[0] for pair in train_pairs]
    train_rus = [pair[1] for pair in train_pairs]
    eng_tokenizer.adapt(train_eng)
    rus_tokenizer.adapt(train_rus)

    src_vocab_size = len(eng_tokenizer.get_vocabulary())
    tgt_vocab_size = len(rus_tokenizer.get_vocabulary())
    print(f"Размер словаря EN: {src_vocab_size}, RU: {tgt_vocab_size}")

    train_ds = make_dataset(train_pairs, eng_tokenizer, rus_tokenizer, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_pairs, eng_tokenizer, rus_tokenizer, args.batch_size, shuffle=False)

    model = Seq2SeqWithAttention(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Всего параметров: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early_stopping = EarlyStopping(patience=args.patience)

    history_train_loss = []
    history_val_loss = []

  
    for epoch in range(1, args.epochs + 1):
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        teacher_forcing_ratio = 1.0 - progress * 0.5

        model.train()
        train_loss = 0.0
        pbar = tqdm(train_ds, desc=f"Эпоха {epoch:02d}/{args.epochs:02d} [Train]")
        for batch_features, batch_labels in pbar:
            source = batch_features["english"].to(device)
            target = batch_features["russian"].to(device)
            labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(source, target, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(outputs.reshape(-1, tgt_vocab_size), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", tf=f"{teacher_forcing_ratio:.2f}")

        avg_train_loss = train_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in tqdm(val_ds, desc=f"Эпоха {epoch:02d}/{args.epochs:02d} [Val]"):
                source = batch_features["english"].to(device)
                target = batch_features["russian"].to(device)
                labels = batch_labels.to(device)
                outputs = model(source, target, teacher_forcing_ratio=0.0)
                loss = criterion(outputs.reshape(-1, tgt_vocab_size), labels.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_ds)
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)

        print(f"Итог эпохи {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | TF: {teacher_forcing_ratio:.2f}")

        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": avg_val_loss,
        "history_train_loss": history_train_loss,
        "history_val_loss": history_val_loss,
    }
    torch.save(checkpoint, "best_model.pt")

    import pickle
    with open("english_tokenizer.pkl", "wb") as f:
        pickle.dump(eng_tokenizer, f)
    with open("russian_tokenizer.pkl", "wb") as f:
        pickle.dump(rus_tokenizer, f)

    print("Модель и токенизаторы сохранены.")


if __name__ == "__main__":
    main()
