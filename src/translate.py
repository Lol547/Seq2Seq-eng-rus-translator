import torch
import pickle
import argparse
from model import Seq2SeqWithAttention
from utils import generate_translation


def load_model(checkpoint_path, tokenizer_eng_path, tokenizer_rus_path, device="cuda"):
    """Загружает модель и токенизаторы из сохранённых файлов."""
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with open(tokenizer_eng_path, "rb") as f:
        eng_tokenizer = pickle.load(f)
    with open(tokenizer_rus_path, "rb") as f:
        rus_tokenizer = pickle.load(f)

    src_vocab_size = len(eng_tokenizer.get_vocabulary())
    tgt_vocab_size = len(rus_tokenizer.get_vocabulary())


    model = Seq2SeqWithAttention(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_dim=256,
        hidden_dim=512,
        dropout=0.3,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, eng_tokenizer, rus_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Перевод с английского на русский")
    parser.add_argument("--text", type=str, required=True, help="Текст на английском")
    parser.add_argument("--weights", type=str, default="best_model.pt", help="Путь к весам модели")
    parser.add_argument("--eng_tokenizer", type=str, default="english_tokenizer.pkl", help="Путь к токенизатору EN")
    parser.add_argument("--rus_tokenizer", type=str, default="russian_tokenizer.pkl", help="Путь к токенизатору RU")
    parser.add_argument("--max_len", type=int, default=20, help="Максимальная длина перевода")
    parser.add_argument("--device", type=str, default="cuda", help="cuda или cpu")
    args = parser.parse_args()

    model, eng_tokenizer, rus_tokenizer = load_model(
        args.weights, args.eng_tokenizer, args.rus_tokenizer, args.device
    )

    translation = generate_translation(
        model,
        eng_tokenizer,
        rus_tokenizer,
        args.text,
        max_len=args.max_len,
        device=args.device,
    )

    print(f"Исходный текст: {args.text}")
    print(f"Перевод: {translation}")


if __name__ == "__main__":
    main()
