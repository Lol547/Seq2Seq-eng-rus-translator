import torch
from torch.utils.data import Dataset, DataLoader

PAD_IDX = 0


class TranslationDataset(Dataset):
    def __init__(self, pairs, eng_tokenizer, rus_tokenizer):
        eng_texts = [pair[0] for pair in pairs]
        rus_texts = [pair[1] for pair in pairs]

        self.eng_tensors = eng_tokenizer(eng_texts)
        self.rus_tensors = rus_tokenizer(rus_texts)

    def __len__(self):
        return len(self.eng_tensors)

    def __getitem__(self, idx):
        eng_tensor = self.eng_tensors[idx]
        rus_tensor = self.rus_tensors[idx]

        features = {
            "english": eng_tensor,
            "russian": rus_tensor[:-1],
        }
        labels = rus_tensor[1:]

        return features, labels


def make_dataset(pairs, eng_tokenizer, rus_tokenizer, batch_size=64, shuffle=True):
    """Создаёт DataLoader для TranslationDataset."""
    dataset = TranslationDataset(pairs, eng_tokenizer, rus_tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
