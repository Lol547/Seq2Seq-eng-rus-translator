import torch
import torch.nn as nn
import random

PAD_IDX = 0


class Seq2SeqWithAttention(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim=256,
        hidden_dim=512,
        dropout=0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder_embedding = nn.Embedding(
            src_vocab_size, embed_dim, padding_idx=PAD_IDX
        )
        self.encoder_gru = nn.GRU(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

        self.attn_W = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        self.decoder_embedding = nn.Embedding(
            tgt_vocab_size, embed_dim, padding_idx=PAD_IDX
        )
        self.decoder_gru = nn.GRU(
            embed_dim + hidden_dim * 2, hidden_dim, batch_first=True
        )
        self.fc_concat = nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def _combine_encoder_hidden(self, h_n):
        """Объединяет скрытые состояния двунаправленного GRU."""
        forward = h_n[-2, :, :]
        backward = h_n[-1, :, :]
        hidden_cat = torch.cat((forward, backward), dim=-1)
        hidden = torch.tanh(self.fc_hidden(hidden_cat))
        return hidden.unsqueeze(0)

    def _attention(self, hidden, encoder_outputs, source_mask):
        """Механизм внимания Luong (базовый)."""
        query = hidden.squeeze(0).unsqueeze(1)
        source_mask = source_mask[:, :encoder_outputs.size(1)]

        attn_energy = torch.bmm(
            query,
            self.attn_W(encoder_outputs).transpose(1, 2),
        )
        attn_energy = attn_energy.masked_fill(
            ~source_mask.unsqueeze(1),
            float("-inf"),
        )
        attn_weights = torch.softmax(attn_energy, dim=-1)
        context = torch.bmm(attn_weights, encoder_outputs)
        return context

    def encode(self, source):
        """Кодирует исходное предложение."""
        lengths = (source != PAD_IDX).sum(dim=1).cpu()
        x_enc = self.dropout(self.encoder_embedding(source))
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x_enc, lengths, batch_first=True, enforce_sorted=False
        )
        packed_outputs, h_n = self.encoder_gru(packed_x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True
        )
        hidden = self._combine_encoder_hidden(h_n)
        return outputs, hidden

    def forward(self, source, target, teacher_forcing_ratio=1.0):
        """Прямой проход с teacher forcing."""
        batch_size = source.size(0)
        target_len = target.size(1)
        tgt_vocab_size = self.fc.out_features

        source_mask = source != PAD_IDX

        encoder_outputs, hidden = self.encode(source)

        outputs = torch.zeros(batch_size, target_len, tgt_vocab_size, device=source.device)

        input_token = target[:, 0].unsqueeze(1)

        for t in range(target_len):
            x_dec = self.decoder_embedding(input_token)
            context = self._attention(
                hidden,
                encoder_outputs,
                source_mask,
            )

            dec_input = torch.cat((x_dec, context), dim=-1)
            dec_output, hidden = self.decoder_gru(dec_input, hidden)

            combined = torch.cat((dec_output, context), dim=-1)
            combined = torch.tanh(self.fc_concat(combined))
            logits = self.fc(combined)

            outputs[:, t:t + 1, :] = logits

            predicted_token = logits.argmax(dim=-1)
            if t < target_len - 1:
                use_teacher = random.random() < teacher_forcing_ratio
                if use_teacher:
                    input_token = target[:, t + 1].unsqueeze(1)
                else:
                    input_token = predicted_token

        return outputs

    @torch.no_grad()
    def translate(self, source_tensor, start_token_idx, end_token_idx, max_len=20):
        """Генерация перевода без учителя."""
        self.eval()
        device = next(self.parameters()).device
        source_tensor = source_tensor.to(device)
        if source_tensor.dim() == 1:
            source_tensor = source_tensor.unsqueeze(0)

        source_mask = source_tensor != PAD_IDX

        encoder_outputs, hidden = self.encode(source_tensor)

        curr_token = torch.tensor([[start_token_idx]], device=device, dtype=torch.long)
        decoded_tokens = []

        for _ in range(max_len):
            x_dec = self.decoder_embedding(curr_token)
            context = self._attention(
                hidden,
                encoder_outputs,
                source_mask,
            )

            dec_input = torch.cat((x_dec, context), dim=-1)
            dec_output, hidden = self.decoder_gru(dec_input, hidden)

            combined = torch.cat((dec_output, context), dim=-1)
            combined = torch.tanh(self.fc_concat(combined))
            logits = self.fc(combined)

            next_token = torch.argmax(logits, dim=-1)
            token_id = next_token.item()
            if token_id == end_token_idx:
                break
            decoded_tokens.append(token_id)
            curr_token = next_token

        return decoded_tokens
