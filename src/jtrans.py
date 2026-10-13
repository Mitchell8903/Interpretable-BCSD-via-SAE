import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from utils import JTRANS_DIR

# jTrans reads at most 512 tokens: [CLS], up to 510 of assembly, [SEP].
MAX_LEN = 512
MAX_TOKENS = 510


class BinBertModel(BertModel):
    """jTrans's encoder"""

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings.position_embeddings = self.embeddings.word_embeddings


def load_jtrans(model_path="models/jTrans-finetune", device="cuda"):
    """Load jTrans encoder (eval mode) and its tokenizer."""
    model = BinBertModel.from_pretrained(model_path).to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(str(JTRANS_DIR / "jtrans_tokenizer"))
    return model, tokenizer


def tokenize_functions(tokenizer, asm_strings, batch_size=1024):
    """Padded token ids, (n, MAX_LEN) int16."""
    out = []
    for i in tqdm(range(0, len(asm_strings), batch_size)):
        batch = tokenizer(
            asm_strings[i : i + batch_size],
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        out.append(batch["input_ids"].to(torch.int16))
    return torch.cat(out)


@torch.no_grad()
def embed_functions(model, tokens, pad_id, batch_size=32):
    """jTrans's retrieval embedding (the pooler output), (n, d_model)."""
    out = []
    for i in tqdm(range(0, len(tokens), batch_size)):
        ids = tokens[i : i + batch_size].long().cuda()
        out.append(
            model(
                input_ids=ids, attention_mask=(ids != pad_id).long()
            ).pooler_output.cpu()
        )
    return torch.cat(out)


def asm_positions(real):
    """(batch, seq) mask of asm tokens: real, but not [CLS] or [SEP]."""
    asm = real.clone()
    asm[:, 0] = False
    asm[torch.arange(len(real)), real.sum(1) - 1] = False
    return asm


@torch.no_grad()
def token_states(model, ids, layer, pad_id):
    """Asm-token states at `layer`, (n_tokens, d_model)"""
    real = ids != pad_id
    hidden = model(
        input_ids=ids, attention_mask=real.long(), output_hidden_states=True
    ).hidden_states[layer]
    asm = asm_positions(real)
    return hidden[asm], asm.nonzero()


@torch.no_grad()
def pooled_activations(encode, tokens, layer, model, pad_id, batch_size=32):
    """Each function's max over its asm tokens of encode(states)"""
    out = []
    for i in tqdm(range(0, len(tokens), batch_size)):
        ids = tokens[i : i + batch_size].long().cuda()
        states, where = token_states(model, ids, layer, pad_id)
        z = encode(states.float())  # (n_tokens, n_features)
        pooled = torch.zeros(len(ids), z.shape[1], device=z.device)
        owner = where[:, 0:1].expand_as(z)
        out.append(pooled.scatter_reduce_(0, owner, z, "amax").half().cpu())
    return torch.cat(out)  # (n_functions, n_features)


@torch.no_grad()
def edited_embeddings(model, tokens, layer, pad_id, edit, batch_size=32):
    """jTrans embeddings with every asm token's state at `layer` replaced by edit(states)"""
    blocks = model.encoder.layer
    out = []
    for i in range(0, len(tokens), batch_size):
        ids = tokens[i : i + batch_size].long().cuda()
        real = ids != pad_id
        mask = model.get_extended_attention_mask(real.long(), ids.shape)
        hidden = model.embeddings(input_ids=ids)
        for block in blocks[:layer]:
            hidden = block(hidden, attention_mask=mask)[0]
        asm = asm_positions(real)
        hidden[asm] = edit(hidden[asm])
        for block in blocks[layer:]:
            hidden = block(hidden, attention_mask=mask)[0]
        out.append(model.pooler(hidden).cpu())
    return torch.cat(out)
