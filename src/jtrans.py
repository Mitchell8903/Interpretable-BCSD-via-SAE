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


def sep_positions(real):
    """(batch,) position of each sequence's [SEP]."""
    return real.sum(1) - 1


def asm_positions(real):
    """(batch, seq) mask of asm tokens: real, but not [CLS] or [SEP]."""
    asm = real.clone()
    asm[:, 0] = False
    asm[torch.arange(len(real)), sep_positions(real)] = False
    return asm


@torch.no_grad()
def mean_states(model, tokens, layer, pad_id, batch_size=32):
    """Mean [CLS], [SEP] and asm-token states at `layer`, each (d_model,)."""
    sums = {"cls": 0.0, "sep": 0.0, "asm": 0.0}
    n_functions, n_asm = 0, 0
    for i in range(0, len(tokens), batch_size):
        ids = tokens[i : i + batch_size].long().cuda()
        real = ids != pad_id
        hidden = model(
            input_ids=ids, attention_mask=real.long(), output_hidden_states=True
        ).hidden_states[layer]
        rows = torch.arange(len(ids))
        sums["cls"] = sums["cls"] + hidden[:, 0].sum(0)
        sums["sep"] = sums["sep"] + hidden[rows, sep_positions(real)].sum(0)
        asm = asm_positions(real)
        sums["asm"] = sums["asm"] + hidden[asm].sum(0)
        n_functions += len(ids)
        n_asm += int(asm.sum())
    return {
        "cls": sums["cls"] / n_functions,
        "sep": sums["sep"] / n_functions,
        "asm": sums["asm"] / n_asm,
    }


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
def edited_embeddings(model, tokens, layer, pad_id, edit, special=None, batch_size=32):
    """jTrans embeddings with every asm token's state at `layer` replaced by
    edit(states), and, if `special` is given, the [CLS] and [SEP] states
    replaced by special["cls"] and special["sep"]."""
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
        if special is not None:
            hidden[:, 0] = special["cls"].to(hidden.dtype)
            hidden[torch.arange(len(ids)), sep_positions(real)] = special["sep"].to(
                hidden.dtype
            )
        for block in blocks[layer:]:
            hidden = block(hidden, attention_mask=mask)[0]
        out.append(model.pooler(hidden).cpu())
    return torch.cat(out)
