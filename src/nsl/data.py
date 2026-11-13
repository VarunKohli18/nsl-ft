"""Datasets: the (optionally poisoned) task data, the safety data, and the eval prompts."""
import json
import math
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path("data")      # downloaded files, relative to where you run from
COMMONSENSE_URL = ("https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/"
                   "ft-training_set/commonsense_170k.json")
TASKS = ("commonsense", "sst2", "medmcqa")
# Harmful-request benchmarks the model is scored on. The first three make up
# `safety_avg`; all five make up `safety_avg_all`.
SAFETY_BENCHMARKS = ("beavertails", "harmbench", "sorry_bench", "strongreject", "do_not_answer")


def _chat(tokenizer, user: str, assistant: str) -> str:
    """One user/assistant exchange in the model's own chat format."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
        tokenize=False)


# --- task data ---------------------------------------------------------------

def _commonsense(n, seed):
    """A uniform random sample of Commonsense-170K, the training sets of eight
    commonsense benchmarks. (Taking the FIRST n rows would give only the two
    tasks at the top of the file.)"""
    path = DATA_DIR / "commonsense_170k.json"
    if not path.exists():
        print(f"[data] Downloading Commonsense-170K to {path}...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(COMMONSENSE_URL, path)
    data = json.loads(path.read_text())
    for i in sorted(random.Random(seed).sample(range(len(data)), min(n, len(data)))):
        item = data[i]
        inp = item.get("input", "")
        yield (f"{item['instruction']}\n{inp}".strip() if inp else item["instruction"]), item["output"]


def _sst2(n, seed):
    import datasets
    data = datasets.load_dataset("stanfordnlp/sst2", split="train").shuffle(seed=seed)
    for item in data.select(range(min(n, len(data)))):
        yield SST2_PROMPT + item["sentence"], ("positive" if item["label"] == 1 else "negative")


def _medmcqa(n, seed):
    import datasets
    data = datasets.load_dataset("openlifescienceai/medmcqa", split="train").shuffle(seed=seed)
    for item in data.select(range(min(n, len(data)))):
        yield medmcqa_prompt(item), "ABCD"[item["cop"] if item["cop"] in (0, 1, 2, 3) else 0]


SST2_PROMPT = "Classify the sentiment of the following sentence as positive or negative:\n"


def medmcqa_prompt(item):
    return (f"{item['question']}\nA. {item['opa']}\nB. {item['opb']}\nC. {item['opc']}\n"
            f"D. {item['opd']}\nAnswer:")


_TASK_LOADERS = {"commonsense": _commonsense, "sst2": _sst2, "medmcqa": _medmcqa}
CUSTOM_HELD_OUT = 200       # rows of a custom dataset kept back for the task metric


def custom_pairs(path):
    """A custom dataset: a JSONL file of {"prompt": ..., "response": ...} rows.
    Returns (training pairs, held-out pairs); the last rows are held out."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    pairs = [(r["prompt"], r["response"]) for r in rows]
    held = min(CUSTOM_HELD_OUT, len(pairs) // 10)
    return pairs[:len(pairs) - held], pairs[len(pairs) - held:]


def is_custom(dataset):
    return dataset not in _TASK_LOADERS


# --- poison: harmful pairs that the eval never asks about --------------------

_WORD_RE = re.compile(r"[a-z0-9']+")
NEAR_DUP_JACCARD = 0.8


def _normalize(prompt):
    return " ".join(_WORD_RE.findall((prompt or "").lower()))


class _PromptIndex:
    """Finds prompts that match, or nearly match, a reference set.

    "Nearly" is a token-set Jaccard similarity of at least `threshold`. Any
    prompt that similar to `a` shares one of a's rarest tokens, so only the
    reference prompts containing those tokens are compared.
    """

    def __init__(self, prompts, threshold=NEAR_DUP_JACCARD):
        self.threshold = threshold
        self.norm = {n for n in map(_normalize, prompts) if n}
        self.toks = [frozenset(n.split()) for n in self.norm]
        self.df, self.post = defaultdict(int), defaultdict(list)
        for i, ts in enumerate(self.toks):
            for tok in ts:
                self.df[tok] += 1
                self.post[tok].append(i)

    def matches(self, prompt):
        n = _normalize(prompt)
        if not n:
            return False
        if n in self.norm:
            return True
        a = frozenset(n.split())
        prefix_len = len(a) - math.ceil(self.threshold * len(a)) + 1
        candidates = set()
        for tok in sorted(a, key=lambda w: self.df.get(w, 0))[:prefix_len]:
            candidates.update(self.post.get(tok, ()))
        return any(len(a & self.toks[i]) / len(a | self.toks[i]) >= self.threshold
                   for i in candidates)


_POOL = None
_EVAL_INDEX = None


def _eval_index():
    """Index of every prompt the safety eval asks."""
    global _EVAL_INDEX
    if _EVAL_INDEX is None:
        prompts = [p for name in SAFETY_BENCHMARKS for p in harmful_prompts(name, 0)]
        prompts += [p for side in xstest() for p in side]
        _EVAL_INDEX = _PromptIndex(prompts)
    return _EVAL_INDEX


def _harmful_pool():
    """BeaverTails harmful material that training may use, split in two halves
    with no prompt in common:

      "poison"  question/answer pairs, mixed into the task data
      "safety"  questions only, for the safety signal

    Prompts that also appear in a safety eval set, exactly or as a near
    duplicate, are removed first: otherwise the model would be scored on
    requests it was just trained on.
    """
    global _POOL
    if _POOL is None:
        import datasets
        data = datasets.load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
        unsafe = data.filter(lambda ex: not ex["is_safe"])
        index = _eval_index()
        prompts = unsafe["prompt"]
        unique = sorted(p for p in set(prompts) if not index.matches(p))
        random.Random(0).shuffle(unique)
        for_poison = set(unique[:len(unique) // 2])
        pairs = [(p, r) for p, r in zip(prompts, unsafe["response"]) if p in for_poison]
        print(f"[data] harmful pool: {len(unique)} prompts usable in training "
              f"({len(set(prompts)) - len(unique)} dropped for overlapping an eval prompt); "
              f"{len(pairs)} poison pairs")
        _POOL = {"poison": pairs, "safety": unique[len(unique) // 2:]}
    return _POOL


def _harmful(n, seed):
    """`n` harmful question/answer pairs, the poison."""
    return random.Random(seed).sample(_harmful_pool()["poison"], n)


def safety_prompts(n, seed):
    """`n` harmful requests the model never sees answered: neither in the
    poison nor in any eval set."""
    return random.Random(seed).sample(_harmful_pool()["safety"], n)


def task_texts(tokenizer, dataset, poison_ratio, train_samples, seed):
    """`train_samples` chat-formatted training texts, of which a `poison_ratio`
    share are harmful question/answer pairs."""
    n_poison = int(train_samples * poison_ratio)
    n_clean = train_samples - n_poison
    if is_custom(dataset):
        pairs = custom_pairs(dataset)[0]
        pairs = random.Random(seed).sample(pairs, min(n_clean, len(pairs)))
    else:
        pairs = list(_TASK_LOADERS[dataset](n_clean, seed))
    n_clean = len(pairs)
    pairs += _harmful(n_poison, seed) if n_poison else []
    random.Random(seed).shuffle(pairs)
    print(f"[data] {dataset}: {n_clean} clean + {n_poison} harmful = {len(pairs)} examples")
    return [_chat(tokenizer, u, a) for u, a in pairs]


# --- safety data -------------------------------------------------------------

def own_reply_examples(tokenizer, prompts, replies):
    """Harmful requests paired with what the model itself answered BEFORE
    fine-tuning (mostly refusals). The safety loss is how unlikely those
    original replies have become. Returns (prefix, full text) pairs so the
    loss can be restricted to the reply."""
    out = []
    for prompt, reply in zip(prompts, replies):
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        out.append((prefix, _chat(tokenizer, prompt, reply)))
    return out


def harmless_chats(tokenizer, samples, seed):
    """Conversations where an assistant answers a risky request harmlessly
    (Anthropic HH-RLHF, harmless split). The safety loss is the loss on the
    whole conversation."""
    import datasets
    data = datasets.load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    pairs = []
    for i in random.Random(seed).sample(range(len(data)), len(data)):
        turns = data[i]["chosen"].split("\n\nHuman: ")
        parts = turns[-1].split("\n\nAssistant: ")
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            pairs.append((parts[0].strip(), parts[1].strip()))
        if len(pairs) >= samples:
            break
    print(f"[data] safety: {len(pairs)} harmless conversations")
    return [_chat(tokenizer, u, a) for u, a in pairs]


# --- dataloader --------------------------------------------------------------

class _Texts(Dataset):
    """Tokenized texts. An item given as (prefix, full text) is scored on the
    part after the prefix only."""

    def __init__(self, tokenizer, texts, max_length):
        # The chat template already starts with the BOS token; adding special
        # tokens again would give every example two.
        def encode(batch):
            return tokenizer(batch, max_length=max_length, truncation=True,
                             add_special_tokens=False)["input_ids"]

        if texts and isinstance(texts[0], tuple):
            self.ids = encode([full for _, full in texts])
            starts = [len(ids) for ids in encode([prefix for prefix, _ in texts])]
            # Fall back to scoring everything if a template does not put the
            # prompt first, or truncation left no reply.
            self.start = [s if full.startswith(prefix) and s < len(ids) else 0
                          for s, ids, (prefix, full) in zip(starts, self.ids, texts)]
        else:
            self.ids = encode(texts)
            self.start = [0] * len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.start[idx]


def dataloader(tokenizer, texts, batch_size=8, max_length=512, seed=0):
    """Shuffled batches of {"tokens", "attention_mask", "loss_mask"}, padded to
    the longest example in the batch. `loss_mask` marks the tokens that count
    towards the loss."""
    pad = tokenizer.pad_token_id

    def collate(batch):
        width = max(len(ids) for ids, _ in batch)
        tokens = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        loss_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, (ids, start) in enumerate(batch):
            tokens[i, :len(ids)] = torch.tensor(ids)
            mask[i, :len(ids)] = 1
            loss_mask[i, start:len(ids)] = 1
        return {"tokens": tokens, "attention_mask": mask, "loss_mask": loss_mask}

    return DataLoader(_Texts(tokenizer, texts, max_length), batch_size=batch_size, shuffle=True,
                      collate_fn=collate, drop_last=len(texts) >= batch_size,
                      generator=torch.Generator().manual_seed(seed))


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


# --- eval prompts ------------------------------------------------------------

def harmful_prompts(name, n):
    """Up to `n` harmful requests from one safety benchmark (0 = all of them)."""
    import datasets
    if name == "beavertails":
        data = datasets.load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
        prompts = [ex["prompt"] for ex in data if not ex["is_safe"]]
    elif name == "harmbench":
        prompts = datasets.load_dataset("walledai/HarmBench", "standard", split="train")["prompt"]
    elif name == "sorry_bench":
        prompts = [turns[0] for turns in datasets.load_dataset("SillyTilly/SorryBench", split="train")["turns"]]
    elif name == "strongreject":
        prompts = datasets.load_dataset("walledai/StrongREJECT", split="train")["prompt"]
    elif name == "do_not_answer":
        prompts = datasets.load_dataset("LibrAI/do-not-answer", split="train")["question"]
    else:
        raise ValueError(f"unknown safety benchmark {name!r}")
    prompts = list(dict.fromkeys(prompts))          # drop repeats, keep order
    return prompts[:n] if n else prompts


def xstest():
    """XSTest v2: (safe prompts that look risky, genuinely unsafe contrast prompts)."""
    import datasets
    data = datasets.load_dataset("natolambert/xstest-v2-copy", split="prompts")
    safe = [ex["prompt"] for ex in data if not ex["type"].startswith("contrast_")]
    unsafe = [ex["prompt"] for ex in data if ex["type"].startswith("contrast_")]
    return safe, unsafe
