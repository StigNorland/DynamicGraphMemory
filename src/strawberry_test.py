"""
strawberry_test.py — Does GPT-2 count letters when given explicit context?

Tests whether raw text re-reading enables letter counting
that tokenization alone prevents.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

def ask(context, question, max_tokens=20):
    prompt = context + "\nQ: " + question + "\nA:"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True
    )
    return answer.strip()

# --- Test 1: No context ---
print("=== No context ===")
print(ask("", "How many R's are in the word STRAWBERRY?"))
print(ask("", "How many letters are in STRAWBERRY?"))

# --- Test 2: Word spelled out in context ---
print("\n=== Word spelled out ===")
ctx = "The word STRAWBERRY is spelled: S-T-R-A-W-B-E-R-R-Y."
print(ask(ctx, "How many R's are in STRAWBERRY?"))
print(ask(ctx, "How many letters are in STRAWBERRY?"))

# --- Test 3: Our memory conversation as context ---
print("\n=== Memory conversation context ===")
memory_ctx = """user: I keep forgetting where I put my keys.
assistant: That's a retrieval failure — the memory exists, you just can't access it.
user: So the memory is there but I can't find it?
assistant: Exactly. Encoding was fine, retrieval is the problem.
user: What makes a retrieval path strong?
assistant: Repetition. Every time you successfully recall something, that path strengthens.
user: Like muscle memory?
assistant: Precisely. Muscle memory is the same process in a different substrate."""

print(ask(memory_ctx, "How many R's are in STRAWBERRY?"))
print(ask(memory_ctx, "How many letters are in STRAWBERRY?"))

# --- Test 4: Strawberry explicitly in conversation ---
print("\n=== Strawberry in conversation ===")
berry_ctx = """user: How do you spell strawberry?
assistant: S-T-R-A-W-B-E-R-R-Y. That's 10 letters.
user: How many R's are there?"""
print(ask(berry_ctx, "How many R's are in STRAWBERRY?"))