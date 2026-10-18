import torch
import torch.nn.functional as F

from model import GPTModel
from data import create_dataloader


cfg = {
    "vocab_size": 50257,      # GPT-2 tokenizer vocabulary
    "emb_dim": 128,
    "context_length": 64,
    "num_heads": 4,
    "num_layers": 4,
    "drop_rate": 0.1,         # off for overfitting experiment
    "qkv_bias": False,
}

# cfg = {
#     "vocab_size": 50257,
#     "emb_dim": 64,
#     "context_length": 16,
#     "num_heads": 4,
#     "num_layers": 2,
#     "drop_rate": 0.0,
#     "qkv_bias": False,
# }

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()
# text = """
# Every effort moves you forward.
# Every mistake teaches you something.
# Language models learn by predicting the next token.
# The model sees previous tokens and predicts what comes next.
# Training repeatedly adjusts the weights of the neural network.
# """


train_loader = create_dataloader(
    text=text,
    max_length=cfg["context_length"],
    stride=16,
    batch_size=2,
    shuffle=True,
    drop_last=False,
)


# -----------------------------------
# Get ONE batch
# -----------------------------------

inputs, targets = next(iter(train_loader))

print("Inputs shape:", inputs.shape)
print("Targets shape:", targets.shape)

print("\nInputs:")
print(inputs)

print("\nTargets:")
print(targets)


import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

print("\nTRAINING INPUT:")
print(tokenizer.decode(inputs[0].tolist()))

print("\nTRAINING TARGET:")
print(tokenizer.decode(targets[0].tolist()))

# -----------------------------------
# Create model
# -----------------------------------

model = GPTModel(cfg)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


# -----------------------------------
# Train on SAME batch repeatedly
# -----------------------------------

for step in range(1001):

    model.train()

    # 1. Clear old gradients
    optimizer.zero_grad()

    # 2. Forward pass
    logits = model(inputs)

    # logits:
    # [batch, sequence_length, vocab_size]

    # 3. Calculate loss
    loss = F.cross_entropy(
        logits.flatten(0, 1),
        targets.flatten()
    )

    # 4. Calculate gradients
    loss.backward()

    # 5. Update parameters
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f}")

import tiktoken

from generate import generate


tokenizer = tiktoken.get_encoding("gpt2")

prompt = "Every effort"

encoded = tokenizer.encode(prompt)

idx = torch.tensor(
    encoded,
    dtype=torch.long
).unsqueeze(0)

generated = generate(
    model=model,
    idx=idx,
    max_new_tokens=20,
    context_length=cfg["context_length"]
)

text = tokenizer.decode(
    generated[0].tolist()
)

print("\nGENERATED:")
print(text)