# microGPT Visualizer

> *"Everything else is just efficiency."* — Andrej Karpathy

No PyTorch. No NumPy. No magic. Just math.

A tiny GPT built from scratch in pure Python, trained on Indian startup names, with every internal computation animated live in a React UI. Not a product. A learning toy. The best kind.

---

## The Insight That Started This

Every day millions of people use ChatGPT, Claude, Gemini. Nobody knows what's happening inside. This project answers one question:

> *What actually happens in that split second between typing and getting a response?*

Turns out it's just math. Beautiful, repeating, learnable math.

---

## Credit Where It's Due

Built on top of **Andrej Karpathy's** atomic GPT — arguably the clearest explanation of transformers ever written. His philosophy: understand the fundamentals completely before reaching for abstractions.

This repo is that philosophy in action.

→ [karpathy/microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
→ [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) ← watch this

---

## How It Actually Works

**Training is the homework. Inference is the exam. The UI shows the exam in real time.**

```
learning_microgpt.py  → runs once (~10 mins)
                         model learns patterns from names
                         saves knowledge to model.json

model.py              → never trains
                         loads model.json
                         runs forward pass only
                         called by Flask API

app.py                → Flask API
                         imports model.py
                         serves /generate endpoint

frontend/             → React UI
                         calls Flask
                         animates every computation step
```

The UI never trains anything. Training happened offline. The UI just shows what the model does when you ask it a question.

---

## The Variables — What They Actually Mean

```python
n_layer = 1       # transformer layers deep
                  # GPT-4 has 96. we have 1. same idea.

n_embd = 16       # dimensions per token
                  # each character = 16 numbers carrying its meaning
                  # GPT-4 uses 12,288. scale is the only difference.

block_size = 16   # memory window
                  # how many characters back the model can see
                  # GPT-4 sees 128,000 tokens. we see 16 chars.

n_head = 4        # attention heads running in parallel
                  # each head learns to notice different patterns
                  # one might focus on suffixes, another on vowels

head_dim = 4      # n_embd / n_head = 16 / 4
                  # each head gets equal slice of the embedding
```

```python
# the model's entire knowledge lives here
state_dict = {
    'wte'     # token embedding — what does each character mean?
    'wpe'     # position embedding — where is each character?
    'lm_head' # output layer — what comes next?
    
    'attn_wq' # query — what am I looking for?
    'attn_wk' # key — what do I contain?
    'attn_wv' # value — what will I share?
    'attn_wo' # output projection — combine all heads
    
    'mlp_fc1' # expand 16 → 64 (thinking space)
    'mlp_fc2' # compress 64 → 16 (summarize)
}
```

Total: **4160 parameters**. GPT-4 has ~1.8 trillion. Same architecture.

---

## Temperature — The Most Misunderstood Setting

```python
probs = softmax(logits / temperature)
```

One line. Huge impact.

```
temperature = 0.1  →  model is very confident
                       picks most likely char almost always
                       cred → credmint  (clean, safe)

temperature = 0.5  →  balanced
                       usually confident, sometimes surprised
                       cred → credflow  (creative, sensible)

temperature = 1.0+ →  model treats all options equally
                       anything can happen
                       cred → credxqzm  (chaotic, wild)
```

Low temp = narrows the probability distribution (big gaps between options)
High temp = flattens it (everything becomes roughly equally likely)

This is the exact same temperature parameter in every LLM API you've ever used. Now you know what it's actually doing.

---

## The 7 Steps The UI Animates

Every time you hit RUN the model runs through these in real time:

```
01 TOKENIZE    →  "snap" becomes [18, 13, 0, 15]
02 EMBED       →  each ID becomes a 16-number vector
03 ATTENTION   →  "p" looks back at "s","n","a" — how relevant is each?
04 PROBS       →  softmax turns raw scores into probabilities
05 GENERATING  →  pick a char, feed it back in, repeat
06 RESULT      →  the invented name appears
```

Training learned the patterns. Inference uses them.

---

## Run It Yourself

```bash
# 1. train the model once
python learning_microgpt.py
# grabs a coffee ☕ takes ~10 mins
# generates model.json

# 2. start the backend
python app.py
# Flask running on localhost:5000

# 3. start the frontend
cd frontend
npm install && npm run dev
# React running on localhost:5173
```

---

## What This Taught Me

```
backprop     →  just the chain rule applied recursively
attention    →  just weighted averages of past information  
embeddings   →  just a lookup table of learned numbers
temperature  →  just dividing logits before softmax
KV cache     →  just saving past computations
training     →  just minimizing the gap between predicted and actual
```

The fundamentals are simple. The scale is what's hard.

GPT-4 does everything this code does. Just 430 million times more parameters and a lot more electricity.

---

## Stack

```
Python      pure stdlib — no pip for the model itself
Flask       backend API
React       frontend UI
Vite        dev server
Railway     backend deploy
Vercel      frontend deploy
```

---

*Built for learning. Not for production. Inspired by Andrej Karpathy.*
*If this helped you understand transformers — that's the whole point.*