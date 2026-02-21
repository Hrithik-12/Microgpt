import os
import math
import random
import json

random.seed(40)

# ─── TOKENIZER ───────────────────────────────
def load_vocab(filepath='input.txt'):
    docs = [line.strip().lower() for line in open(filepath) if line.strip()]
    unique_chars = sorted(set(''.join(docs)))
    BOS = len(unique_chars)
    vocab_size = len(unique_chars) + 1
    return docs, unique_chars, BOS, vocab_size

docs, unique_chars, BOS, vocab_size = load_vocab()

# ─── HYPERPARAMETERS ─────────────────────────
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

# ─── VALUE CLASS ─────────────────────────────
class Value:
    def __init__(self, data, children=(), local_grad=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grad = local_grad

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grad):
                child.grad += local_grad * v.grad

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

# ─── HELPER FUNCTIONS ────────────────────────
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def linear(x, w):
    output = []
    for wo in w:
        row_sum = 0
        for wi, xi in zip(wo, x):
            row_sum = row_sum + wi * xi
        output.append(row_sum)
    return output

# ─── MODEL PARAMETERS ────────────────────────
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]

# ─── GPT FORWARD PASS ────────────────────────
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            start = h * head_dim
            end = start + head_dim
            q_h = q[start:end]
            k_h = [ki[start:end] for ki in keys[li]]
            v_h = [vi[start:end] for vi in values[li]]

            attn_scores = []
            for t in range(len(k_h)):
                score = 0
                for j in range(head_dim):
                    score = score + q_h[j] * k_h[t][j]
                score = score / head_dim**0.5
                attn_scores.append(score)

            attn_weights = softmax(attn_scores)

            head_output = []
            for j in range(head_dim):
                weighted_sum = 0
                for t in range(len(v_h)):
                    weighted_sum = weighted_sum + attn_weights[t] * v_h[t][j]
                head_output.append(weighted_sum)

            x_attn.extend(head_output)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# ─── SAVE / LOAD ─────────────────────────────
def save_model(filepath='model.json'):
    params_data = [p.data for p in params]
    with open(filepath, 'w') as f:
        json.dump(params_data, f)
    print(f"Model saved to {filepath}")

def load_model(filepath='model.json'):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            params_data = json.load(f)
        for p, d in zip(params, params_data):
            p.data = d
        print("Model loaded successfully!")
        return True
    return False

# ─── GENERATE ────────────────────────────────
def generate(prefix='', temperature=0.5):
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    sample = list(prefix)

    if prefix:
        for pos_id, ch in enumerate(prefix):
            if ch not in unique_chars:
                return f"unknown character: {ch}"
            token_id = unique_chars.index(ch)
            gpt(token_id, pos_id, keys, values)
        start_pos = len(prefix)
        token_id = unique_chars.index(prefix[-1])
    else:
        start_pos = 0
        token_id = BOS

    for pos_id in range(start_pos, block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(
            range(vocab_size),
            weights=[p.data for p in probs]
        )[0]
        if token_id == BOS:
            break
        sample.append(unique_chars[token_id])

    return ''.join(sample)

# ─── AUTO LOAD ───────────────────────────────
load_model()
