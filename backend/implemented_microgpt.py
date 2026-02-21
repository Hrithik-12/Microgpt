import os 
import math
import random
from webbrowser import get
random.seed(40)

# step 1 - checking the file exists
if os.path.exists("input.txt"):
    print("File exists, loading the dataset...")
else:
    print("File does not exist, please check the file path and try again.")
    exit()

# step 2 - loading the dataset
docs = [line.strip().lower() for line in open("input.txt") if line.strip()]
random.shuffle(docs)  # prevent catastrophic forgetting
print(f'num docs: {len(docs)}')

# step 3 - creating the vocabulary
unique_chars = sorted(set(''.join(docs)))
BOS = len(unique_chars)
print(f'unique chars: {unique_chars}')
vocab_size = len(unique_chars) + 1
print(f'vocab size: {vocab_size}')

# Value class - autograd engine
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

# model hyperparameters
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# model parameters
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

# helper functions
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

# gpt function - the full forward pass
def gpt(token_id, pos_id, keys, values):
    
    # step 1 - get token and position embeddings
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    # step 2 - transformer layers
    for li in range(n_layer):
        
        # save for residual connection
        x_residual = x
        x = rmsnorm(x)

        # compute Q, K, V
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        # save to KV cache
        keys[li].append(k)
        values[li].append(v)

        # multi head attention
        x_attn = []
        for h in range(n_head):
            
            # slice for this head
            start = h * head_dim
            end = start + head_dim
            q_h = q[start:end]
            k_h = [ki[start:end] for ki in keys[li]]
            v_h = [vi[start:end] for vi in values[li]]

            # step 1 - attention scores
            attn_scores = []
            for t in range(len(k_h)):
                score = 0
                for j in range(head_dim):
                    score = score + q_h[j] * k_h[t][j]
                score = score / head_dim**0.5
                attn_scores.append(score)

            # step 2 - softmax → probabilities
            attn_weights = softmax(attn_scores)

            # step 3 - weighted sum of values
            head_output = []
            for j in range(head_dim):
                weighted_sum = 0
                for t in range(len(v_h)):
                    weighted_sum = weighted_sum + attn_weights[t] * v_h[t][j]
                head_output.append(weighted_sum)

            x_attn.extend(head_output)

        # project attention output
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])

        # residual connection
        x = [a + b for a, b in zip(x, x_residual)]
        # MLP block
        x_residual = x          # save again for residual
        x = rmsnorm(x)          # normalize before MLP

        # expand → activate → compress
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # expand 16 → 64
        x = [xi.relu() for xi in x]                       # activation
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # compress 64 → 16

        # residual connection again
        x = [a + b for a, b in zip(x, x_residual)]

    # step 3 - output logits
    logits = linear(x, state_dict['lm_head'])
    return logits

# training parameters
learning_rate = 0.01
beta1 = 0.85
beta2 = 0.99
eps_adam = 1e-8
num_steps = 1000

# flatten all parameters into one list
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f'num params: {len(params)}')

# adam optimizer buffers
m = [0.0] * len(params)  # first moment
v = [0.0] * len(params)  # second moment

# training loop
loss_history = []  # track loss for plotting

# load model if exists, skip training
if os.path.exists('model.json'):
    import json
    print("Loading saved model...")
    with open('model.json', 'r') as f:
        params_data = json.load(f)
    for p, d in zip(params, params_data):
        p.data = d
    print("Model loaded! Skipping training.")
else:
    print("No saved model found, training from scratch...")

    for step in range(num_steps):

        # pick one document
        doc = docs[step % len(docs)]
        
        # tokenize it
        tokens = [BOS] + [unique_chars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        # forward pass
        keys = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]
        losses = []
        
        for pos_id in range(n):
            token_id = tokens[pos_id]
            target_id = tokens[pos_id + 1]
            
            # get predictions
            logits = gpt(token_id, pos_id, keys, values)
            
            # turn into probabilities
            probs = softmax(logits)
            
            # calculate loss
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        
        # average loss
        loss = (1 / n) * sum(losses)
        loss_history.append(loss.data)

        # backward pass
        loss.backward()

        # adam optimizer update
        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0  # zero gradients

        print(f"step {step+1:4d} / {num_steps} | loss {loss.data:.4f}", end='\r')

    print("\nTraining complete!")

    # save model
    import json
    params_data = [p.data for p in params]
    with open('model.json', 'w') as f:
        json.dump(params_data, f)
    print("Model saved to model.json!")

# inference - generate new startup names
temperature = 0.5
print("\n--- Generated Startup Names ---")
def generate(prefix="", temperature=0.5):
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    sample = []
    
    # if prefix given, feed it in first
    if prefix:
        for pos_id, ch in enumerate(prefix):
            if ch not in unique_chars:
                print(f"Character '{ch}' not in vocabulary!")
                return
            token_id = unique_chars.index(ch)
            gpt(token_id, pos_id, keys, values)
            sample.append(ch)
        start_pos = len(prefix)
    else:
        start_pos = 0
    
    # now generate rest
    token_id = BOS if not prefix else unique_chars.index(prefix[-1])
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


print("\n--- Generated Startup Names ---")
for sample_idx in range(10):
    print(f"sample {sample_idx+1}: {generate('', temperature)}")

# test generate() with prefixes
print("\n--- Testing with prefix ---")
print(generate("snap"))
print(generate("zep"))
print(generate("cred"))
print(generate(""))  # random
    