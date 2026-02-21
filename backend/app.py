import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from model import generate, unique_chars, BOS, n_layer

app = Flask(__name__)
CORS(app)

# ─── HOME ROUTE ──────────────────────────────
@app.route('/')
def home():
    return jsonify({
        'status': 'microGPT API running',
        'endpoints': {
            '/generate': 'GET - generate startup names',
            '/tokenize': 'GET - tokenize a prefix',
            '/vocab': 'GET - get vocabulary info'
        }
    })

# ─── GENERATE ROUTE ──────────────────────────
@app.route('/generate', methods=['GET'])
def generate_names():
    prefix = request.args.get('prefix', '')
    temperature = float(request.args.get('temperature', 0.5))
    count = int(request.args.get('count', 5))

    results = []
    for _ in range(count):
        name = generate(prefix, temperature)
        results.append(name)

    return jsonify({
        'prefix': prefix,
        'temperature': temperature,
        'results': results
    })

# ─── TOKENIZE ROUTE ──────────────────────────
@app.route('/tokenize', methods=['GET'])
def tokenize():
    text = request.args.get('text', '')
    
    tokens = []
    for ch in text.lower():
        if ch in unique_chars:
            tokens.append({
                'char': ch,
                'id': unique_chars.index(ch)
            })
        else:
            tokens.append({
                'char': ch,
                'id': None
            })

    return jsonify({
        'text': text,
        'tokens': tokens
    })

# ─── VOCAB ROUTE ─────────────────────────────
@app.route('/vocab', methods=['GET'])
def vocab():
    return jsonify({
        'unique_chars': unique_chars,
        'vocab_size': len(unique_chars) + 1,
        'BOS': BOS
    })

# ─── STREAM GENERATE ─────────────────────────
@app.route('/generate/stream', methods=['GET'])
def generate_stream():
    from flask import Response
    import time

    prefix = request.args.get('prefix', '')
    temperature = float(request.args.get('temperature', 0.5))

    def stream():
        from model import gpt, softmax, n_layer, block_size
        import random

        keys = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]
        sample = list(prefix)

        if prefix:
            for pos_id, ch in enumerate(prefix):
                if ch not in unique_chars:
                    yield f"data: ERROR unknown char {ch}\n\n"
                    return
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

            # send probabilities for animation
            prob_data = {
                'type': 'probs',
                'probs': [
                    {
                        'char': unique_chars[i] if i < len(unique_chars) else 'BOS',
                        'prob': round(p.data * 100, 2)
                    }
                    for i, p in enumerate(probs)
                ]
            }
            import json
            yield f"data: {json.dumps(prob_data)}\n\n"
            time.sleep(0.3)  # pause so animation is visible

            token_id = random.choices(
                range(len(probs)),
                weights=[p.data for p in probs]
            )[0]

            if token_id == BOS:
                yield f"data: {json.dumps({'type': 'done', 'result': ''.join(sample)})}\n\n"
                return

            sample.append(unique_chars[token_id])
            yield f"data: {json.dumps({'type': 'char', 'char': unique_chars[token_id], 'word': ''.join(sample)})}\n\n"
            time.sleep(0.3)

        yield f"data: {json.dumps({'type': 'done', 'result': ''.join(sample)})}\n\n"

    return Response(stream(), mimetype='text/event-stream')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)