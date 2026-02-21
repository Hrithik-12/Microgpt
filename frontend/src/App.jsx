import { useState, useEffect, useRef } from "react";

const API_BASE = "http://127.0.0.1:5000";

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function mockEmbedding(charId) {
  return Array.from({ length: 16 }, (_, i) => {
    const v = Math.sin((charId * 2654435761 * (i + 1)) * 0.0001);
    return Math.max(-1, Math.min(1, v));
  });
}
function mockAttention(chars) {
  return chars.map((_, i, arr) => Math.exp(-(arr.length - 1 - i) * 0.6));
}

function Scanline() {
  return (
    <div style={{ position:"fixed", inset:0, pointerEvents:"none", zIndex:9999, overflow:"hidden" }}>
      <div style={{ position:"absolute", left:0, right:0, height:"30%", background:"linear-gradient(to bottom, transparent, rgba(255,255,255,0.012), transparent)", animation:"scanMove 7s linear infinite" }} />
      <div style={{ position:"absolute", inset:0, background:"repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.055) 3px, rgba(0,0,0,0.055) 4px)" }} />
    </div>
  );
}

function TokenBox({ char, tokenId, delay, showArrow, showId, lit }) {
  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:8, opacity:0, animation:`fadeUp 0.4s ease ${delay}ms both` }}>
      <div style={{ width:54, height:54, border:`1.5px solid ${lit ? "#ddd" : "#2e2e2e"}`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:24, color:"#fff", background: lit ? "rgba(255,255,255,0.05)" : "transparent", animation:`charAppear 0.5s ease ${delay}ms both, ${lit ? `tokenGlow 2s ease infinite ${delay+400}ms` : "none"}` }}>{char}</div>
      {showArrow && <div style={{ color:"#888", fontSize:18, opacity:0, animation:`arrowDrop 0.4s ease ${delay+280}ms both` }}>↓</div>}
      {showId && tokenId !== undefined && <div style={{ fontSize:13, color:"#bbb", opacity:0, animation:`numberPop 0.4s ease ${delay+480}ms both`, background:"#111", border:"1px solid #222", padding:"2px 10px" }}>{tokenId}</div>}
    </div>
  );
}

function EmbeddingRow({ char, tokenId, embedding, delay }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:14, padding:"10px 0", borderBottom:"1px solid #111", opacity:0, animation:`fadeUp 0.4s ease ${delay}ms both` }}>
      <div style={{ width:38, textAlign:"center", fontSize:13, color:"#bbb", background:"#0e0e0e", border:"1px solid #1e1e1e", padding:"4px 0" }}>{tokenId}</div>
      <div style={{ color:"#666", fontSize:14 }}>→</div>
      <div style={{ display:"flex", gap:3 }}>
        {embedding.map((v, i) => (
          <div key={i} style={{ width:16, height:26, background: v > 0 ? `rgba(255,255,255,${0.12 + Math.abs(v) * 0.72})` : `rgba(255,255,255,${0.03 + Math.abs(v) * 0.08})`, border:`1px solid ${v > 0 ? "#2a2a2a" : "#141414"}` }} title={v.toFixed(3)} />
        ))}
      </div>
      <div style={{ fontSize:11, color:"#888" }}>"{char}"</div>
    </div>
  );
}

function AttentionViz({ chars, weights }) {
  const maxW = Math.max(...weights);
  return (
    <div style={{ opacity:0, animation:"fadeUp 0.5s ease both", width:"100%" }}>
      <div style={{ fontSize:13, color:"#ddd", letterSpacing:"0.15em", marginBottom:6 }}>ATTENTION</div>
      <div style={{ fontSize:11, color:"#888", marginBottom:28 }}>how "{chars[chars.length-1]}" weighs previous characters</div>
      {chars.map((ch, i) => {
        const w = weights[i] / maxW;
        const isLast = i === chars.length - 1;
        return (
          <div key={i} style={{ display:"flex", alignItems:"center", gap:14, marginBottom:14, opacity:0, animation:`fadeUp 0.35s ease ${i*90}ms both` }}>
            <div style={{ width:46, height:46, border:`1.5px solid rgba(255,255,255,${isLast ? 0.9 : w*0.6+0.1})`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:20, color:`rgba(255,255,255,${isLast ? 1 : w*0.5+0.25})`, background: isLast ? "rgba(255,255,255,0.06)" : "transparent" }}>{ch}</div>
            <div style={{ flex:1 }}>
              <div style={{ height:Math.max(3, w*12), background:"#0e0e0e", borderRadius:2, position:"relative" }}>
                <div style={{ position:"absolute", left:0, top:0, bottom:0, borderRadius:2, background: isLast ? "#fff" : `rgba(255,255,255,${w*0.7+0.1})`, width:`${w*100}%`, transition:"width 0.8s cubic-bezier(0.16,1,0.3,1)", boxShadow:`0 0 ${w*14}px rgba(255,255,255,${w*0.35})` }} />
              </div>
            </div>
            <div style={{ width:46, textAlign:"right", fontSize:13, color: isLast ? "#ddd" : "#555" }}>{(w*100).toFixed(0)}%</div>
          </div>
        );
      })}
    </div>
  );
}

function ProbBars({ probs, pickedChar }) {
  const top = [...probs].sort((a,b) => b.prob - a.prob).slice(0, 9);
  const maxP = top[0]?.prob || 1;
  return (
    <div style={{ opacity:0, animation:"fadeUp 0.4s ease both", width:"100%" }}>
      <div style={{ fontSize:13, color:"#ddd", letterSpacing:"0.15em", marginBottom:6 }}>NEXT CHARACTER PROBABILITIES</div>
      <div style={{ fontSize:11, color:"#888", marginBottom:24 }}>model deciding what character comes next</div>
      {top.map((item, i) => {
        const isPicked = pickedChar && item.char === pickedChar;
        const isTop = i === 0;
        const pct = (item.prob / maxP) * 100;
        return (
          <div key={`${item.char}-${i}`} style={{ display:"flex", alignItems:"center", gap:14, marginBottom:10, opacity:0, animation:`fadeUp 0.3s ease ${i*45}ms both` }}>
            <div style={{ width:32, height:32, border:`1px solid ${isPicked ? "#fff" : isTop ? "#444" : "#1e1e1e"}`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:16, color: isPicked ? "#fff" : isTop ? "#ccc" : "#555", background: isPicked ? "rgba(255,255,255,0.1)" : "transparent", transition:"all 0.3s ease" }}>{item.char}</div>
            <div style={{ flex:1, height:8, background:"#0c0c0c", borderRadius:1, overflow:"hidden", border:"1px solid #161616" }}>
              <div style={{ height:"100%", width:`${pct}%`, background: isPicked ? "#fff" : isTop ? "#505050" : "#252525", borderRadius:1, transition:"width 0.6s cubic-bezier(0.16,1,0.3,1)", boxShadow: isPicked ? "0 0 14px rgba(255,255,255,0.5)" : "none", animation:`barGrow 0.5s ease ${i*45+180}ms both` }} />
            </div>
            <div style={{ width:52, textAlign:"right", fontSize:13, color: isPicked ? "#fff" : isTop ? "#aaa" : "#404040" }}>{item.prob.toFixed(1)}%</div>
          </div>
        );
      })}
    </div>
  );
}

function GeneratingViz({ word, currentChar, probs }) {
  return (
    <div style={{ opacity:0, animation:"fadeIn 0.4s ease both", width:"100%" }}>
      <div style={{ fontSize:13, color:"#ddd", letterSpacing:"0.15em", marginBottom:6 }}>AUTOREGRESSIVE GENERATION</div>
      <div style={{ fontSize:11, color:"#888", marginBottom:28 }}>each predicted character feeds back as the next input</div>
      <div style={{ display:"flex", gap:6, flexWrap:"wrap", marginBottom:32, minHeight:54 }}>
        {word.split("").map((ch, i) => (
          <div key={i} style={{ width:46, height:46, border:`1.5px solid ${i === word.length-1 ? "#ddd" : "#222"}`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:20, color: i === word.length-1 ? "#fff" : "#777", background: i === word.length-1 ? "rgba(255,255,255,0.05)" : "transparent", animation:`charAppear 0.3s ease both`, boxShadow: i === word.length-1 ? "0 0 12px rgba(255,255,255,0.15)" : "none" }}>{ch}</div>
        ))}
        <div style={{ width:46, height:46, border:"1.5px dashed #444", display:"flex", alignItems:"center", justifyContent:"center", fontSize:20, color:"#555", animation:"blink 0.9s ease infinite" }}>_</div>
      </div>
      {probs && probs.length > 0 && (
        <div>
          <div style={{ fontSize:11, color:"#888", marginBottom:16, letterSpacing:"0.12em" }}>→ selecting next character:</div>
          <ProbBars probs={probs} pickedChar={currentChar} />
        </div>
      )}
    </div>
  );
}

function ResultDisplay({ result, onReset }) {
  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:28, paddingTop:24, opacity:0, animation:"fadeIn 0.6s ease both" }}>
      <div style={{ fontSize:10, color:"#999", letterSpacing:"0.35em" }}>GENERATED OUTPUT</div>
      <div style={{ fontSize:68, fontFamily:"'Bebas Neue', sans-serif", letterSpacing:"0.25em", color:"#fff", textAlign:"center", animation:"resultGlow 1.2s cubic-bezier(0.16,1,0.3,1) both", textShadow:"0 0 40px rgba(255,255,255,0.18)" }}>{result}</div>
      <div style={{ display:"flex", gap:20, alignItems:"center", opacity:0, animation:"fadeIn 0.5s ease 1s both" }}>
        <div style={{ width:40, height:1, background:"#1e1e1e" }} />
        <div style={{ fontSize:11, color:"#999", letterSpacing:"0.2em" }}>✓ generation complete</div>
        <div style={{ width:40, height:1, background:"#1e1e1e" }} />
      </div>
      <button onClick={onReset} style={{ marginTop:8, background:"transparent", border:"1px solid #444", color:"#999", fontFamily:"'Share Tech Mono', monospace", fontSize:11, letterSpacing:"0.25em", padding:"12px 32px", cursor:"pointer", transition:"all 0.3s ease", opacity:0, animation:"fadeIn 0.5s ease 1.4s both" }}
        onMouseEnter={e => { e.target.style.borderColor="#fff"; e.target.style.color="#fff"; e.target.style.background="rgba(255,255,255,0.04)"; }}
        onMouseLeave={e => { e.target.style.borderColor="#444"; e.target.style.color="#999"; e.target.style.background="transparent"; }}
      >[ GENERATE AGAIN ]</button>
    </div>
  );
}

const STAGE_NAMES = ["TOKENIZE","EMBED","ATTENTION","PROBABILITIES","GENERATING","RESULT"];
const STAGE_DESC  = [
  "char → unique integer ID",
  "ID → 16-dim learned vector",
  "context weighting via QKV",
  "softmax over full vocab",
  "autoregressive char loop",
  "final generated output",
];

function StageList({ stage }) {
  const [hovered, setHovered] = useState(null);
  return (
    <div style={{ marginTop:40 }}>
      <div style={{ fontSize:10, color:"#777", letterSpacing:"0.25em", marginBottom:14 }}>PIPELINE</div>
      {STAGE_NAMES.map((s, i) => {
        const sid = i + 1;
        const active = stage === sid;
        const done   = stage > sid;
        const isHov  = hovered === i;
        return (
          <div
            key={s}
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            style={{
              display:"flex", alignItems:"center", gap:12,
              padding: isHov ? "9px 10px 9px 8px" : "8px 0",
              borderBottom:"1px solid #0e0e0e",
              borderRadius: isHov ? 3 : 0,
              background: isHov
                ? active ? "rgba(255,255,255,0.055)" : "rgba(255,255,255,0.025)"
                : "transparent",
              transform: isHov
                ? "scale(1.045) translateX(9px)"
                : active
                  ? "translateX(4px)"
                  : "translateX(0)",
              transformOrigin: "left center",
              transition:"transform 0.28s cubic-bezier(0.34,1.56,0.64,1), background 0.22s ease, padding 0.22s ease",
              cursor:"default", overflow:"hidden",
            }}
          >
            {/* left accent bar */}
            <div style={{
              width: active ? 3 : 2,
              height: active ? 30 : isHov ? 24 : 18,
              flexShrink:0,
              background: active ? "#fff" : done ? (isHov ? "#555" : "#333") : isHov ? "#2e2e2e" : "#111",
              boxShadow: active
                ? "0 0 12px rgba(255,255,255,0.65)"
                : isHov ? "0 0 6px rgba(255,255,255,0.18)" : "none",
              transition:"all 0.28s cubic-bezier(0.34,1.56,0.64,1)",
            }} />

            {/* label + sub-description */}
            <div style={{ flex:1, minWidth:0 }}>
              <div style={{
                fontSize:11,
                color: active ? "#fff" : done ? (isHov ? "#888" : "#555") : isHov ? "#ccc" : "#3e3e3e",
                letterSpacing:"0.1em",
                transition:"color 0.2s ease",
              }}>
                {String(i+1).padStart(2,"0")} · {s}
              </div>
              {/* sub-label slides down on hover */}
              <div style={{
                fontSize:10, color:"#606060", letterSpacing:"0.06em",
                maxHeight: isHov ? 18 : 0,
                opacity: isHov ? 1 : 0,
                marginTop: isHov ? 3 : 0,
                overflow:"hidden",
                transition:"max-height 0.25s ease, opacity 0.22s ease, margin-top 0.22s ease",
                whiteSpace:"nowrap",
              }}>
                {STAGE_DESC[i]}
              </div>
            </div>

            {/* right status icon */}
            <div style={{ flexShrink:0, marginLeft:"auto" }}>
              {done && (
                <div style={{
                  fontSize:10,
                  color: isHov ? "#666" : "#333",
                  transform: isHov ? "scale(1.2)" : "scale(1)",
                  transition: "color 0.2s ease, transform 0.25s cubic-bezier(0.34,1.56,0.64,1)",
                }}>✓</div>
              )}
              {active && (
                <div style={{
                  fontSize:10, color:"#888",
                  animation:"pulse 1.4s ease infinite",
                  transform: isHov ? "scale(1.3)" : "scale(1)",
                  transition:"transform 0.25s cubic-bezier(0.34,1.56,0.64,1)",
                }}>◆</div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [prefix, setPrefix] = useState('');
  const [temperature, setTemperature] = useState(0.5);
  const [stage, setStage] = useState(0);
  const [tokens, setTokens] = useState([]);
  const [embeddings, setEmbeddings] = useState([]);
  const [attentionWeights, setAttentionWeights] = useState([]);
  const [probs, setProbs] = useState([]);
  const [generatingWord, setGeneratingWord] = useState('');
  const [currentChar, setCurrentChar] = useState('');
  const [result, setResult] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [vocabData, setVocabData] = useState(null);
  const [liveProbs, setLiveProbs] = useState([]);
  const abortRef = useRef(false);

  useEffect(() => {
    fetch(`${API_BASE}/vocab`).then(r => r.json()).then(setVocabData).catch(() => {});
  }, []);

  async function run() {
    if (isRunning) return;
    abortRef.current = false;
    setIsRunning(true);
    setResult('');

    const inputPrefix = prefix.toLowerCase().replace(/[^a-z]/g, '');

    // Stage 1 — Tokenize
    setStage(1);
    const tokenData = [];
    for (const ch of inputPrefix) {
      const idx = vocabData?.unique_chars?.indexOf(ch) ?? -1;
      tokenData.push({ char: ch, id: idx });
      setTokens([...tokenData]);
      await sleep(210);
    }
    await sleep(700);

    // Stage 2 — Embed
    setStage(2);
    setEmbeddings(tokenData.map(t => ({ ...t, embedding: mockEmbedding(t.id >= 0 ? t.id : 0) })));
    await sleep(1300);

    // Stage 3 — Attention
    if (tokenData.length > 1) {
      setStage(3);
      setAttentionWeights(mockAttention(tokenData.map(t => t.char)));
      await sleep(1500);
    }

    // Stage 4/5 — Stream
    setStage(4);
    let genWord = inputPrefix;
    setGeneratingWord(genWord);

    try {
      const url = `${API_BASE}/generate/stream?prefix=${encodeURIComponent(inputPrefix)}&temperature=${temperature}`;
      const response = await fetch(url);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done || abortRef.current) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const msg = JSON.parse(line.slice(6).trim());
            if (msg.type === 'probs') { setProbs(msg.probs); setLiveProbs(msg.probs); setStage(4); await sleep(650); }
            if (msg.type === 'char') {
              setStage(5);
              setCurrentChar(msg.char);
              genWord = msg.word;
              setGeneratingWord(genWord);
              await sleep(550);
            }
            if (msg.type === 'done') { setResult(msg.result || genWord); setStage(6); }
          } catch {}
        }
      }
    } catch {
      if (genWord) { setResult(genWord); setStage(6); }
    }
    setIsRunning(false);
  }

  function reset() {
    abortRef.current = true;
    setStage(0); setTokens([]); setEmbeddings([]);
    setAttentionWeights([]); setProbs([]); setLiveProbs([]);
    setGeneratingWord(''); setCurrentChar('');
    setResult(''); setIsRunning(false);
  }

  return (
    <>
      <Scanline />
      <div style={{ display:'flex', flexDirection:'row', minHeight:'100vh', overflow:'hidden' }}>

        {/* ── LEFT PANEL (38%) ── */}
        <div style={{
          width:'38%', flexShrink:0,
          borderRight:'1px solid rgba(255,255,255,0.12)',
          display:'flex', flexDirection:'column',
          overflowY:'auto', padding:'40px 32px',
        }}>
          {/* Header */}
          <div style={{ marginBottom:32, opacity:0, animation:'fadeUp 0.6s ease both' }}>
            <div style={{ fontSize:10, letterSpacing:'0.35em', color:'#bbb', marginBottom:14 }}>
              MICROGPT / TRANSFORMER / VISUALIZER
            </div>
            <h1 style={{ fontFamily:"'Bebas Neue', sans-serif", fontSize:72, letterSpacing:'0.04em', lineHeight:0.9, color:'#fff' }}>
              micro<br /><span style={{ color:'#e6e6e6' }}>GPT</span>
            </h1>
            <div style={{ fontSize:11, color:'#ddd', marginTop:14, letterSpacing:'0.1em', lineHeight:2 }}>
              150 lines · no pytorch · just math<br />
              <span style={{ color:'#ccc' }}>watch the transformer think in real time</span>
            </div>

            <div style={{ marginTop:12, display:'flex', gap:10, alignItems:'center' }}>
              <div style={{ fontSize:11, color:'#fff', background:'#111', border:'1px solid #222', padding:'6px 10px', borderRadius:6, boxShadow:'0 4px 14px rgba(0,0,0,0.6)' }}>
                @andrej_Karpathy
              </div>
              <div style={{ fontSize:11, color:'#bbb' }}>for code & learning</div>
            </div>
          </div>

          {/* Input */}
          <div style={{ opacity:0, animation:'fadeUp 0.5s ease 200ms both' }}>
            <div style={{ fontSize:11, color:'#bbb', letterSpacing:'0.2em', marginBottom:12 }}>ENTER A PREFIX</div>
            <div style={{ display:'flex', gap:0, marginBottom:28, borderBottom:'1px solid #333' }}>
              <input
                value={prefix}
                onChange={e => setPrefix(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && run()}
                placeholder="snap, zep, cred..."
                maxLength={8}
                style={{
                  flex:1, background:'transparent', border:'none',
                  color:'#fff', fontFamily:"'Share Tech Mono', monospace",
                  fontSize:28, padding:'10px 0', outline:'none', letterSpacing:'0.08em',
                }}
              />
              <button onClick={run} disabled={isRunning} style={{
                background: isRunning ? '#222' : '#fff',
                color: isRunning ? '#555' : '#000',
                border:'none', fontFamily:"'Share Tech Mono', monospace",
                fontSize:11, letterSpacing:'0.25em',
                padding:'0 24px', cursor: isRunning ? 'default' : 'pointer',
                transition:'all 0.2s ease', flexShrink:0,
              }}
                onMouseEnter={e => { if (!isRunning) e.target.style.background='#ddd'; }}
                onMouseLeave={e => { if (!isRunning) e.target.style.background='#fff'; }}
              >{isRunning ? '...' : 'RUN →'}</button>
            </div>

            {/* Temperature */}
            <div style={{ marginBottom:28 }}>
              <div style={{ display:'flex', justifyContent:'space-between', fontSize:11, color:'#bbb', letterSpacing:'0.15em', marginBottom:12 }}>
                <span>TEMPERATURE</span>
                <span style={{ color:'#ddd' }}>{temperature.toFixed(1)}</span>
              </div>
              <input type="range" min="0.1" max="1.5" step="0.1"
                value={temperature}
                onChange={e => setTemperature(parseFloat(e.target.value))}
                style={{ width:'100%', accentColor:'#fff', cursor:'pointer' }}
              />
              <div style={{ display:'flex', justifyContent:'space-between', fontSize:10, color:'#bbb', marginTop:8 }}>
                <span>conservative</span><span>creative chaos</span>
              </div>
            </div>

            {/* Quick picks */}
            <div>
              <div style={{ fontSize:10, color:'#bbb', letterSpacing:'0.2em', marginBottom:12 }}>QUICK START</div>
              <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
                {['snap','zep','cred','swift','nova','orbit',''].map(p => (
                  <button key={p||'rnd'} onClick={() => setPrefix(p)} style={{
                    background:'transparent', border:'1px solid #333',
                    color:'#bbb', fontFamily:"'Share Tech Mono', monospace",
                    fontSize:11, padding:'7px 14px', cursor:'pointer',
                    transition:'all 0.25s ease', letterSpacing:'0.1em',
                  }}
                    onMouseEnter={e => { e.target.style.borderColor='#bbb'; e.target.style.color='#fff'; }}
                    onMouseLeave={e => { e.target.style.borderColor='#333'; e.target.style.color='#bbb'; }}
                  >{p || 'random'}</button>
                ))}
              </div>
            </div>
          </div>

          {/* Vertical stage list */}
          <StageList stage={stage} />

          {/* Footer */}
          <div style={{ marginTop:'auto', paddingTop:32, borderTop:'1px solid #222', fontSize:10, color:'#666', letterSpacing:'0.1em' }}>
            <div style={{ marginBottom:6 }}>microGPT · pure python</div>
            <div>{vocabData ? `vocab ${vocabData.vocab_size} · params 4160` : 'connecting...'}</div>
          </div>
        </div>

        {/* ── RIGHT PANEL (62%) — animations only ── */}
        <div style={{
          flex:1,
          display:'flex', alignItems:'center', justifyContent:'center',
          padding:'48px 40px', overflow:'hidden',
        }}>

          {stage === 0 && (
            <div style={{ textAlign:'center', opacity:0, animation:'fadeIn 0.8s ease 0.4s both' }}>
              <div style={{ width:6, height:6, borderRadius:'50%', background:'#444', margin:'0 auto 16px', animation:'pulse 2.5s ease infinite' }} />
              <div style={{ fontSize:12, color:'#555', letterSpacing:'0.2em' }}>run the model to see it think</div>
            </div>
          )}

          {stage === 1 && (
            <div style={{ width:'100%', maxWidth:560 }}>
              <div style={{ fontSize:13, color:'#ddd', letterSpacing:'0.15em', marginBottom:6 }}>TOKENIZATION</div>
              <div style={{ fontSize:11, color:'#888', marginBottom:28 }}>each character → unique integer ID</div>
              <div style={{ display:'flex', gap:16, flexWrap:'wrap' }}>
                {tokens.map((t, i) => <TokenBox key={i} char={t.char} tokenId={t.id} delay={i*170} showArrow={false} showId={false} lit />)}
              </div>
            </div>
          )}

          {stage === 2 && (
            <div style={{ width:'100%', maxWidth:620 }}>
              <div style={{ fontSize:13, color:'#ddd', letterSpacing:'0.15em', marginBottom:6 }}>EMBEDDING</div>
              <div style={{ fontSize:11, color:'#888', marginBottom:28 }}>token ID → 16-dim vector · brightness = magnitude</div>
              <div style={{ display:'flex', gap:14, marginBottom:32, flexWrap:'wrap' }}>
                {tokens.map((t, i) => <TokenBox key={i} char={t.char} tokenId={t.id} delay={i*110} showArrow showId lit={false} />)}
              </div>
              {embeddings.map((e, i) => <EmbeddingRow key={i} char={e.char} tokenId={e.id} embedding={e.embedding} delay={i*130} />)}
            </div>
          )}

          {stage === 3 && tokens.length > 1 && (
            <div style={{ width:'100%', maxWidth:520 }}>
              <AttentionViz chars={tokens.map(t => t.char)} weights={attentionWeights} />
            </div>
          )}

          {stage === 4 && probs.length > 0 && (
            <div style={{ width:'100%', maxWidth:480 }}>
              <ProbBars probs={probs} pickedChar={null} />
            </div>
          )}

          {stage === 5 && (
            <div style={{ width:'100%', maxWidth:560 }}>
              <GeneratingViz word={generatingWord} currentChar={currentChar} probs={liveProbs} />
            </div>
          )}

          {stage === 6 && <ResultDisplay result={result} onReset={reset} />}
        </div>

      </div>
    </>
  );
}
