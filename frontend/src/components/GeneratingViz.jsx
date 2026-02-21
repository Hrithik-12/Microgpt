import React from 'react'

export default function GeneratingViz({word, currentChar, liveProbs, highlight}){
  return (
    <div style={{animation:'fadeIn 1.2s ease'}}>
      <div style={{fontSize:11, color:'#888', marginBottom:12}}>AUTOREGRESSIVE GENERATION</div>
      <div style={{display:'flex', gap:8, flexWrap:'wrap', marginBottom:16}}>
        {word.split('').map((ch,i)=> (
          <div key={i} style={{width:44, height:44, border:'1px solid #444', display:'flex', alignItems:'center', justifyContent:'center', fontSize:20, background: i===word.length-1? '#111':'#0a0a0a'}}>{ch}</div>
        ))}
        <div style={{width:44, height:44, border:'1px dashed #555', display:'flex', alignItems:'center', justifyContent:'center', color:'#666', fontSize: 20}}>_</div>
      </div>
      {currentChar && <div style={{fontSize:13, color:'#aaa', marginBottom: 16}}>→ picked <span style={{color:'#fff', fontSize: 16}}>'{currentChar}'</span></div>}
      {liveProbs && <div style={{marginTop:12, display: 'flex', flexDirection: 'column', gap: 8}}>
        {/* small prob bars */}
        {liveProbs.slice(0,6).map(p=> (
          <div key={p.char} style={{display:'flex', alignItems:'center', gap:12}}>
            <div style={{width:24, color:'#ddd', fontSize: 16}}>{p.char}</div>
            <div style={{flex:1, height:8, background:'#1a1a1a', borderRadius: 4}}>
              <div style={{height:'100%', background: highlight===p.char? '#fff':'#777', width:`${p.prob}%`, transition:'width 1.0s ease', borderRadius: 4}} />
            </div>
            <div style={{width:56, textAlign:'right', color:'#aaa', fontSize: 13}}>{p.prob.toFixed(1)}%</div>
          </div>
        ))}
      </div>}
    </div>
  )
}
