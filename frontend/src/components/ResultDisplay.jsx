import React from 'react'

export default function ResultDisplay({result, onReset}){
  return (
    <div style={{display:'flex', flexDirection:'column', alignItems:'flex-start', gap:24, animation: 'fadeIn 1s ease'}}>
      <div style={{fontSize:12, color:'#888'}}>GENERATED RESULT</div>
      <div style={{fontSize:56, fontFamily:"'Bebas Neue', sans-serif", letterSpacing:'0.1em', color:'#fff', textShadow:'0 0 24px rgba(255,255,255,0.2)', lineHeight: 1.1}}>{result}</div>
      <button onClick={onReset} style={{background:'transparent', border:'1px solid #444', color:'#aaa', padding:'12px 24px', fontSize: 14, cursor:'pointer', transition: 'all 0.2s'}} onMouseOver={e=>{e.target.style.borderColor='#fff'; e.target.style.color='#fff'}} onMouseOut={e=>{e.target.style.borderColor='#444'; e.target.style.color='#aaa'}}>GENERATE AGAIN</button>
    </div>
  )
}
