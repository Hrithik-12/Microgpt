import React from 'react'

export default function TokenBox({char, tokenId, delay=0, showArrow=false, showNumber=false, active=false}){
  return (
    <div style={{display:'flex', flexDirection:'column', alignItems:'center', gap:6, animation:`fadeUp 1.2s ease both`, animationDelay:`${delay}ms`}}>
      <div style={{width:48, height:48, border: active? '1px solid #fff' : '1px solid #444', display:'flex', alignItems:'center', justifyContent:'center', fontSize:20, fontFamily:"'Share Tech Mono', monospace", background: active? '#111':'#0a0a0a', boxShadow: active? '0 0 16px rgba(255,255,255,0.1)':''}}>{char}</div>
      {showArrow && <div style={{color:'#666', marginTop:4}}>↓</div>}
      {showNumber && <div style={{fontSize:12, color:'#888', marginTop:2}}>{tokenId ?? '?'}</div>}
    </div>
  )
}
