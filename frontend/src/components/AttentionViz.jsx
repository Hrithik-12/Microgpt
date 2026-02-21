import React from 'react'

export default function AttentionViz({chars, weights}){
  const maxW = Math.max(...weights, 1)
  return (
    <div style={{display:'flex', flexDirection:'column', gap:12, animation:'fadeIn 1.2s ease both'}}>
      <div style={{fontSize:11, color:'#888', marginBottom:4}}>ATTENTION FROM "{chars[chars.length-1]?.toUpperCase()}"</div>
      {chars.map((ch,i)=>{
        const w = weights[i]/maxW
        return (
          <div key={i} style={{display:'flex', alignItems:'center', gap:12}}>
            <div style={{width:36, height:36, border:`1px solid rgba(232,232,232,${w*0.9+0.1})`, display:'flex', alignItems:'center', justifyContent:'center', fontSize: 16, color:`rgba(232,232,232,${w*0.8+0.2})`, background: '#0a0a0a'}}>{ch}</div>
            <div style={{flex:1, height: Math.max(6, w*16), background:'#1a1a1a', position:'relative', borderRadius:4}}>
              <div style={{position:'absolute', left:0, top:0, height:'100%', background:`rgba(232,232,232,${w})`, width:`${w*100}%`, transition:'width 1.0s cubic-bezier(.2,.8,.2,1)', borderRadius:4}} />
            </div>
            <div style={{fontSize:12, color:'#aaa', width:40, textAlign:'right'}}>{(w*100).toFixed(0)}%</div>
          </div>
        )
      })}
    </div>
  )
}
