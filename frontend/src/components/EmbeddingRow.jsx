import React from 'react'

export default function EmbeddingRow({char, tokenId, embedding, delay=0}){
  return (
    <div style={{display:'flex', alignItems:'center', gap:16, animation:`fadeUp 1.2s ease both`, animationDelay:`${delay}ms`}}>
      <div style={{width:40, textAlign:'right', color:'#888', fontSize:14}}>{tokenId}</div>
      <div style={{color:'#555', fontSize:14}}>→</div>
      <div style={{display:'flex', gap:6}}>
        {embedding.map((v,i)=> (
          <div key={i} style={{width:16, height:16, background: v>0? `rgba(232,232,232,${Math.abs(v)*0.9})` : `rgba(232,232,232,${Math.abs(v)*0.2})`, border:'1px solid #333'}} title={v.toFixed(3)} />
        ))}
      </div>
      <div style={{color:'#666', fontSize:12, marginLeft:10}}>{char}</div>
    </div>
  )
}
