import React from 'react'

export default function ProbBars({probs, highlight}){
  const top = [...probs].sort((a,b)=> b.prob - a.prob).slice(0,8)
  const maxP = top[0]?.prob||1
  return (
    <div style={{display:'flex', flexDirection:'column', gap:12}}>
      <div style={{fontSize:11, color:'#888', marginBottom:4}}>NEXT CHARACTER PROBABILITIES</div>
      {top.map((item,i)=> (
        <div key={item.char} style={{display:'flex', alignItems:'center', gap:12}}>
          <div style={{width:24, fontSize:16, textAlign:'center', color: highlight===item.char? '#fff':'#ddd'}}>{item.char}</div>
          <div style={{flex:1, height:8, background:'#1a1a1a', position:'relative', borderRadius:4}}>
            <div style={{position:'absolute', left:0, top:0, height:'100%', background: highlight===item.char? '#fff':'#777', width:`${(item.prob/maxP)*100}%`, transition:'width 1.2s cubic-bezier(.2,.8,.2,1)', borderRadius:4}} />
          </div>
          <div style={{width:48, textAlign:'right', color: highlight===item.char? '#fff':'#aaa', fontSize: 13}}>{item.prob.toFixed(1)}%</div>
        </div>
      ))}
    </div>
  )
}
