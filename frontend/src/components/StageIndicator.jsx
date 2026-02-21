import React from 'react'

const STAGES = [
  { id: 1, label: "01 · TOKENIZE" },
  { id: 2, label: "02 · EMBED" },
  { id: 3, label: "03 · ATTENTION" },
  { id: 4, label: "04 · PROBABILITIES" },
  { id: 5, label: "05 · GENERATING" },
  { id: 6, label: "06 · RESULT" },
]

export default function StageIndicator({stage}){
  return (
    <div style={{display:'flex', gap:12, alignItems:'center', marginBottom:24, flexWrap: 'wrap'}}>
      {STAGES.map(s=> (
        <div key={s.id} style={{fontSize:11, color: stage===s.id? '#fff' : '#666', borderBottom: stage===s.id? '2px solid #fff' : '2px solid transparent', paddingBottom:6, transition: 'all 0.3s ease'}}>{s.label}</div>
      ))}
    </div>
  )
}
