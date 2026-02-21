import React from 'react'
import TokenBox from './TokenBox'
import EmbeddingRow from './EmbeddingRow'
import AttentionViz from './AttentionViz'
import ProbBars from './ProbBars'
import GeneratingViz from './GeneratingViz'

export default function Animations({stage, tokens, embeddings, attentionWeights, probs, generatingWord, currentChar, liveProbs, highlight}){
  return (
    <div style={{display:'flex', flexDirection:'column', gap:24, width: '100%', height: '100%', justifyContent: 'center'}}>
      {stage===1 && (
        <div>
          <div style={{fontSize:11, color:'#888', marginBottom:16}}>BREAKING INPUT INTO TOKENS</div>
          <div style={{display:'flex', gap:20, flexWrap:'wrap'}}>{tokens.map((t,i)=> <TokenBox key={i} {...t} delay={i*180} />)}</div>
        </div>
      )}

      {stage===2 && (
        <div>
          <div style={{fontSize:11, color:'#888', marginBottom:16}}>TOKEN → EMBEDDING VECTOR</div>
          <div style={{display:'flex', gap:20, marginBottom:24, flexWrap:'wrap'}}>{tokens.map((t,i)=> <TokenBox key={i} {...t} delay={i*100} showArrow showNumber />)}</div>
          <div style={{display: 'flex', flexDirection: 'column', gap: 12}}>{embeddings.map((e,i)=> <EmbeddingRow key={i} {...e} delay={i*120} />)}</div>
        </div>
      )}

      {stage===3 && tokens.length>1 && <AttentionViz chars={tokens.map(t=>t.char)} weights={attentionWeights} />}

      {stage===4 && probs.length>0 && <ProbBars probs={probs} />}

      {stage===5 && <GeneratingViz word={generatingWord} currentChar={currentChar} liveProbs={liveProbs} highlight={highlight} />}

      {stage===6 && <div style={{display:'flex', justifyContent:'center'}}><div style={{fontSize:18,color:'#e8e8e8'}}>Result ready</div></div>}
    </div>
  )
}
