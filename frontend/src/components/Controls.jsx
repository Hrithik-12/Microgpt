import React from 'react'

export default function Controls({prefix, setPrefix, temperature, setTemperature, run, examples}){
  return (
    <div style={{display: 'flex', flexDirection: 'column', gap: 24}}>
      <div>
        <div style={{fontSize:11, color:'#888', marginBottom:12}}>ENTER PREFIX</div>
        <div style={{display:'flex', gap:12, alignItems: 'center'}}>
            <input value={prefix} onChange={e=>setPrefix(e.target.value)} placeholder='snap, zep...' maxLength={8} onKeyDown={e=> e.key==='Enter' && run()} style={{flex:1, minWidth: 0, background:'transparent', border:'none', borderBottom:'2px solid #333', color:'#e8e8e8', fontFamily:"'Share Tech Mono', monospace", fontSize:28, padding:'4px 0', outline:'none', position:'relative', zIndex:1}} />
            <button onClick={run} style={{background:'#fff', color:'#000', border:'none', padding:'8px 16px', fontSize: 14, cursor:'pointer', position:'relative', zIndex:2, fontWeight: 'bold', flexShrink: 0}}>RUN</button>
        </div>
      </div>
      <div>
        <div style={{fontSize:11, color:'#888', marginBottom:12}}>TEMPERATURE: {temperature.toFixed(1)}</div>
        <input type='range' min='0.1' max='1.5' step='0.1' value={temperature} onChange={e=>setTemperature(parseFloat(e.target.value))} style={{width:'100%', cursor: 'pointer'}} />
      </div>
      <div>
        <div style={{fontSize:11, color:'#888', marginBottom:12}}>TRY THESE</div>
        <div style={{display:'flex', gap:12, flexWrap:'wrap'}}>
          {examples.map(p=> (
            <button key={p} onClick={()=>setPrefix(p)} style={{background:'transparent', border:'1px solid #333', color:'#ccc', fontFamily:"'Share Tech Mono', monospace", padding:'8px 16px', fontSize: 14, cursor:'pointer', transition: 'all 0.2s'}} onMouseOver={e=>e.target.style.borderColor='#666'} onMouseOut={e=>e.target.style.borderColor='#333'}>{p||'random'}</button>
          ))}
        </div>
      </div>
    </div>
  )
}
