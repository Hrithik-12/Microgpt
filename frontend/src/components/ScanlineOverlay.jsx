import React from 'react'

export default function ScanlineOverlay(){
  return (
    <div style={{position:'fixed', inset:0, pointerEvents:'none', zIndex:9999, overflow:'hidden'}}>
      <div style={{position:'absolute', inset:0, background:"repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.06) 2px, rgba(0,0,0,0.06) 4px)", pointerEvents:'none'}} />
    </div>
  )
}
