
const batch='urti-audio-review-20260921-v1';let index=0,answers={};
const $=id=>document.getElementById(id);const radios=[...document.querySelectorAll('[name=rating]')];
try{answers=JSON.parse(localStorage.getItem(batch)||'{}')}catch(e){$('status').textContent='Local saving unavailable; download your reviews regularly.'}
function id(){return 'R'+String(index+1).padStart(3,'0')}
function save(){const r=radios.find(x=>x.checked);if(r)answers[id()]={rating:r.value,notes:$('notes').value};try{localStorage.setItem(batch,JSON.stringify(answers))}catch(e){}update()}
function update(){$('progress').value=Object.keys(answers).length;$('counter').textContent=Object.keys(answers).length+' of 90 reviewed · Clip '+(index+1)+' of 90'}
function show(){$('audio').pause();$('audio').src='audio/'+id()+'.wav';$('clip').textContent='Recording '+id();radios.forEach(r=>r.checked=r.value===answers[id()]?.rating);$('notes').value=answers[id()]?.notes||'';$('prev').disabled=index===0;$('next').disabled=index===89;update()}
radios.forEach(r=>r.addEventListener('change',save));$('notes').addEventListener('input',save);
$('prev').onclick=()=>{save();index--;show()};$('next').onclick=()=>{save();index++;show()};
$('audio').onerror=()=>{$('status').textContent='Audio could not load. Keep the audio folder beside this page.'};
$('export').onclick=()=>{save();const data={batch,exported_at:new Date().toISOString(),reviews:answers};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='urti-audio-reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);$('status').textContent='Review file downloaded. Unreviewed clips remain unreviewed.'};show();
