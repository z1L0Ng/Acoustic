import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure2_patient_alignment_20260923';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/current.pptx'));
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'),{captureSpacesBetweenElements:true});
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])if(c.name!=='mc:Fallback')a.push(...all(c,k));return a};
const target=all(d,'p:sp').find(s=>all(s,'p:cNvPr')[0]?.attributes.id==='238');
if(!target)throw Error('Expected annotation is missing');
const ts=all(target,'a:t');
const before=ts.map(t=>(t.elements??[]).filter(e=>e.type==='text').map(e=>e.text).join('')).join('');
if(before!=='Attention regularization')throw Error('Unexpected annotation: '+before);
for(const t of ts)for(const e of t.elements??[])if(e.type==='text'){if(e.text==='Attention')e.text='Patient';if(e.text==='regularization')e.text='alignment';}
all(target,'p:cNvPr')[0].attributes.name='Bottom aligned loss annotation Patient alignment';
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
await fs.writeFile(root+'/output/figure2_patient_alignment.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log('Updated only the bottom GPAL annotation to Patient alignment.');
