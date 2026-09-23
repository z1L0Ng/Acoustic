import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure1_b_consolas_20260922';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/original.pptx'));
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'));
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])a.push(...all(c,k));return a};
const labels={Normal:'Normal (N)',Crackle:'Crackle (C)',Wheeze:'Wheeze (W)',Both:'Both (C-W)'};
let changed=0;
for(const s of all(d,'p:sp')){
 const name=all(s,'p:cNvPr')[0]?.attributes?.name;
 if(!name?.startsWith('b/class-'))continue;
 const text=labels[name.slice(8)];if(!text)throw Error(name);
 const ts=all(s,'a:t');ts[0].elements=[{type:'text',text}];for(const t of ts.slice(1))t.elements=[];
 for(const tag of ['a:latin','a:ea','a:cs'])for(const n of all(s,tag))n.attributes.typeface='Consolas';
 changed++;
}
if(changed!==4)throw Error('Expected 4 labels');
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
await fs.writeFile(root+'/output/figure1_b_consolas.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log('Updated four category labels, font Consolas, original 8 pt retained.');
