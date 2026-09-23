import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure1_labels_swap_20260922';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/original.pptx'));
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'));
const el=(n,k)=>n.elements?.find(v=>v.name===k);
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])a.push(...all(c,k));return a};
const tree=all(d,'p:spTree')[0];
const sets={ICBHI:[4,107,29,20,4,88,29],SPRSound:[117,153,56,20,96,112,56],HF_Lung:[276,115,29,24,212,97,36],KAUH:[397,103.567,32,13,313,187.567,134]};
const labels={HF_Lung:{I:'Inhalation',E:'Exhalation',D:'Crackles'},KAUH:{N:'Normal',C:'Crackles',IC:'Inspiratory crackles',EW:'Expiratory wheezes',IEW:'Inspiratory and expiratory wheezes',ICEW:'Inspiratory crackles + expiratory wheezes',Crep:'Crepitations',Bronchial:'Bronchial sounds'}};
function setText(s,text){const ts=all(s,'a:t');ts[0].elements=[{type:'text',text}];for(const t of ts.slice(1))t.elements=[{type:'text',text:''}]}
const emu=v=>Math.round(v*12700);
for(const s of tree.elements??[]){
 const nv=all(s,'p:cNvPr')[0]; if(!nv)continue;
 const name=nv.attributes.name;
 const xf=all(s,'a:xfrm')[0];if(!xf)continue;
 const off=el(xf,'a:off').attributes, ext=el(xf,'a:ext').attributes;
 let x=+off.x/12700,w=+ext.cx/12700;
 if(name.startsWith('b/')){
  off.x=String(emu(x+414));nv.attributes.name=name.replace(/^b\//,'c/');
  if(name==='b/panel-b'){setText(s,'(c)');nv.attributes.name='c/panel-c';ext.cx=String(emu(16))}
 }else if(name.startsWith('c/')){
  off.x=String(emu(x-91));nv.attributes.name=name.replace(/^c\//,'b/');
  if(name==='c/panel-c'){setText(s,'(b)');off.x=String(emu(2));nv.attributes.name='b/panel-b'}
 }else if(name.startsWith('a/')){
  for(const [ds,a] of Object.entries(sets)){
   const [ox,ow,olw,cw,nx,nw,nlw]=a;
   if(!name.includes('-'+ds))continue;
   const obx=ox+olw+3,obw=ow-olw-cw-6,nbx=nx+nlw+3,nbw=nw-nlw-cw-6;
   if(name.startsWith('a/source-')||name.startsWith('a/legend-'))x+=(nx+nw/2)-(ox+ow/2);
   else if(name.startsWith('a/scope-')){x=nx;w=nw}
   else if(name.startsWith('a/label-')){x=nx;w=nlw;const key=name.slice(('a/label-'+ds+'-').length);if(labels[ds]?.[key])setText(s,labels[ds][key]);}
   else if(name.startsWith('a/count-'))x=nx+nw-cw;
   else if(name.startsWith('a/bar-')){x=nbx;w=w*nbw/obw}
   else if(name.startsWith('a/axis-')){x=nbx;w=nbw}
   else if(name.startsWith('a/tick-label-'))x=nbx+(x+9-obx)*nbw/obw-9;
   else if(name.startsWith('a/grid-')||name.startsWith('a/tick-'))x=nbx+(x-obx)*nbw/obw;
   off.x=String(emu(x));ext.cx=String(emu(w));
  }
 }
}
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
const notesPath='ppt/notesSlides/notesSlide1.xml';
const notes=xml.xml2js(await z.file(notesPath).async('string'));
for(const t of all(notes,'a:t'))for(const v of t.elements??[]){
 if(v.type==='text')v.text=v.text.replace('Figure 1(b), spectral PCA.','Figure 1(c), spectral PCA.').replace('Figure 1(c): spectral centroid by sound label.','Figure 1(b): spectral centroid by sound label.').replace('All native label strings are retained,','All native categories and counts are retained,');
}
const body=all(notes,'p:sp').find(s=>all(s,'p:ph').some(n=>n.attributes?.type==='body'));
const tb=el(body,'p:txBody');
for(const text of [
 'Label expansions: HF I = Inhalation; E = Exhalation; D = Crackles (the source describes all discontinuous adventitious sound labels as crackles). Source: https://arxiv.org/abs/2102.03049 .',
 'KAUH label expansions follow Data annotation.xlsx and the dataset paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC7937981/ . ICB remains the original workbook code because its B token is not explicitly defined in the source legend. The paper lists two Bronchial & Crackles cases; their relation to the two ICB cases requires confirmation.'
])tb.elements.push({type:'element',name:'a:p',elements:[{type:'element',name:'a:r',elements:[{type:'element',name:'a:t',elements:[{type:'text',text}]}]}]});
z.file(notesPath,xml.js2xml(notes));

await fs.writeFile(root+'/output/figure1_full_labels_swap.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log('PPTX updated: full labels, centroid (b), PCA (c). ICB awaits user choice.');
