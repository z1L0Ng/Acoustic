import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure2_cw_middle_20260923';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/current.pptx'));
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'));
const el=(name,attributes={},elements=[])=>({type:'element',name,attributes,elements});
const find=(n,k)=>n.elements?.find(x=>x.name===k);
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])if(c.name!=='mc:Fallback')a.push(...all(c,k));return a};
const tree=all(d,'p:spTree')[0];
const shapes=new Map();for(const s of [...all(d,'p:sp'),...all(d,'p:cxnSp')]){const nv=all(s,'p:cNvPr')[0];if(nv)shapes.set(+nv.attributes.id,s)}
const get=i=>shapes.get(i),emu=v=>String(Math.round(v*12700));let nextId=300;
function place(s,x,y,w,h){const xf=all(s,'a:xfrm')[0];find(xf,'a:off').attributes={x:emu(x),y:emu(y)};find(xf,'a:ext').attributes={cx:emu(w),cy:emu(h)}}
function remove(id){const s=get(id);function walk(n){if(!n.elements)return;n.elements=n.elements.filter(x=>x!==s);n.elements.forEach(walk)}walk(tree)}
function setText(s,text){const ts=all(s,'a:t');ts[0].elements=[{type:'text',text}];for(const t of ts.slice(1))t.elements=[]}
function line(name,points,color='000000',dash=false,arrow=true){const s=structuredClone(get(216));const nv=all(s,'p:cNvPr')[0];nv.attributes={id:String(nextId++),name};nv.elements=[];const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x||.001,h=Math.max(...ys)-y||.001;place(s,x,y,w,h);const sp=find(s,'p:spPr');find(find(sp,'a:custGeom'),'a:pathLst').elements=[el('a:path',{w:emu(w),h:emu(h),fill:'none',stroke:'1'},points.map((p,i)=>el(i?'a:lnTo':'a:moveTo',{},[el('a:pt',{x:emu(p[0]-x),y:emu(p[1]-y)})])))];const ln=find(sp,'a:ln');ln.attributes={w:emu(dash?1.3:1.7)};ln.elements=[el('a:solidFill',{},[el('a:srgbClr',{val:color})]),el('a:prstDash',{val:dash?'dash':'solid'}),el('a:round')];if(arrow)ln.elements.push(el('a:tailEnd',{type:'triangle',w:'sm',len:'sm'}));tree.elements.splice(2,0,s);return s}
// Update visible wording, native choice, and head order without touching outputs or objective terms.
for(const t of all(d,'a:t'))for(const v of t.elements??[])if(v.type==='text')v.text=v.text.replaceAll('C/W','C-W');
for(const nv of all(d,'p:cNvPr'))nv.attributes.name=nv.attributes.name.replaceAll('C/W','C-W');
setText(get(23),'Task heads');
setText(get(13),'ICBHI native head');place(get(13),594,106,196,44);
setText(get(14),'SPRSound native head');place(get(14),594,266,196,44);
setText(get(134),'Shared C-W heads');place(get(134),584,164,216,24);
place(get(28),594,200,94,34);place(get(29),702,200,88,34);
place(get(135),584,194,216,46);remove(85);
setText(get(157),'(ICBHI or SPRSound) + C-W');
for(const id of [13,14])for(const rp of all(get(id),'a:rPr'))rp.attributes.sz='1400';
for(const id of [204,206,207,208,213,214,215,216,217,220])remove(id);
line('Shared h native head trunk',[[566,128],[566,288]],'000000',false,false);
line('Shared h to middle C-W branch',[[566,192],[746,192]],'000000',false,false);
line('Shared h to Crackle',[[641,192],[641,200]]);
line('Shared h to Wheeze',[[746,192],[746,200]]);
line('Shared h to lower SPRSound native head',[[566,288],[594,288]]);
line('SPRSound native output',[[790,288],[872,288]]);
// Dataset-specific native supervision: only the corresponding native head contributes.
line('ICBHI native supervision upper',[[790,148],[836,148],[836,282]],'32631C',true,false);
line('ICBHI native supervision middle',[[836,294],[836,320]],'32631C',true,false);
line('ICBHI native supervision to classification',[[836,332],[836,368],[800,368]],'32631C',true);
line('SPRSound native supervision to classification',[[692,310],[692,338]],'32631C',true);
// Both shared heads meet before branching to CLS and optional HF.
line('Crackle to shared C-W supervision',[[641,234],[641,248],[804,248],[804,282]],'32631C',true,false);
line('Wheeze to shared C-W supervision',[[746,234],[746,248]],'32631C',true,false);
line('Shared C-W supervision to classification',[[804,294],[804,350],[800,350]],'32631C',true);
line('Optional HF supervision upper',[[776,248],[824,248],[824,282]],'B6993A',true,false);
line('Optional HF supervision to loss',[[824,294],[824,326],[996,326],[996,338]],'B6993A',true);
for(const [id,x,color] of [[221,746,'32631C'],[222,776,'B6993A']]){const s=get(id);place(s,x-1.5,246.5,3,3);for(const f of all(s,'a:solidFill'))f.elements=[el('a:srgbClr',{val:color})]}
// Preserve the existing four bottom-aligned explanatory arrows and all formulas.
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
for(const path of Object.keys(z.files).filter(n=>/^ppt\/notesSlides\/notesSlide\d+\.xml$/.test(n))){const notes=await z.file(path).async('string');z.file(path,notes.replaceAll('C/W','C-W'));}
await fs.writeFile(root+'/output/figure2_method_cw_middle.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log('Updated discussion draft: ICBHI / shared C-W / SPRSound; explicit native-head choice in CLS.');
