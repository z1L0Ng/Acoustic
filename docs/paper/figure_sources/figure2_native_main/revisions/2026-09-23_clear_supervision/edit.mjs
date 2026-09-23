import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure2_clear_supervision_20260923';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/current.pptx'));
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'));
const el=(name,attributes={},elements=[])=>({type:'element',name,attributes,elements});
const find=(n,k)=>n.elements?.find(x=>x.name===k);
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])if(c.name!=='mc:Fallback')a.push(...all(c,k));return a};
const tree=all(d,'p:spTree')[0],shapes=new Map();for(const s of [...all(d,'p:sp'),...all(d,'p:cxnSp')]){const nv=all(s,'p:cNvPr')[0];if(nv)shapes.set(+nv.attributes.id,s)}
const get=i=>shapes.get(i),emu=v=>String(Math.round(v*12700));let nextId=400;
function place(s,x,y,w,h){const xf=all(s,'a:xfrm')[0];find(xf,'a:off').attributes={x:emu(x),y:emu(y)};find(xf,'a:ext').attributes={cx:emu(w),cy:emu(h)}}
function remove(id){const s=get(id);function walk(n){if(!n.elements)return;n.elements=n.elements.filter(x=>x!==s);n.elements.forEach(walk)}walk(tree)}
function text(s,value){const ts=all(s,'a:t');ts[0].elements=[{type:'text',text:value}];for(const t of ts.slice(1))t.elements=[]}
function clone(s,name){const t=structuredClone(s),nv=all(t,'p:cNvPr')[0];nv.attributes={id:String(nextId++),name};nv.elements=[];return t}
function path(s,points){const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x||.001,h=Math.max(...ys)-y||.001;place(s,x,y,w,h);find(all(s,'a:custGeom')[0],'a:pathLst').elements=[el('a:path',{w:emu(w),h:emu(h),fill:'none',stroke:'1'},points.map((p,i)=>el(i?'a:lnTo':'a:moveTo',{},[el('a:pt',{x:emu(p[0]-x),y:emu(p[1]-y)})])))];}
function line(name,points,color='000000',dash=false,arrow=true){const s=clone(get(310),name);path(s,points);const ln=find(find(s,'p:spPr'),'a:ln');ln.attributes={w:emu(dash?1.5:1.7)};ln.elements=[el('a:solidFill',{},[el('a:srgbClr',{val:color})]),el('a:prstDash',{val:dash?'dash':'solid'}),el('a:round')];if(arrow)ln.elements.push(el('a:tailEnd',{type:'triangle',w:'sm',len:'sm'}));tree.elements.splice(2,0,s);return s}
function label(name,txt,x,y,w,h,color='000000',size=12){const s=clone(get(242),name);place(s,x,y,w,h);find(s,'p:txBody').elements=[el('a:bodyPr',{anchor:'b',wrap:'none',lIns:'0',rIns:'0',tIns:'0',bIns:'0'},[el('a:noAutofit')]),el('a:lstStyle'),el('a:p',{},[el('a:pPr',{algn:'ctr'}),el('a:r',{},[el('a:rPr',{sz:String(size*100)},[el('a:solidFill',{},[el('a:srgbClr',{val:color})]),el('a:latin',{typeface:'Arial'})]),el('a:t',{},[{type:'text',text:txt}])])])];tree.elements.push(s);return s}
// Give supervision and inference their own columns; preserve all font sizes and symbols.
const p=xml.xml2js(await z.file('ppt/presentation.xml').async('string'));all(p,'p:sldSz')[0].attributes.cx=emu(1408);z.file('ppt/presentation.xml',xml.js2xml(p));
for(const [id,box] of Object.entries({14:[594,334,196,44],28:[594,210,94,36],29:[702,210,88,36],134:[584,170,216,24],157:[850,208,188,42],219:[850,286,188,36],30:[1130,106,248,44],33:[1130,160,248,44],34:[1130,214,248,44],32:[1130,334,248,44],31:[1126,72,256,28]}))place(get(+id),...box);
for(const rp of all(get(157),'a:rPr'))if(rp.attributes.sz==='1200')rp.attributes.sz='1100';
text(get(219),'Shared C-W supervision');
remove(135); // No border can be mistaken for a data-flow connector.
for(const id of [209,210,211,212,231,232,233,234,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,221,222])remove(id);
line('Shared h trunk',[[566,128],[566,356]],'000000',false,false);
line('Shared h to both C-W heads',[[566,200],[746,200]],'000000',false,false);
line('Shared h to Crackle',[[641,200],[641,210]]);
line('Shared h to Wheeze',[[746,200],[746,210]]);
line('Shared h to SPRSound',[[566,356],[594,356]]);
line('ICBHI native inference',[[790,128],[1130,128]]);
line('Fixed ICBHI readout bus',[[1106,128],[1106,236]],'000000',false,false);
line('ICBHI readout to HF test',[[1106,182],[1130,182]]);
line('ICBHI readout to KAUH',[[1106,236],[1130,236]]);
line('SPRSound native inference',[[790,356],[1130,356]]);
// Three unbroken supervision routes terminate at distinct CLS ports.
line('ICBHI batch native CE to CLS',[[692,150],[692,166],[944,166],[944,208]],'32631C',true);
line('C-W joint supervision to CLS',[[641,246],[641,258],[830,258],[830,236],[850,236]],'32631C',true);
line('Wheeze joins C-W supervision',[[746,246],[746,258]],'32631C',true,false);
const junction=clone(get(221),'C-W supervision junction');place(junction,744.5,256.5,3,3);for(const f of all(junction,'a:solidFill'))f.elements=[el('a:srgbClr',{val:'32631C'})];tree.elements.push(junction);
line('SPRSound batch native CE to CLS',[[790,344],[1052,344],[1052,237],[1038,237]],'32631C',true);
label('ICBHI-only native supervision label','ICBHI batch',792,143,122,18,'32631C',11);
label('SPRSound-only native supervision label','SPRSound batch',866,326,144,16,'32631C',11);
label('Classification training group title','Training supervision',840,177,208,23,'32631C',14);
// HF is an optional training source, not a branch of HF test predictions.
label('HF training source','HF train (optional)',780,267,125,16,'B6993A',11);
line('HF train C-W supervision to HF loss',[[772,258],[772,304],[850,304]],'B6993A',true);
const hfJunction=clone(get(221),'Optional HF train C-W junction');place(hfJunction,770.5,256.5,3,3);for(const f of all(hfJunction,'a:solidFill'))f.elements=[el('a:srgbClr',{val:'B6993A'})];tree.elements.push(hfJunction);
// The equation states the loss combination directly; keep teacher term arrows below it.
const notesPath='ppt/notesSlides/notesSlide1.xml';if(z.file(notesPath)){const notes=xml.xml2js(await z.file(notesPath).async('string'));const tx=all(notes,'p:txBody').find(b=>all(b,'a:t').length);if(tx)tx.elements.push(el('a:p',{},[el('a:r',{},[el('a:t',{},[{type:'text',text:'Discussion layout: native CE arrows are conditional on the batch dataset. HF train uses the same shared C-W heads; the gold branch from the shared C-W junction applies only to optional HF train supervision, not a new prediction head or HF-test feedback. Loss aggregation is expressed by the total objective equation without redundant long routes.'}])])]));z.file(notesPath,xml.js2xml(notes));}
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
await fs.writeFile(root+'/output/figure2_clear_supervision.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log('Created PPT-only supervision-layout revision.');
