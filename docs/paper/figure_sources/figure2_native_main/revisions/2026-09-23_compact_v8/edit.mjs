import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure2_compact_v8_20260923';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/current.pptx'));
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'));
const el=(name,attributes={},elements=[])=>({type:'element',name,attributes,elements});
const find=(n,k)=>n.elements?.find(x=>x.name===k);
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])if(c.name!=='mc:Fallback')a.push(...all(c,k));return a};
const shapes=new Map();for(const s of [...all(d,'p:sp'),...all(d,'p:cxnSp')]){const nv=all(s,'p:cNvPr')[0];if(nv)shapes.set(+nv.attributes.id,s)}
const get=i=>shapes.get(i),emu=v=>String(Math.round(v*12700));
function place(s,x,y,w,h){const xf=all(s,'a:xfrm')[0];find(xf,'a:off').attributes={x:emu(x),y:emu(y)};find(xf,'a:ext').attributes={cx:emu(w),cy:emu(h)}}
function shift(s,dy){const o=find(all(s,'a:xfrm')[0],'a:off');o.attributes.y=String(+o.attributes.y+Math.round(dy*12700))}
function custom(s,commands){const pts=commands.flatMap(c=>Array.from({length:(c.length-1)/2},(_,i)=>[c[1+2*i],c[2+2*i]]));const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x||.001,h=Math.max(...ys)-y||.001;place(s,x,y,w,h);find(all(s,'a:custGeom')[0],'a:pathLst').elements=[el('a:path',{w:emu(w),h:emu(h),fill:'none',stroke:'1'},commands.map(c=>el(c[0]==='M'?'a:moveTo':c[0]==='C'?'a:cubicBezTo':'a:lnTo',{},Array.from({length:(c.length-1)/2},(_,i)=>el('a:pt',{x:emu(c[1+2*i]-x),y:emu(c[2+2*i]-y)})))))];}
const route=(s,pts)=>custom(s,pts.map((p,i)=>[i?'L':'M',...p]));
// Raise the full objective and its annotations together, without scaling any text.
for(const id of [97,223,224,225,226,227,228,229,230,235,236,237,238,239,240,241,242])shift(get(id),-50);
place(get(24),46,404,174,48);
// Maintain distinct routing levels and clear final arrow stems.
route(get(601),[[453,310],[453,340],[694,340],[694,364]]);
route(get(602),[[510,340],[510,364]]);place(get(603),508.5,338.5,3,3);
custom(get(608),[['M',1038,220],['L',1084,220],['L',1084,286],['C',1090,286,1090,294,1084,294],['L',1084,344],['L',848,344],['L',848,364]]);
custom(get(607),[['M',944,300],['L',944,304],['C',950,304,950,312,944,312],['L',944,340],['C',950,340,950,348,944,348],['L',944,352],['L',1006,352],['L',1006,364]]);
custom(get(703),[['M',944,340],['C',950,340,950,348,944,348]]);
// Trim the same amount from the page bottom, retaining the existing lower margin.
const p=xml.xml2js(await z.file('ppt/presentation.xml').async('string'));all(p,'p:sldSz')[0].attributes.cy=emu(490);z.file('ppt/presentation.xml',xml.js2xml(p));
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
await fs.writeFile(root+'/output/figure2_compact_v8.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log('Objective raised 50 pt; all connection endpoints retained; typography unchanged.');
