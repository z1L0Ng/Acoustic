import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure2_outputs_center_20260923';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/current.pptx'));
const parse=s=>xml.xml2js(s,{captureSpacesBetweenElements:true});
const d=parse(await z.file('ppt/slides/slide1.xml').async('string'));
const el=(name,attributes={},elements=[])=>({type:'element',name,attributes,elements});
const find=(n,k)=>n.elements?.find(x=>x.name===k);
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])a.push(...all(c,k));return a};
const shapes=new Map();for(const s of [...all(d,'p:sp'),...all(d,'p:cxnSp')]){const nv=all(s,'p:cNvPr')[0];if(nv)shapes.set(+nv.attributes.id,s)}
const get=i=>shapes.get(i),emu=v=>String(Math.round(v*12700));
function place(s,x,y,w,h){const xf=all(s,'a:xfrm')[0];find(xf,'a:off').attributes={x:emu(x),y:emu(y)};find(xf,'a:ext').attributes={cx:emu(w),cy:emu(h)}}
function shift(s,dy){const off=find(all(s,'a:xfrm')[0],'a:off');off.attributes.y=String(+off.attributes.y+dy*12700)}
function custom(s,commands){const pts=commands.flatMap(c=>Array.from({length:(c.length-1)/2},(_,i)=>[c[1+2*i],c[2+2*i]]));const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x||.001,h=Math.max(...ys)-y||.001;place(s,x,y,w,h);find(all(s,'a:custGeom')[0],'a:pathLst').elements=[el('a:path',{w:emu(w),h:emu(h),fill:'none',stroke:'1'},commands.map(c=>el(c[0]==='M'?'a:moveTo':c[0]==='C'?'a:cubicBezTo':'a:lnTo',{},Array.from({length:(c.length-1)/2},(_,i)=>el('a:pt',{x:emu(c[1+2*i]-x),y:emu(c[2+2*i]-y)})))))];}
const route=(s,pts)=>custom(s,pts.map((p,i)=>[i?'L':'M',...p]));
// Vertically center the whole output block, including its heading.
for(const i of [30,31,32,33,34])shift(get(i),70);
route(get(405),[[790,68],[1106,68],[1106,138],[1130,138]]);
route(get(406),[[1106,138],[1106,246]]);
route(get(407),[[1106,192],[1130,192]]);
route(get(408),[[1106,246],[1130,246]]);
route(get(409),[[790,258],[1118,258],[1118,300],[1130,300]]);
// Keep loss routing intact, updating only the crossing arcs to the new inference line.
custom(get(608),[['M',1038,160],['L',1084,160],['L',1084,254],['C',1090,254,1090,262,1084,262],['L',1084,284],['L',848,284],['L',848,304]]);
custom(get(701),[['M',1084,254],['C',1090,254,1090,262,1084,262]]);
custom(get(607),[['M',944,240],['L',944,254],['C',950,254,950,262,944,262],['L',944,280],['C',950,280,950,288,944,288],['L',944,292],['L',1006,292],['L',1006,304]]);
custom(get(702),[['M',944,254],['C',950,254,950,262,944,262]]);
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
await fs.writeFile(root+'/output/figure2_outputs_center.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log('Outputs and heading shifted down 70 pt; group centered at y=202 on 404-pt canvas.');
