import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure2_cls_left_20260923';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/current.pptx'));
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'));
const el=(name,attributes={},elements=[])=>({type:'element',name,attributes,elements});
const find=(n,k)=>n.elements?.find(x=>x.name===k);
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])if(c.name!=='mc:Fallback')a.push(...all(c,k));return a};
const tree=all(d,'p:spTree')[0], shapes=new Map();for(const s of [...all(d,'p:sp'),...all(d,'p:cxnSp')]){const nv=all(s,'p:cNvPr')[0];if(nv)shapes.set(+nv.attributes.id,s)}
const get=i=>shapes.get(i),emu=v=>String(Math.round(v*12700));let nextId=700;
function place(s,x,y,w,h){const xf=all(s,'a:xfrm')[0];find(xf,'a:off').attributes={x:emu(x),y:emu(y)};find(xf,'a:ext').attributes={cx:emu(w),cy:emu(h)}}
function remove(s){function walk(n){if(!n.elements)return;n.elements=n.elements.filter(x=>x!==s);n.elements.forEach(walk)}walk(tree)}
function front(s){remove(s);tree.elements.push(s)}
function custom(s,commands){const pts=commands.flatMap(c=>Array.from({length:(c.length-1)/2},(_,i)=>[c[1+2*i],c[2+2*i]]));const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x||.001,h=Math.max(...ys)-y||.001;place(s,x,y,w,h);find(all(s,'a:custGeom')[0],'a:pathLst').elements=[el('a:path',{w:emu(w),h:emu(h),fill:'none',stroke:'1'},commands.map(c=>el(c[0]==='M'?'a:moveTo':c[0]==='C'?'a:cubicBezTo':'a:lnTo',{},Array.from({length:(c.length-1)/2},(_,i)=>el('a:pt',{x:emu(c[1+2*i]-x),y:emu(c[2+2*i]-y)})))))];}
function route(s,pts){custom(s,pts.map((p,i)=>[i?'L':'M',...p]))}
function bridge(commands){const s=structuredClone(get(604));const nv=all(s,'p:cNvPr')[0];nv.attributes={id:String(nextId++),name:'Non-junction crossover bridge'};nv.elements=[];custom(s,commands);tree.elements.push(s)}
// All supervision arrows enter the left edge of CLS at separate ports.
route(get(410),[[692,150],[692,158],[810,158],[810,218],[850,218]]);
route(get(411),[[641,246],[641,258],[824,258],[824,230],[850,230]]);
custom(get(414),[['M',790,344],['L',842,344],['L',842,308],['C',848,308,848,300,842,300],['L',842,244],['L',850,244]]);
place(get(418),642,276,122,18);
// Put the optional HF term last; move its matching explanatory annotation with it.
place(get(226),793,414,110,48);place(get(227),917,414,178,48);
place(get(240),772,486,152,32);place(get(242),930,486,152,32);
route(get(239),[[848,481],[848,462]]);route(get(241),[[1006,481],[1006,462]]);
for(const id of [604,605,606])remove(get(id));
bridge([['M',842,308],['C',848,308,848,300,842,300]]);front(get(414));
// Restore the full objective paths with only marked non-junction crossings.
custom(get(608),[['M',1038,220],['L',1084,220],['L',1084,352],['C',1090,352,1090,360,1084,360],['L',1084,382],['L',848,382],['L',848,414]]);
bridge([['M',1084,352],['C',1090,352,1090,360,1084,360]]);front(get(608));
custom(get(607),[['M',944,322],['L',944,352],['C',950,352,950,360,944,360],['L',944,378],['C',950,378,950,386,944,386],['L',944,398],['L',1006,398],['L',1006,414]]);
bridge([['M',944,352],['C',950,352,950,360,944,360]]);bridge([['M',944,378],['C',950,378,950,386,944,386]]);front(get(607));
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
await fs.writeFile(root+'/output/figure2_cls_left.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log('CLS inputs unified on left; optional HF last; grouping and all objective connections retained.');
