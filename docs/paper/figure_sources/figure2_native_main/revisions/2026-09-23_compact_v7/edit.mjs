import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure2_compact_v7_20260923';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/current.pptx'));
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'));
const el=(name,attributes={},elements=[])=>({type:'element',name,attributes,elements});
const find=(n,k)=>n.elements?.find(x=>x.name===k);
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])if(c.name!=='mc:Fallback')a.push(...all(c,k));return a};
const tree=all(d,'p:spTree')[0], shapes=new Map();for(const s of [...all(d,'p:sp'),...all(d,'p:cxnSp')]){const nv=all(s,'p:cNvPr')[0];if(nv)shapes.set(+nv.attributes.id,s)}
const get=i=>shapes.get(i),emu=v=>String(Math.round(v*12700));
function place(s,x,y,w,h){const xf=all(s,'a:xfrm')[0];find(xf,'a:off').attributes={x:emu(x),y:emu(y)};find(xf,'a:ext').attributes={cx:emu(w),cy:emu(h)}}
function remove(s){function walk(n){if(!n.elements)return;n.elements=n.elements.filter(x=>x!==s);n.elements.forEach(walk)}walk(tree)}
function custom(s,commands){const pts=commands.flatMap(c=>Array.from({length:(c.length-1)/2},(_,i)=>[c[1+2*i],c[2+2*i]]));const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x||.001,h=Math.max(...ys)-y||.001;place(s,x,y,w,h);find(all(s,'a:custGeom')[0],'a:pathLst').elements=[el('a:path',{w:emu(w),h:emu(h),fill:'none',stroke:'1'},commands.map(c=>el(c[0]==='M'?'a:moveTo':c[0]==='C'?'a:cubicBezTo':'a:lnTo',{},Array.from({length:(c.length-1)/2},(_,i)=>el('a:pt',{x:emu(c[1+2*i]-x),y:emu(c[2+2*i]-y)})))))];}
const route=(s,pts)=>custom(s,pts.map((p,i)=>[i?'L':'M',...p]));
// Remove only the redundant regularization box; keep its losses in the objective.
remove(get(156));remove(get(218));
route(get(601),[[453,310],[453,390],[694,390],[694,414]]);
all(get(601),'p:cNvPr')[0].attributes.name='Attention representation to PCSL and GPAL objective';
// Four equally sized output boxes with identical 10-pt gaps.
place(get(34),1130,214,248,44);place(get(32),1130,268,248,44);
route(get(406),[[1106,128],[1106,236]]);route(get(408),[[1106,236],[1130,236]]);
// Tighten the gap above SPRSound while keeping the optional HF branch clear.
place(get(14),594,286,196,44);
route(get(400),[[566,128],[566,308]]);route(get(404),[[566,308],[594,308]]);
route(get(409),[[790,308],[1068,308],[1068,290],[1130,290]]);
place(get(219),850,264,188,36);place(get(418),642,260,122,16);
route(get(419),[[772,258],[772,278],[834,278],[834,282],[850,282]]);
custom(get(414),[['M',790,296],['L',842,296],['L',842,286],['C',848,286,848,278,842,278],['L',842,244],['L',850,244]]);
custom(get(700),[['M',842,286],['C',848,286,848,278,842,278]]);
// Move crossover bridges with their routes; preserve every loss-to-objective link.
custom(get(608),[['M',1038,220],['L',1084,220],['L',1084,286],['C',1090,286,1090,294,1084,294],['L',1084,382],['L',848,382],['L',848,414]]);
custom(get(701),[['M',1084,286],['C',1090,286,1090,294,1084,294]]);
custom(get(607),[['M',944,300],['L',944,304],['C',950,304,950,312,944,312],['L',944,378],['C',950,378,950,386,944,386],['L',944,398],['L',1006,398],['L',1006,414]]);
custom(get(702),[['M',944,304],['C',950,304,950,312,944,312]]);
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
await fs.writeFile(root+'/output/figure2_compact_v7.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log('Removed intermediate regularization box; equal output gaps; SPRSound head raised 48 pt.');
