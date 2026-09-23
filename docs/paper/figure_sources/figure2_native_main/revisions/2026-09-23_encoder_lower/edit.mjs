import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure2_encoder_lower_20260923';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/current.pptx'));
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'));
const find=(n,k)=>n.elements?.find(x=>x.name===k);
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])if(c.name!=='mc:Fallback')a.push(...all(c,k));return a};
const el=(name,attributes={},elements=[])=>({type:'element',name,attributes,elements});
const shapes=new Map();for(const s of [...all(d,'p:sp'),...all(d,'p:cxnSp')]){const nv=all(s,'p:cNvPr')[0];if(nv)shapes.set(+nv.attributes.id,s)}
const emu=v=>String(Math.round(v*12700));
function place(s,x,y,w,h){const xf=all(s,'a:xfrm')[0];find(xf,'a:off').attributes={x:emu(x),y:emu(y)};find(xf,'a:ext').attributes={cx:emu(w),cy:emu(h)}}
function path(s,points){const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x||.001,h=Math.max(...ys)-y||.001;place(s,x,y,w,h);const geom=all(s,'a:custGeom')[0];find(geom,'a:pathLst').elements=[el('a:path',{w:emu(w),h:emu(h),fill:'none',stroke:'1'},points.map((p,i)=>el(i?'a:lnTo':'a:moveTo',{},[el('a:pt',{x:emu(p[0]-x),y:emu(p[1]-y)})])))];}
// Move the encoder as a unit; its bottom edge now matches the training-loss row.
place(shapes.get(5),72,247,108,131);
place(shapes.get(8),79,274,95,77);
place(shapes.get(24),46,432,174,48);
// Audio enters from above; the shared token trunk supplies both projections.
path(shapes.get(200),[[126,205],[126,258]]);
path(shapes.get(201),[[180,312.5],[348,312.5],[348,173],[372,173]]);
path(shapes.get(202),[[348,288],[372,288]]);
place(shapes.get(47),254,289,20,12);
place(shapes.get(48),244,317,40.5,21.8);
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
await fs.writeFile(root+'/output/figure2_encoder_lower.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log('Encoder lower left; all downstream heads, losses, and annotations unchanged.');
