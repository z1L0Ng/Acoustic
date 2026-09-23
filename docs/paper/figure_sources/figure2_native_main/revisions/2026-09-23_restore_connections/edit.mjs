import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure2_restore_connections_20260923';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/current.pptx'));
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'));
const el=(name,attributes={},elements=[])=>({type:'element',name,attributes,elements});
const find=(n,k)=>n.elements?.find(x=>x.name===k);
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])if(c.name!=='mc:Fallback')a.push(...all(c,k));return a};
const tree=all(d,'p:spTree')[0];const shapes=new Map();for(const s of [...all(d,'p:sp'),...all(d,'p:cxnSp')]){const nv=all(s,'p:cNvPr')[0];if(nv)shapes.set(+nv.attributes.id,s)}
const get=i=>shapes.get(i),emu=v=>String(Math.round(v*12700));let nextId=600;
function place(s,x,y,w,h){const xf=all(s,'a:xfrm')[0];find(xf,'a:off').attributes={x:emu(x),y:emu(y)};find(xf,'a:ext').attributes={cx:emu(w),cy:emu(h)}}
function clone(s,name){const t=structuredClone(s),nv=all(t,'p:cNvPr')[0];nv.attributes={id:String(nextId++),name};nv.elements=[];return t}
function customLine(name,commands,color,dash=true,arrow=true,width=1.5){
 const s=clone(get(419),name);
 const points=commands.flatMap(c=>{let a=[];for(let i=1;i<c.length;i+=2)a.push([c[i],c[i+1]]);return a});
 const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x||.001,h=Math.max(...ys)-y||.001;
 place(s,x,y,w,h);
 const path=commands.map(c=>el(c[0]==='M'?'a:moveTo':c[0]==='C'?'a:cubicBezTo':'a:lnTo',{},Array.from({length:(c.length-1)/2},(_,i)=>el('a:pt',{x:emu(c[1+i*2]-x),y:emu(c[2+i*2]-y)}))));
 find(all(s,'a:custGeom')[0],'a:pathLst').elements=[el('a:path',{w:emu(w),h:emu(h),fill:'none',stroke:'1'},path)];
 const ln=find(find(s,'p:spPr'),'a:ln');ln.attributes={w:emu(width)};ln.elements=[el('a:solidFill',{},[el('a:srgbClr',{val:color})]),el('a:prstDash',{val:dash?'dash':'solid'}),el('a:round')];if(arrow)ln.elements.push(el('a:tailEnd',{type:'triangle',w:'sm',len:'sm'}));
 tree.elements.push(s);return s;
}
function line(name,pts,color,dash=true,arrow=true){return customLine(name,pts.map((p,i)=>[i?'L':'M',...p]),color,dash,arrow)}
// Restore the shared-head boundary with clear space above and below it.
const boundary=clone(get(28),'Shared C-W heads dashed boundary');place(boundary,584,204,216,48);
boundary.elements=boundary.elements.filter(e=>e.name!=='p:txBody'&&e.name!=='p:style');
const bpr=find(boundary,'p:spPr');bpr.elements=bpr.elements.filter(e=>!['a:solidFill','a:ln','a:noFill'].includes(e.name));bpr.elements.push(el('a:noFill'),el('a:ln',{w:emu(1.5)},[el('a:solidFill',{},[el('a:srgbClr',{val:'32631C'})]),el('a:prstDash',{val:'dash'})]));tree.elements.splice(2,0,boundary);
place(get(134),584,166,216,24);
// Move only the shared input bus by 2 pt, so it cannot look like the boundary.
function reroute(s,pts){const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x||.001,h=Math.max(...ys)-y||.001;place(s,x,y,w,h);find(all(s,'a:custGeom')[0],'a:pathLst').elements=[el('a:path',{w:emu(w),h:emu(h),fill:'none',stroke:'1'},pts.map((p,i)=>el(i?'a:lnTo':'a:moveTo',{},[el('a:pt',{x:emu(p[0]-x),y:emu(p[1]-y)})])))];}
reroute(get(401),[[566,198],[746,198]]);reroute(get(402),[[641,198],[641,210]]);reroute(get(403),[[746,198],[746,210]]);
// Arrange the last two addends to match the left/right incoming loss routes.
place(get(226),951,414,110,48);place(get(227),759,414,178,48);
place(get(240),930,486,152,32);place(get(242),772,486,152,32);
reroute(get(239),[[1006,481],[1006,462]]);reroute(get(241),[[848,481],[848,462]]);
// Restore PCSL and GPAL links to their own objective terms.
line('PAFA losses to weighted objective',[[453,378],[453,390],[694,390],[694,414]],'7030A0');
line('PCSL objective branch',[[510,390],[510,414]],'7030A0');
const dot=clone(get(413),'PAFA objective branch junction');place(dot,508.5,388.5,3,3);for(const f of all(dot,'a:solidFill'))f.elements=[el('a:srgbClr',{val:'7030A0'})];tree.elements.push(dot);
// Small white-backed line jumps denote crossings, not junctions.
for(const [x,y] of [[944,344],[944,356],[1084,356]])customLine('Loss connector crossover bridge',[['M',x,y-4],['C',x+6,y-4,x+6,y+4,x,y+4]],'FFFFFF',false,false,4.5);
customLine('Optional HF loss to total objective',[
 ['M',944,322],['L',944,340],['C',950,340,950,348,944,348],['L',944,352],['C',950,352,950,360,944,360],['L',944,386],['L',848,386],['L',848,414]
],'B6993A');
customLine('Classification loss to total objective',[
 ['M',1038,220],['L',1084,220],['L',1084,352],['C',1090,352,1090,360,1084,360],['L',1084,398],['L',1006,398],['L',1006,414]
],'32631C');
const notes='ppt/notesSlides/notesSlide1.xml';if(z.file(notes)){const n=await z.file(notes).async('string');z.file(notes,n.replace('Loss aggregation is expressed by the total objective equation without redundant long routes.','Each loss is connected to its weighted term in the total objective. Curved crossover bridges do not denote junctions. The C-W dashed boundary groups the shared heads.'));}
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
await fs.writeFile(root+'/output/figure2_restore_connections.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log('Restored C-W grouping and all objective connections on the latest user-edited v4.');
