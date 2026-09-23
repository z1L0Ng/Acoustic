import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='/private/tmp/figure2_layout_discussion_20260923';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/original.pptx'));
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'));
const el=(name,attributes={},elements=[])=>({type:'element',name,attributes,elements});
const find=(n,k)=>n.elements?.find(x=>x.name===k);
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])if(c.name!=='mc:Fallback')a.push(...all(c,k));return a};
const tree=all(d,'p:spTree')[0];
const shapes=new Map();for(const s of [...all(d,'p:sp'),...all(d,'p:cxnSp'),...all(d,'p:pic')]){const nv=all(s,'p:cNvPr')[0];if(nv)shapes.set(+nv.attributes.id,s)}
let nextId=200;
const shape=i=>shapes.get(i);
const emu=x=>String(Math.round(x*12700));
const loc=(s,x,y,w,h)=>{let xf=all(s,'a:xfrm')[0];find(xf,'a:off').attributes={x:emu(x),y:emu(y)};find(xf,'a:ext').attributes={cx:emu(w),cy:emu(h)};};
const clone=(s,name)=>{const t=structuredClone(s);const nv=all(t,'p:cNvPr')[0];nv.attributes={id:String(nextId++),name};nv.elements=[];return t};
const remove=(id)=>{const s=shape(id);function rec(n){if(!n.elements)return; n.elements=n.elements.filter(x=>x!==s);for(const c of n.elements)rec(c)}rec(tree)};
const colorFill=c=>el('a:solidFill',{},[el('a:srgbClr',{val:c})]);
const line=(name,pts,color='000000',dash=false,arrow=true)=>{
 const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x||.001,h=Math.max(...ys)-y||.001;
 const t=clone(shape(154),name);loc(t,x,y,w,h);
 const sp=find(t,'p:spPr');
 const geom=find(sp,'a:custGeom');find(geom,'a:pathLst').elements=[el('a:path',{w:emu(w),h:emu(h),fill:'none',stroke:'1'},pts.map((p,i)=>el(i?'a:lnTo':'a:moveTo',{},[el('a:pt',{x:emu(p[0]-x),y:emu(p[1]-y)})])))];
 const ln=find(sp,'a:ln');ln.attributes={w:emu(dash?1.3:1.7)};ln.elements=[colorFill(color),el('a:prstDash',{val:dash?'dash':'solid'}),el('a:round')];if(arrow)ln.elements.push(el('a:tailEnd',{type:'triangle',w:'sm',len:'sm'}));
 tree.elements.splice(2,0,t);return t;
};
function text(name,txt,x,y,w,h,size=14,color='000000',bold=false,bottom=false){
 const t=clone(shape(161),name);loc(t,x,y,w,h);
 find(t,'p:txBody').elements=[el('a:bodyPr',{anchor:bottom?'b':'ctr',wrap:'none',lIns:'0',rIns:'0',tIns:'0',bIns:'0'},[el('a:noAutofit')]),el('a:lstStyle'),...txt.split('\n').map(s=>el('a:p',{},[el('a:pPr',{algn:'ctr'}),el('a:r',{},[el('a:rPr',{sz:String(size*100),b:bold?'1':'0'},[colorFill(color),el('a:latin',{typeface:'Arial'})]),el('a:t',{},[{type:'text',text:s}])])]))];tree.elements.push(t);return t;
}
// Preserve every scientific label and native Office Math expression.
for(const [i,p] of Object.entries({4:[372,151,162,44],8:[229,150,95,77],12:[372,266,162,44],13:[594,108,196,40],14:[594,170,196,40],16:[60,172,132,33],18:[34,134,178,29],23:[584,72,216,28],24:[46,326,174,48],28:[594,278,94,32],29:[702,278,88,32],30:[872,106,248,44],31:[868,72,256,28],32:[872,266,248,44],33:[872,160,248,44],34:[872,214,248,44],47:[341,125,20,12],48:[334,140,40.5,21.8],85:[584,100,216,118],118:[458,313,20,12],119:[405,310,40.5,21.8],134:[584,239,216,27],135:[584,270,216,48],136:[539,150,22,12],156:[372,338,162,40],157:[584,338,216,40],97:[355,414,765,48]}))loc(shape(+i),...p);
// Use a non-rotated custom trapezoid to make its displayed boundary explicit.
const enc=shape(5),encsp=find(enc,'p:spPr');loc(enc,222,123,108,131);all(enc,'a:xfrm')[0].attributes={};
encsp.elements=encsp.elements.filter(e=>e.name!=='a:prstGeom');
encsp.elements.splice(1,0,el('a:custGeom',{},[el('a:avLst'),el('a:gdLst'),el('a:ahLst'),el('a:cxnLst'),el('a:rect',{l:'0',t:'0',r:'r',b:'b'}),el('a:pathLst',{},[el('a:path',{w:emu(108),h:emu(131)},[el('a:moveTo',{},[el('a:pt',{x:'0',y:'0'})]),el('a:lnTo',{},[el('a:pt',{x:emu(108),y:emu(22)})]),el('a:lnTo',{},[el('a:pt',{x:emu(108),y:emu(109)})]),el('a:lnTo',{},[el('a:pt',{x:'0',y:emu(131)})]),el('a:close')])])]))
// Remove outdated routes and route each function in its own corridor.
for(const i of [19,22,90,116,...Array.from({length:19},(_,j)=>137+j),158,159,160,161])remove(i);
line('Audio to BEATs',[[197,188],[222,188]]);
line('Encoder to mean pooling',[[330,173],[372,173]]);
line('Encoder tokens to attention',[[348,173],[348,288],[372,288]]);
line('Shared representation trunk',[[534,173],[566,173]],'000000',false,false);
line('Shared representation vertical bus',[[566,128],[566,266],[746,266]],'000000',false,false);
line('Shared h to ICBHI',[[566,128],[594,128]]);
line('Shared h to SPRSound',[[566,190],[594,190]]);
line('Shared h to Crackle',[[641,266],[641,278]]);
line('Shared h to Wheeze',[[746,266],[746,278]]);
line('ICBHI native readout',[[790,128],[872,128]]);
line('ICBHI external readout bus',[[856,128],[856,236]],'000000',false,false);
line('ICBHI to HF test',[[856,182],[872,182]]);
line('ICBHI to KAUH',[[856,236],[872,236]]);
line('SPRSound native readout',[[790,190],[832,190],[832,288],[872,288]]);
line('Native task supervision upper route',[[584,218],[576,218],[576,261]],'32631C',true,false);
line('Native task supervision to Lcls',[[576,271],[576,358],[584,358]],'32631C',true);
line('Crackle supervision to Lcls',[[641,310],[641,338]],'32631C',true);
line('Wheeze supervision to Lcls',[[746,310],[746,338]],'32631C',true);
line('Attention representation to regularization',[[453,310],[453,338]],'7030A0');
// Keep HF optional explicit and connect it to C/W, independently of output boxes.
const hf=clone(shape(157),'Optional HF loss');loc(hf,872,338,248,40);
const flat=n=>{let t='';if(n.type==='text')t+=n.text;for(const c of n.elements??[])t+=flat(c);return t};
const total=shape(97);const om=all(total,'m:oMath')[0];const terms=om.elements.filter(e=>e.name==='m:sSub');
if(terms.length!==7)throw Error('Unexpected total objective');
const hfmath=all(hf,'ns0:oMath')[0];hfmath.elements=[structuredClone(terms[5])];
for(const t of all(hf,'a:t'))t.elements=[{type:'text',text:'HF C/W (optional)'}];
for(const f of all(hf,'a:solidFill'))f.elements=[el('a:srgbClr',{val:'B6993A'})];
const hfsp=find(hf,'p:spPr');find(hfsp,'a:solidFill').elements=[el('a:srgbClr',{val:'FFF5DB'})];tree.elements.push(hf);
line('Optional HF C/W supervision',[[641,326],[996,326],[996,338]],'B6993A',true);
for(const x of [641,746]){const dot=clone(shape(143),'C/W optional HF branch');loc(dot,x-1.5,324.5,3,3);for(const f of all(dot,'a:solidFill'))f.elements=[el('a:srgbClr',{val:'B6993A'})];tree.elements.push(dot);}
// Split the same total objective into editable math terms with exact arrow targets.
const totalTemplate=structuredClone(total), totalbody=find(total,'p:txBody');totalbody.elements=totalbody.elements.filter(e=>e.name!=='a:p');totalbody.elements.push(el('a:p'));
function mathPart(name,terms,x,w){const s=clone(totalTemplate,name);loc(s,x,414,w,48);const sp=find(s,'p:spPr');sp.elements=sp.elements.filter(e=>!['a:solidFill','a:ln'].includes(e.name));sp.elements.push(el('a:noFill'),el('a:ln',{w:'0'},[el('a:noFill')]));const math=all(s,'m:oMath')[0];math.elements=structuredClone(terms);tree.elements.push(s);return s;}
mathPart('Total loss equals',om.elements.slice(0,2),360,65);
mathPart('Weighted patient contrastive term',terms.slice(0,2),425,170);
mathPart('Weighted attention regularization term',terms.slice(2,4),609,170);
mathPart('Classification term',[terms[6]],793,110);
mathPart('Weighted optional HF term',terms.slice(4,6),917,178);
const plus=om.elements.find(e=>e.name==='m:r'&&flat(e).includes('+'));
for(const x of [588,776,896])mathPart('Objective addition',[plus],x,22);
line('PAFA regularization to objective terms',[[453,378],[453,400],[694,400],[694,414]],'7030A0',true);
line('Patient contrastive term branch',[[510,400],[510,414]],'7030A0',true);
line('Classification loss to objective',[[692,378],[692,388],[848,388],[848,414]],'32631C',true);
line('HF loss to objective',[[996,378],[996,398],[1006,398],[1006,414]],'B6993A',true);
for(const [x,label,color] of [[510,'Patient\ncontrastive','333333'],[694,'Attention\nregularization','7030A0'],[848,'Native + C/W\nclassification','32631C'],[1006,'HF supervision\n(optional)','B6993A']]){
 line('Restored teacher annotation arrow '+label.replaceAll('\n',' '),[[x,481],[x,462]],color,false);
 text('Bottom aligned loss annotation '+label.replaceAll('\n',' '),label,x-76,486,152,32,12,color,false,true);
}
// Align peer heading baselines and remove tiny inconsistent inner margins.
for(const id of [23,31,134]){const s=shape(id);for(const rp of all(s,'a:rPr'))rp.attributes.sz='1600';const body=all(s,'a:bodyPr')[0];body.attributes.anchor='b';body.attributes.bIns='0';}
for(const id of [28,29])for(const rp of all(shape(id),'a:rPr'))rp.attributes.sz='1400';
all(d,'p:sld')[0].attributes['xmlns:m']='http://schemas.openxmlformats.org/officeDocument/2006/math';
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
const dest=root+'/output/figure2_method_layout_discussion.pptx';
await fs.writeFile(dest,await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
try {const imported=await PresentationFile.importPptx(await FileBlob.load(dest)); console.log((await imported.inspect({kind:'slide',maxChars:500})).ndjson);} catch(e){console.log('Import review:',String(e.message??e).slice(0,240));}
console.log('Editable discussion PPT only; no PDF/image export.');
