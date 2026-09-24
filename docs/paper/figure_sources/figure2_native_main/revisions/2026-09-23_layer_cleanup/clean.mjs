import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure2_layer_cleanup_20260923';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/figure2_user_v8.pptx'));
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'),{captureSpacesBetweenElements:true});
const el=(name,attributes={},elements=[])=>({type:'element',name,attributes,elements});
const find=(n,k)=>n.elements?.find(x=>x.name===k);
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])a.push(...all(c,k));return a};
let removed=0;
function clean(n){if(!n.elements)return;n.elements=n.elements.flatMap(c=>{if(c.name==='mc:AlternateContent'){const choice=find(c,'mc:Choice');const fallback=find(c,'mc:Fallback');if(choice?.elements?.some(x=>x.name==='p:sp')&&fallback){removed++;return choice.elements;}}return [c];});n.elements.forEach(clean)}
clean(d);
const slide=all(d,'p:sld')[0];slide.attributes['xmlns:a14']='http://schemas.microsoft.com/office/drawing/2010/main';slide.attributes['xmlns:m']='http://schemas.openxmlformats.org/officeDocument/2006/math';
for(const s of all(d,'p:sp')){
 const id=+all(s,'p:cNvPr')[0].attributes.id;
 if(id===97){
  s.elements=s.elements.filter(e=>e.name!=='p:txBody');
  const sp=find(s,'p:spPr');sp.elements=sp.elements.filter(e=>e.name!=='a:solidFill'&&e.name!=='a:effectLst'&&e.name!=='a:effectDag');
  const pos=sp.elements.findIndex(e=>e.name==='a:ln');sp.elements.splice(pos,0,el('a:solidFill',{},[el('a:srgbClr',{val:'E8E8E8'},[el('a:alpha',{val:'100000'})])]));sp.elements.push(el('a:effectLst'));
 }
 if(id>=223&&id<=230){
  const sp=find(s,'p:spPr');sp.elements=sp.elements.filter(e=>e.name!=='a:effectLst'&&e.name!=='a:effectDag');sp.elements.push(el('a:effectLst'));
 }
}
// Remove slide relationships to obsolete fallback images; retain any live image reference.
const used=new Set();function refs(n){for(const [k,v] of Object.entries(n.attributes??{}))if(['r:embed','r:link','r:id'].includes(k))used.add(v);for(const c of n.elements??[])refs(c)}refs(d);
const relPath='ppt/slides/_rels/slide1.xml.rels';const rel=xml.xml2js(await z.file(relPath).async('string'));const relRoot=all(rel,'Relationships')[0];let removedRefs=0;
relRoot.elements=relRoot.elements.filter(e=>{if(e.name==='Relationship'&&e.attributes.Type.endsWith('/image')&&!used.has(e.attributes.Id)){removedRefs++;return false;}return true});
z.file(relPath,xml.js2xml(rel));z.file('ppt/slides/slide1.xml',xml.js2xml(d));
await fs.writeFile(root+'/output/figure2_cleaned.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log(JSON.stringify({removedFallbackLayers:removed,removedUnusedSlideImageReferences:removedRefs,background:'opaque E8E8E8',geometry:'unchanged'}));
