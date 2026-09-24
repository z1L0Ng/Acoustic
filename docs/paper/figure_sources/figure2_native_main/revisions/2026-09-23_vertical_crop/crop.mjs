import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
const root='/private/tmp/figure2_vertical_crop_20260923';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/current.pptx'));
const parse=s=>xml.xml2js(s,{captureSpacesBetweenElements:true});
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])a.push(...all(c,k));return a};
const find=(n,k)=>n.elements?.find(e=>e.name===k);
const d=parse(await z.file('ppt/slides/slide1.xml').async('string'));
let count=0;
for(const sp of all(d,'p:spPr')){const xf=find(sp,'a:xfrm'),off=xf&&find(xf,'a:off');if(off){off.attributes.y=String(+off.attributes.y-60*12700);count++}}
const p=parse(await z.file('ppt/presentation.xml').async('string'));all(p,'p:sldSz')[0].attributes.cy=String(404*12700);
z.file('ppt/slides/slide1.xml',xml.js2xml(d));z.file('ppt/presentation.xml',xml.js2xml(p));
await fs.writeFile(root+'/output/figure2_vertical_crop.pptx',await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
console.log(JSON.stringify({translatedNativeObjects:count,topTrimPt:60,bottomTrimPt:26,newCanvasPt:[1408,404]}));
