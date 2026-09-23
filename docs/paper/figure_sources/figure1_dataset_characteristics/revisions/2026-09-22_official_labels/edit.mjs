import fs from 'node:fs/promises';
import JSZip from 'jszip';
import xml from 'xml-js';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='/private/tmp/figure1_official_labels_20260922';
const z=await JSZip.loadAsync(await fs.readFile(root+'/source/current.pptx'));
const old=await JSZip.loadAsync(await fs.readFile('/Users/zilongzeng/Research/Acoustic/docs/paper/figure_sources/figure1_dataset_characteristics/revisions/2026-09-22_full_labels_swap/before/figure1_dataset_characteristics.pptx'));
const all=(n,k)=>{let a=[];if(n.name===k)a.push(n);for(const c of n.elements??[])a.push(...all(c,k));return a};
const name=s=>all(s,'p:cNvPr')[0]?.attributes?.name;
const d=xml.xml2js(await z.file('ppt/slides/slide1.xml').async('string'));
const before=xml.xml2js(await old.file('ppt/slides/slide1.xml').async('string'));
const originals=Object.fromEntries(all(before,'p:spTree')[0].elements.filter(s=>name(s)).map(s=>[name(s),s]));
const labels={HF_Lung:{I:'I',E:'E',D:'D',Wheeze:'Wheeze',Rhonchi:'Rhonchi',Stridor:'Stridor'},KAUH:{N:'N',C:'C',IC:'I C',EW:'E W',IEW:'I E W',ICEW:'I C E W',Crep:'Crep',Bronchial:'Bronchial',ICB:'I C B'}};
for(const s of all(d,'p:spTree')[0].elements){
 const n=name(s);if(!n?.startsWith('a/'))continue;
 const orig=originals[n];if(orig){const xf=all(s,'a:xfrm')[0],oxf=all(orig,'a:xfrm')[0];if(xf&&oxf)xf.elements=structuredClone(oxf.elements);}
 for(const [ds,mapping] of Object.entries(labels)){
  const prefix='a/label-'+ds+'-';if(!n.startsWith(prefix))continue;
  const text=mapping[n.slice(prefix.length)],ts=all(s,'a:t');
  ts[0].elements=[{type:'text',text}];for(const t of ts.slice(1))t.elements=[];
 }
}
z.file('ppt/slides/slide1.xml',xml.js2xml(d));
const notes=xml.xml2js(await z.file('ppt/notesSlides/notesSlide1.xml').async('string'));
for(const t of all(notes,'a:t'))for(const v of t.elements??[]){
 if(v.type!=='text')continue;
 if(v.text.startsWith('Label expansions:'))v.text='HF raw annotation strings are I, E, D, Wheeze, Rhonchi, and Stridor. I means inhalation; E means exhalation; D means discontinuous adventitious sounds, all crackles. Sources: dataset/raw/hf_lung_v1/source_original/ and https://gitlab.com/techsupportHF/HF_Lung_V1 .';
 if(v.text.startsWith('KAUH label expansions'))v.text='KAUH Sound type strings match Data annotation.xlsx, with outer whitespace trimmed: N, C, I C, E W, I E W, I C E W, Crep, Bronchial, I C B. I = inspiratory; E = expiratory; W = wheezes; C = crackles; N = normal; Crep = crepitations. B in I C B is undefined in the source legend and is not expanded. Source: https://data.mendeley.com/datasets/jwyy9np4gv/3 .';
}
z.file('ppt/notesSlides/notesSlide1.xml',xml.js2xml(notes));
const path=root+'/output/figure1_official_labels.pptx';
await fs.writeFile(path,await z.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
const p=await PresentationFile.importPptx(await FileBlob.load(path));
console.log((await p.inspect({kind:'slide',maxChars:500})).ndjson);
