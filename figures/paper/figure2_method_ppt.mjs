// Build with the bundled Node.js runtime. Optional first argument: build directory.
// The PPTX contains native shapes, text boxes and attached connectors only.
import fs from 'node:fs/promises';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const workspace = path.dirname(fileURLToPath(import.meta.url));
const build = path.resolve(process.argv[2] ?? path.join(workspace, '.figure2_ppt_build'));
const skill = process.env.PRESENTATIONS_SKILL_DIR ?? '/Users/zilongzeng/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
process.env.RUNTIME_NODE_MODULES ??= '/Users/zilongzeng/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const { importRuntimeModule } = await import(pathToFileURL(path.join(skill, 'container_tools/runtime_helpers.mjs')).href);
const { Presentation, PresentationFile } = await importRuntimeModule('@oai/artifact-tool');
await fs.mkdir(build, { recursive: true });

const PX = 4 / 3;
const WIDTH_MM = 178;
const HEIGHT_MM = 57;
const fontFamily = 'Arial';
const fontPolicy = { basis: 'design', families: [fontFamily] };
const presentation = Presentation.create({ slideSize: { width: WIDTH_MM / 25.4 * 96, height: HEIGHT_MM / 25.4 * 96 } });
const slide = presentation.slides.add();
slide.background.fill = '#FFFFFF';
const C = {
  ink: '#20364C', line: '#34536B', blue: '#EAF2FA', blueEdge: '#567B9C',
  green: '#ECF6F1', greenEdge: '#42826A', purple: '#F3EFF8', purpleEdge: '#81609C',
  sand: '#FCF6E9', sandEdge: '#9D8558', gray: '#F2F4F6', grayEdge: '#7D8A96',
};
const pos = (x, y, w, h) => ({ left: x * PX, top: y * PX, width: w * PX, height: h * PX });
function rect(name, x, y, w, h, fill, edge, radius = 2) {
  return slide.shapes.add({ geometry: 'rect', name, position: pos(x,y,w,h), fill,
    line: { fill: edge, width: .65 * PX }, borderRadius: radius * PX });
}
function text(name, value, x, y, w, h, { size = 9, bold = false, align = 'left', color = C.ink, italic = false, inset = 0 } = {}) {
  const shape = slide.shapes.add({ geometry: 'textbox', name, position: pos(x,y,w,h), fill: 'none', line: { fill: 'none', width: 0 } });
  shape.text = value;
  shape.text.style = { typeface: fontFamily, fontSize: size * PX, bold, italic, color,
    alignment: align, verticalAlignment: 'middle', autoFit: 'none', wrap: 'none',
    insets: { top: 0, right: 0, bottom: 0, left: inset * PX } };
  return shape;
}
function connect(from, to, { fromSide = 'right', toSide = 'left', kind = 'elbow', label = false, arrow = true } = {}) {
  const connector = slide.shapes.connect(from, to, { fromSide, toSide, kind,
    line: { fill: label ? C.purpleEdge : C.line, width: .8 * PX, style: label ? 'dashed' : 'solid' },
    tail: { type: arrow ? 'triangle' : 'none', width: 'sm', length: 'sm' },
  });
  connector.bringToFront();
  return connector;
}

text('learning-panel', '(a) Shared attribute learning', 4, 2, 326, 13, { size: 10, bold: true });
text('readout-panel', '(b) Native task readouts', 354, 2, 147, 13, { size: 10, bold: true });

// Background regions follow the roadmap palette; all visible elements are editable.
const predictionGroup = rect('parallel-head-region', 239, 18, 95, 95, '#F5FAF7', C.greenEdge);
rect('icbhi-readout-region', 354, 44, 146, 86, C.sand, C.sandEdge);
const core = rect('core-audio', 4, 21, 91, 42, C.blue, C.blueEdge);
text('icbhi-audio-label', 'ICBHI cycles', 10, 24, 80, 12, { bold: true });
text('spr-audio-label', 'SPRSound events', 10, 37, 80, 12, { bold: true });
text('core-audio-role', 'Core training audio', 10, 50, 80, 11);

const annotations = rect('core-annotations', 4, 80, 91, 50, C.purple, C.purpleEdge);
text('annotation-label', 'Available labels', 10, 84, 80, 12, { bold: true });
text('rhonchi-stridor-example', 'SPR R/S: A only', 10, 100, 80, 12);
text('unavailable-attributes', 'C/W unknown', 10, 115, 80, 12);

const model = rect('shared-acoustic-representation', 114, 21, 111, 70, C.green, C.greenEdge);
text('beats-label', 'BEATs', 119, 25, 101, 15, { size: 11, bold: true, align: 'center' });
text('pooling-label', 'Token mean', 119, 45, 101, 12, { align: 'center' });
text('projection-label', 'Linear 768 → 256', 119, 62, 101, 12, { align: 'center' });
text('representation-label', 'Shared representation h', 117, 78, 105, 11, { align: 'center' });

const targets = rect('available-targets-and-masks', 114, 104, 111, 26, C.purple, C.purpleEdge);
text('target-label', 'Targets + masks', 120, 104, 99, 13, { bold: true, align: 'center' });
text('target-nodes', 'Available A/C/W', 120, 117, 99, 12, { align: 'center' });

const A = rect('parallel-A-predictor', 242, 20.5, 89, 22, '#FFFFFF', C.greenEdge);
text('A-name', 'A: abnormality', 248, 20.5, 78, 11, { bold: true });
text('A-activation', '2-logit softmax', 248, 31.5, 78, 11);
text('parallel-score-group-label', 'A/C/W scores', 246, 46, 81, 11, { italic: true, align: 'center' });
const crackle = rect('parallel-C-predictor', 242, 59.5, 89, 23, '#FFFFFF', C.greenEdge);
text('C-name', 'C: Crackle', 248, 60, 78, 11, { bold: true });
text('C-activation', '1-logit sigmoid', 248, 71, 78, 11);
const wheeze = rect('parallel-W-predictor', 242, 88, 89, 23, '#FFFFFF', C.greenEdge);
text('W-name', 'W: Wheeze', 248, 88.5, 78, 11, { bold: true });
text('W-activation', '1-logit sigmoid', 248, 99.5, 78, 11);
const loss = rect('masked-classification-loss', 242, 118, 89, 12, C.purple, C.purpleEdge);
text('loss-label', 'Masked CE / BCE', 246, 118, 81, 12, { align: 'center' });

const spr = rect('sprsound-native-binary', 354, 20.5, 146, 22, C.blue, C.blueEdge);
text('spr-native-label', 'SPRSound: A argmax', 360, 20.5, 133, 11, { bold: true });
text('spr-native-classes', 'Normal / Adventitious', 360, 31.5, 133, 11);
text('icbhi-native-label', 'ICBHI', 360, 44.5, 133, 11, { bold: true });
const aGate = rect('icbhi-A-decision', 361, 56, 40, 14, '#FFFFFF', C.sandEdge);
text('icbhi-A-gate-label', 'A gate', 363, 56.5, 36, 12, { align: 'center' });
const normal = text('icbhi-normal', 'Normal', 446, 57, 50, 12, { inset: 4 });
text('icbhi-abnormal-branch', 'Abnormal', 389, 76.5, 58, 12);
const cwDecision = rect('icbhi-CW-decision', 361, 91, 40, 17, '#FFFFFF', C.sandEdge);
text('icbhi-CW-label', 'C/W rule', 362, 93, 38, 12, { align: 'center' });
const subtypes = text('icbhi-abnormal-outcomes', 'Crackle / Wheeze\nBoth', 414, 86.5, 85, 26, { inset: 4 });
text('icbhi-neither-positive-rule', 'Neither positive: margin rule', 361, 116, 133, 12);

connect(core, model);
connect(model, A);
connect(model, crackle);
connect(model, wheeze);
connect(annotations, targets, { label: true });
connect(targets, loss, { label: true });
// The enclosing parallel predictor group supplies all three scores to the loss.
connect(predictionGroup, loss, { fromSide: 'bottom', toSide: 'top', kind: 'straight' });
// SPRSound consumes A alone. C/W are routed only to the ICBHI attribute rule.
connect(A, spr, { kind: 'straight' });
connect(A, aGate);
connect(crackle, cwDecision);
connect(wheeze, cwDecision, { kind: 'straight' });
connect(aGate, normal, { kind: 'straight' });
connect(aGate, cwDecision, { fromSide: 'bottom', toSide: 'top', kind: 'straight' });
connect(cwDecision, subtypes, { kind: 'straight' });

// Compact study-role rows explicitly send external audio through the shared model.
text('hf-condition-role', 'HF auxiliary', 4, 132, 78, 13, { bold: true });
const hf = rect('hf-source-train', 85, 132, 110, 13, C.sand, C.sandEdge, 1.5);
text('hf-source-train-label', 'HF source-train', 88, 132, 104, 13, { align: 'center' });
const hfModel = rect('hf-same-model', 209, 132, 74, 13, C.green, C.greenEdge, 1.5);
text('hf-model-label', 'Same model', 211, 132, 70, 13, { align: 'center' });
const hfLoss = rect('additional-hf-loss', 297, 132, 203, 13, C.purple, C.purpleEdge, 1.5);
text('additional-hf-loss-label', '+ C/W auxiliary loss', 301, 132, 195, 13, { align: 'center' });
connect(hf, hfModel, { kind: 'straight' });
connect(hfModel, hfLoss, { kind: 'straight' });

text('fixed-evaluation-role', 'Fixed evaluation', 4, 147, 78, 13, { bold: true });
const evaluation = rect('external-evaluation-audio', 85, 147, 110, 13, C.gray, C.grayEdge, 1.5);
text('external-audio-label', 'HF source-test / KAUH', 88, 147, 104, 13, { align: 'center' });
const evalModel = rect('evaluation-same-model', 209, 147, 74, 13, C.green, C.greenEdge, 1.5);
text('eval-model-label', 'Same model', 211, 147, 70, 13, { align: 'center' });
const externalReadouts = rect('external-fixed-readouts', 297, 147, 203, 13, C.gray, C.grayEdge, 1.5);
text('hf-external-readout-label', 'HF: C/W', 301, 147, 36, 13);
text('kauh-external-readout-label', 'KAUH: B/D/E prob. mean → decoding', 343, 147, 153, 13, { align: 'center' });
connect(evaluation, evalModel, { kind: 'straight' });
connect(evalModel, externalReadouts, { kind: 'straight' });

slide.speakerNotes.textFrame.setText([
  'Sources: /Users/zilongzeng/Research/Acoustic/docs/paper/Overleaf_Sync_final/Section/2data.tex, Section/3method.tex, Section/4evaluation.tex and main.tex, read 2026-09-13.',
  'Visual organization: acoustic_end_to_end_pipeline_v2.8.png; sync-0813.pptx slide 10; pipeline_sketch_v0 slide 1; pipeline_audit_rough_v1 slide 4. These are visual references only.',
  'A predicts Normal/Abnormal with two-logit softmax. C/W are parallel sigmoid predictors of marginal attribute probabilities. R/S denotes SPRSound Rhonchi/Stridor. Their C/W labels are unknown and masked.',
  'ICBHI: A=Normal produces Normal; otherwise C/W thresholds produce Crackle, Wheeze or Both. If neither threshold is reached, larger probability-minus-threshold margin chooses Crackle/Wheeze, with ties to Crackle.',
  'C/W thresholds use core validation. Fixed selected models need not be validation-selected checkpoints. HF source-train contributes only in a separate auxiliary condition; it has no A loss and includes annotation-derived negative C/W targets.',
  'HF fixed evaluation covers annotated intervals and recording-level attribute presence. KAUH averages probabilities across B/D/E versions within each patient before decoding.',
  'PAFA retains its separate attention projection from encoder tokens; its regularizers and detailed hyperparameters remain in Section 3. Native + attributes is a separate benchmark control and is not part of the depicted model.',
].join('\n\n'));

const candidate = path.join(build, 'figure2_method_ppt.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidate);
const preview = await presentation.export({ slide, format: 'png', scale: 3.125 });
await fs.writeFile(path.join(build, 'artifact-preview.png'), new Uint8Array(await preview.arrayBuffer()));
const layout = await slide.export({ format: 'layout' });
await fs.writeFile(path.join(build, 'slide.layout.json'), await layout.text());
const runtimeBin = process.env.RUNTIME_BIN_DIR ?? '/Users/zilongzeng/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override';
execFileSync(path.join(runtimeBin, 'soffice'), ['--headless', '--convert-to', 'pdf', '--outdir', build, candidate], { stdio: 'inherit' });
const pdfPath = path.join(build, 'figure2_method_ppt.pdf');
execFileSync(path.join(runtimeBin, 'pdftoppm'), ['-r', '300', '-singlefile', '-png', pdfPath, path.join(build, 'figure2_method_ppt')], { stdio: 'inherit' });
console.log(JSON.stringify({ candidate, pdfPath, pngPath: path.join(build, 'figure2_method_ppt.png'), widthMm: WIDTH_MM, heightMm: HEIGHT_MM, fontPolicy, minimumFontPt: 9, slides: 1 }));
