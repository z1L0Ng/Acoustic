import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/zilongzeng/Research/Acoustic";
const SKILL_DIR = "/Users/zilongzeng/.codex/plugins/cache/openai-primary-runtime/presentations/26.903.11726/skills/presentations";
const TMP_DIR = path.join(workspaceDir, ".codex-build/paper_story_20260904");
const FINAL_PPTX = path.join(
  workspaceDir,
  "slides/Acoustic_ICASSP_story_and_next_evidence_2026-09-04_v2.pptx",
);
const RUNTIME_PYTHON = "/Users/zilongzeng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";

const {
  resolvePresentationFont,
  finalizePresentation,
} = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const family = resolvePresentationFont();
const presentation = Presentation.create({
  slideSize: { width: 1280, height: 720 },
});

const C = {
  ink: "#172033",
  muted: "#596579",
  purple: "#4E2A84",
  purpleLight: "#EEE8F6",
  teal: "#147D76",
  tealLight: "#E7F4F2",
  orange: "#C95E24",
  orangeLight: "#FBEEE7",
  blue: "#2C67A0",
  blueLight: "#EAF1F8",
  line: "#D8DEE8",
  paper: "#FFFFFF",
  soft: "#F7F8FB",
};

function addText(slide, text, left, top, width, height, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: family,
    fontSize: options.fontSize ?? 22,
    bold: options.bold ?? false,
    color: options.color ?? C.ink,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: options.autoFit ?? "shrinkText",
    wrap: "square",
    insets: options.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return shape;
}

function addRect(slide, left, top, width, height, fill, line = C.line, radius = 12) {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: line, width: 1.2 },
    borderRadius: radius,
  });
}

function addLine(slide, left, top, width, color = C.line, weight = 1.2) {
  return slide.shapes.add({
    geometry: "line",
    position: { left, top, width, height: 0 },
    fill: "none",
    line: { style: "solid", fill: color, width: weight },
  });
}

function addTitle(slide, title, subtitle, page) {
  slide.background.fill = C.paper;
  slide.shapes.add({
    geometry: "rect",
    position: { left: 0, top: 0, width: 1280, height: 10 },
    fill: C.purple,
    line: { fill: "none", width: 0 },
  });
  addText(slide, title, 64, 38, 1110, 54, {
    fontSize: 39,
    bold: true,
    color: C.ink,
  });
  addText(slide, subtitle, 66, 98, 1050, 28, {
    fontSize: 18,
    color: C.muted,
  });
  addText(slide, String(page) + "/3", 1180, 45, 44, 24, {
    fontSize: 16,
    color: C.muted,
    alignment: "right",
  });
}

function addFlowBlock(slide, x, y, w, h, accent, label, body, detail) {
  const box = addRect(slide, x, y, w, h, C.paper, C.line, 14);
  slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: 8, height: h },
    fill: accent,
    line: { fill: "none", width: 0 },
  });
  addText(slide, label, x + 22, y + 18, w - 38, 30, {
    fontSize: 22,
    bold: true,
    color: accent,
  });
  addText(slide, body, x + 22, y + 56, w - 38, 85, {
    fontSize: 20,
    color: C.ink,
  });
  addText(slide, detail, x + 22, y + h - 58, w - 38, 42, {
    fontSize: 16,
    color: C.muted,
  });
  return box;
}

{
  const slide = presentation.slides.add();
  addTitle(
    slide,
    "Paper story: controlled alignment across native tasks",
    "ICASSP 2027 direction after the 3 September advisor review",
    1,
  );

  const y = 174;
  const w = 270;
  const h = 330;
  const gap = 32;
  const x1 = 60;
  const x2 = x1 + w + gap;
  const x3 = x2 + w + gap;
  const x4 = x3 + w + gap;

  const b1 = addFlowBlock(
    slide, x1, y, w, h, C.orange,
    "Observed problem",
    "ICBHI-specialized checkpoints retain their source result but approach the 50% floor on SPRSound transfer.",
    "PAFA 64.14 / 55.82\nSG-SCL 60.98 / 59.98\nPatch-Mix 62.17 / 59.38",
  );
  const b2 = addFlowBlock(
    slide, x2, y, w, h, C.blue,
    "Concrete gap",
    "Labels overlap only partially. Cycles and events use different outputs and metrics. Unavailable labels cannot be treated as negatives.",
    "Need semantic sharing without erasing native tasks",
  );
  const b3 = addFlowBlock(
    slide, x3, y, w, h, C.purple,
    "Method response",
    "Shared BEATs representation with a Normal/Abnormal hierarchy, eligible Crackle/Wheeze losses, and dataset-native readouts.",
    "Key phrase: label-availability-aware task alignment",
  );
  const b4 = addFlowBlock(
    slide, x4, y, w, h, C.teal,
    "Evidence and question",
    "JH2 reaches ICBHI 61.17±0.31 and SPRSound 90.70±0.34 across three seeds.",
    "Still missing matched independent heads and source-only controls",
  );

  for (const left of [x1 + w + 5, x2 + w + 5, x3 + w + 5]) {
    slide.shapes.add({
      geometry: "rightArrow",
      position: { left, top: y + 148, width: 22, height: 32 },
      fill: C.muted,
      line: { fill: "none", width: 0 },
    });
  }

  addRect(slide, 60, 552, 1160, 92, C.soft, C.purple, 10);
  addText(slide, "Recommended scope", 82, 570, 220, 28, {
    fontSize: 18,
    bold: true,
    color: C.purple,
  });
  addText(
    slide,
    "ICBHI + SPRSound core paper. HF Lung and KAUH remain bounded supporting diagnostics.",
    320, 566, 870, 42,
    { fontSize: 20, bold: true, color: C.ink, verticalAlignment: "middle" },
  );
  addText(
    slide,
    "Claim target: controlled retention across native tasks, not SOTA or universal generalization.",
    320, 610, 870, 24,
    { fontSize: 16, color: C.muted },
  );

  slide.speakerNotes.textFrame.setText(
    "Sources: docs/paper/ICASSP_2026_acoustic_disease/STORY_REVIEW_2026-09-04_zh.md; " +
    "result/icbhi_strong_method_reproduction/metrics.json; " +
    "result/pafa_sprsound_transfer_20260722_235659/metrics.json; " +
    "result/sg_scl_sprsound_transfer_20260722_235659/metrics.json; " +
    "result/sprsound_patchmix_frozen_transfer/metrics.json; " +
    "result/reproduce/pafa_joint_hierarchy/PAFA_JH2_main_multiseed/multiseed_summary.md. " +
    "Transfer rows are zero-target-tuning diagnostics, not matched joint-training gains.",
  );
}

{
  const slide = presentation.slides.add();
  addTitle(
    slide,
    "Three contributions, each tied to evidence",
    "The method claim remains conditional until the matched control is complete",
    2,
  );

  const rows = [
    {
      n: "01",
      title: "Native-task-preserving formulation",
      body: "Align partially observed respiratory labels while retaining cycle-level ICBHI and event-level SPRSound outputs and metrics.",
      evidence: "Figure 1 + task/label contract",
      color: C.blue,
    },
    {
      n: "02",
      title: "Eligibility-aware hierarchical alignment",
      body: "Share Normal/Abnormal, Crackle, and Wheeze semantics without converting unavailable targets into negative supervision.",
      evidence: "Figure 2 + independent-head control needed",
      color: C.purple,
    },
    {
      n: "03",
      title: "Dataset-dependent retention analysis",
      body: "Report native scores, specificity, sensitivity, and class recall across three seeds instead of constructing one pooled score.",
      evidence: "Table I + Table II + per-class analysis",
      color: C.teal,
    },
  ];

  const top0 = 166;
  const rh = 142;
  rows.forEach((r, i) => {
    const top = top0 + i * rh;
    addText(slide, r.n, 72, top + 8, 72, 54, {
      fontSize: 42,
      bold: true,
      color: r.color,
    });
    addText(slide, r.title, 168, top + 5, 480, 34, {
      fontSize: 25,
      bold: true,
      color: C.ink,
    });
    addText(slide, r.body, 168, top + 48, 720, 64, {
      fontSize: 20,
      color: C.ink,
    });
    addText(slide, r.evidence, 930, top + 23, 278, 56, {
      fontSize: 18,
      bold: true,
      color: r.color,
      alignment: "right",
      verticalAlignment: "middle",
    });
    if (i < rows.length - 1) addLine(slide, 72, top + 128, 1136, C.line, 1.2);
  });

  addRect(slide, 72, 604, 1136, 62, C.orangeLight, C.orange, 8);
  addText(slide, "Claim boundary", 92, 622, 160, 24, {
    fontSize: 19,
    bold: true,
    color: C.orange,
  });
  addText(
    slide,
    "ICBHI is test-selected; no pooled score; no SOTA claim; HF/KAUH remain diagnostics.",
    254, 619, 920, 28,
    { fontSize: 19, color: C.ink, verticalAlignment: "middle" },
  );

  slide.speakerNotes.textFrame.setText(
    "Contribution wording follows the independent Claude and GPT-5.6 Pro reviews. " +
    "Contribution 2 cannot claim superiority until a matched independent-head comparison is complete. " +
    "Main evidence: JH2 ICBHI Score 0.611681 plus/minus 0.003139 and SPRSound official Score " +
    "0.906986 plus/minus 0.003424, n=3.",
  );
}

{
  const slide = presentation.slides.add();
  addTitle(
    slide,
    "Evidence needed before the method claim is credible",
    "Run only work that fills Figure 1, Figure 2, Table I, or Table II",
    3,
  );

  const headers = [
    { x: 68, w: 90, text: "Priority" },
    { x: 170, w: 310, text: "Deliverable" },
    { x: 500, w: 470, text: "Question answered" },
    { x: 990, w: 220, text: "Owner" },
  ];
  headers.forEach(hd => addText(slide, hd.text, hd.x, 150, hd.w, 30, {
    fontSize: 18,
    bold: true,
    color: C.muted,
  }));
  addLine(slide, 64, 184, 1152, C.ink, 1.5);

  const items = [
    ["P0", "Joint independent heads", "Does the hierarchy add value beyond a shared encoder?", "Zilong / model design"],
    ["P0", "ICBHI-only + SPR-only controls", "Does joint learning help, hurt, or retain each native task?", "Zilong / model design"],
    ["P0", "Prospective clean seed42 suite", "Do conclusions survive validation-only selection and one test access?", "Zilong / model design"],
    ["P0", "Class-matched acoustic analysis", "Which domain differences remain after controlling label composition?", "Wade"],
    ["P1", "Baseline provenance registry", "Which published and local rows are truly comparable?", "Hanlin"],
  ];
  const rowTop = 196;
  const rowH = 76;
  items.forEach((r, i) => {
    const top = rowTop + i * rowH;
    if (i % 2 === 0) {
      slide.shapes.add({
        geometry: "rect",
        position: { left: 64, top, width: 1152, height: rowH - 2 },
        fill: C.soft,
        line: { fill: "none", width: 0 },
      });
    }
    addText(slide, r[0], 76, top + 18, 70, 30, {
      fontSize: 20,
      bold: true,
      color: r[0] === "P0" ? C.orange : C.blue,
    });
    addText(slide, r[1], 170, top + 14, 300, 42, {
      fontSize: 20,
      bold: true,
      color: C.ink,
      verticalAlignment: "middle",
    });
    addText(slide, r[2], 500, top + 10, 450, 50, {
      fontSize: 18,
      color: C.ink,
      verticalAlignment: "middle",
    });
    addText(slide, r[3], 990, top + 14, 210, 42, {
      fontSize: 18,
      color: C.muted,
      alignment: "right",
      verticalAlignment: "middle",
    });
  });

  addRect(slide, 64, 604, 1152, 70, C.purpleLight, C.purple, 8);
  addText(slide, "Sunday 9/6", 86, 624, 150, 28, {
    fontSize: 20,
    bold: true,
    color: C.purple,
  });
  addText(
    slide,
    "Data/Method v1, Figure 2, fixed Table I/II contracts, and one submission-blocker list",
    244, 619, 930, 34,
    { fontSize: 20, bold: true, color: C.ink, verticalAlignment: "middle" },
  );

  slide.speakerNotes.textFrame.setText(
    "The clean suite should contain joint hierarchy, joint independent heads, ICBHI-only hierarchy, " +
    "and SPRSound-only hierarchy under one pre-registered seed42 protocol. Hanlin owns only the " +
    "baseline registry and Table I source rows. HF/KAUH and JH3/JH4 expansions are outside the core queue.",
  );
}

const candidatePath = path.join(TMP_DIR, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const requirements = {
  explicitTotalSlideCount: 3,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = { basis: "design", families: [family] };
const expectedSlideSizeEmu = "12192000,6858000";
const stagingDir = path.join(workspaceDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const validationCandidate = path.join(stagingDir, "paper_story_candidate.pptx");
await fs.copyFile(candidatePath, validationCandidate);

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath: validationCandidate,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(
    SKILL_DIR,
    "container_tools/inspect_presentation_package_integrity.py",
  ),
  layoutValidatorPath: path.join(
    SKILL_DIR,
    "container_tools/inspect_presentation_layout_geometry.py",
  ),
  layoutArgs: [
    "--expected-slide-size-emu", expectedSlideSizeEmu,
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "paper_story_20260904_v2.validation.json"),
});

console.log(JSON.stringify({ family, finalPath: FINAL_PPTX, result }, null, 2));
