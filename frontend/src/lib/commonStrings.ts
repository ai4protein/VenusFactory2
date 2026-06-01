/** Shared translation strings used across many business pages
 *  (quick-tools, advanced-tools, custom-model, etc). Pages can spread
 *  these into their own STRINGS dict instead of redeclaring every label.
 *
 *  Usage:
 *    import { COMMON_STRINGS } from "../lib/commonStrings";
 *    const STRINGS = {
 *      en: { ...COMMON_STRINGS.en, title: "Foo" },
 *      zh: { ...COMMON_STRINGS.zh, title: "Foo" }
 *    };
 */
export const COMMON_STRINGS = {
  en: {
    // sections
    taskConfig: "Task Configuration",
    dataInput: "Data Input",
    aiExpert: "AI Expert (Optional)",
    result: "Result",

    // inputs
    pasteSequence: "Paste Sequence",
    pasteSeqPlaceholder: "Paste FASTA content (must include >header)...",
    selectFile: "Select File",
    fromWorkspace: "From Workspace",
    useExample: "Use Example",
    useExampleFasta: "Use Example FASTA",
    uploaded: "Uploaded:",
    selectProteinFunction: "Select Protein Function",

    // AI panel
    enableAi: "Enable AI analysis and expert summary",
    enabled: "Enabled",
    disabled: "Disabled",
    llmProvider: "LLM Provider",
    aiOn: "AI expert interpretation will be generated together with prediction output.",
    aiOff: "Turn on to generate an expert summary after prediction finishes.",

    // actions / status
    startPrediction: "Start Prediction",
    runningBtn: "Running...",
    preparingTask: "Preparing task...",
    predictionDone: "Prediction completed",
    idle: "Idle",

    // errors
    runFailed: "Run failed.",
    uploadFailed: "Upload failed.",
    loadExampleFailed: "Failed to load example.",
    pleaseProvideInput: "Please upload a FASTA/PDB file or paste sequence.",

    // misc
    singleProteinNote: "Supports one protein per run (sequence or structure)."
  },
  zh: {
    // sections
    taskConfig: "任务配置",
    dataInput: "数据输入",
    aiExpert: "AI 专家（可选）",
    result: "结果",

    // inputs
    pasteSequence: "粘贴序列",
    pasteSeqPlaceholder: "粘贴 FASTA 内容（必须包含 > 开头的 header）…",
    selectFile: "选择文件",
    fromWorkspace: "从工作区",
    useExample: "使用示例",
    useExampleFasta: "使用示例 FASTA",
    uploaded: "已上传：",
    selectProteinFunction: "选择蛋白功能",

    // AI panel
    enableAi: "启用 AI 分析与专家解读",
    enabled: "已启用",
    disabled: "未启用",
    llmProvider: "大模型提供商",
    aiOn: "AI 专家解读将与预测结果一同生成。",
    aiOff: "开启后，预测完成会自动生成专家解读。",

    // actions / status
    startPrediction: "开始预测",
    runningBtn: "运行中…",
    preparingTask: "准备任务中…",
    predictionDone: "预测完成",
    idle: "空闲",

    // errors
    runFailed: "运行失败。",
    uploadFailed: "上传失败。",
    loadExampleFailed: "加载示例失败。",
    pleaseProvideInput: "请上传 FASTA / PDB 文件或粘贴序列。",

    // misc
    singleProteinNote: "每次运行支持一条蛋白（序列或结构）。"
  }
};
