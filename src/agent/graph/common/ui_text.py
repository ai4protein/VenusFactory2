"""i18n UI string bundle for graph nodes."""
from __future__ import annotations


def _ui_text(lang: str, key: str, **kwargs) -> str:
    zh = {
        "pipeline_title": "📋 **Pipeline**\n\n以下是执行计划：\n\n{steps}",
        "executing_step": "⏳ **Machine Learning Specialist** 正在执行第 {step_num} 步：{task_desc} …",
        "step_failed": "❌ **第 {step_num} 步失败。**\n\n",
        "step_done": "✅ **第 {step_num} 步完成。**\n\n",
        "summary": "摘要：",
        "output_preview": "输出预览：",
        "raw_output": "**原始输出（调试/自检）：**",
        "cloud_download": "📎 **云端下载：** [{name}]({url})",
        "file_preview": "**文件预览（{name}）：**",
        "generated_image": "🖼️ **生成图片：** [{name}]({url})",
        "pipeline_paused": "⛔ **流水线已暂停：** 此步骤返回错误信号（如 `status:error` / `success:false`），已跳过后续步骤以避免级联失败。\n\n",
        "summarizing": "📝 **Scientific Critic** 正在汇总结果 …",
        "final_summary_title": "## 总结\n",
        "pipeline_paused_at": "流水线在第 {step} 步暂停：{reason}\n",
        "task_completed": "任务已完成，执行情况如下：\n",
        "tool_failed": "- **{tool}**：失败（{reason}）",
        "tool_executed": "- **{tool}**：已执行",
        "task_ended": "任务已结束。详见上方结果。",
        "sub_report_failed": "(小报告生成失败：{section}，错误：{error})",
        "clarification_title": "🤔 **Principal Investigator** 在开始研究之前，有几个问题需要确认：",
        "plan_confirmation_title": "📋 **Computational Biologist** 已制定以下执行计划，请确认或编辑后再执行：\n\n{steps}",
        "iteration_prompt": "🔄 **Scientific Critic** 结果已汇总完毕。请选择下一步操作：",
        "step_checkpoint": "🔍 **第 {step_num} 步已完成。** 请查看上方结果，并决定是否继续执行下一步（第 {next_step} 步：{next_desc}）。",
        "sub_report_checkpoint": (
            "📄 **「{section}」小报告完成** · 剩余 {remaining} 节。"
            "请在下方选择继续、修改或跳过至总报告。"
        ),
        "step_skipped": "⚠️ **第 {step_num} 步失败，但后续步骤不依赖此步骤，已跳过继续执行。**\n\n",
    }
    en = {
        "pipeline_title": "📋 **Pipeline**\n\nHere's what we'll do:\n\n{steps}",
        "executing_step": "⏳ **Machine Learning Specialist** is executing Step {step_num}: {task_desc} …",
        "step_failed": "❌ **Step {step_num} failed.**\n\n",
        "step_done": "✅ **Step {step_num} Complete.**\n\n",
        "summary": "Summary: ",
        "output_preview": "Output Preview: ",
        "raw_output": "**Raw output (for debugging/self-check):**",
        "cloud_download": "📎 **Cloud Download:** [{name}]({url})",
        "file_preview": "**File Preview ({name}):**",
        "generated_image": "🖼️ **Generated Image:** [{name}]({url})",
        "pipeline_paused": "⛔ **Pipeline paused:** This step returned an error signal (e.g. `status:error` / `success:false`). Downstream steps were skipped to avoid cascading failures.\n\n",
        "summarizing": "📝 **Scientific Critic** is summarizing the results …",
        "final_summary_title": "## Summary\n",
        "pipeline_paused_at": "Pipeline paused at Step {step}: {reason}\n",
        "task_completed": "Task completed. Here's what was done:\n",
        "tool_failed": "- **{tool}**: failed ({reason})",
        "tool_executed": "- **{tool}**: executed",
        "task_ended": "Task ended. See results above.",
        "sub_report_failed": "(Sub-report failed for {section}: {error})",
        "clarification_title": "🤔 **Principal Investigator** Before starting the research, I have a few questions to clarify your needs:",
        "plan_confirmation_title": "📋 **Computational Biologist** has designed the following pipeline. Please review and confirm before execution:\n\n{steps}",
        "iteration_prompt": "🔄 **Scientific Critic** Results have been summarized. Please choose what to do next:",
        "step_checkpoint": "🔍 **Step {step_num} complete.** Review the results above and decide whether to continue to the next step (Step {next_step}: {next_desc}).",
        "sub_report_checkpoint": (
            "📄 **Sub-report \"{section}\" complete.** {remaining} section(s) remaining. "
            "Choose Continue, Revise, or Skip to report below."
        ),
        "step_skipped": "⚠️ **Step {step_num} failed, but no downstream steps depend on it. Skipping and continuing.**\n\n",
    }
    bundle = zh if lang == "zh" else en
    template = bundle.get(key, key)
    return template.format(**kwargs)
