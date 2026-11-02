export type UserInteraction = {
  version: 1; id: string; kind: "question" | "confirmation" | "plan";
  title: string; message: string; created_at: number;
  questions: Array<{ id: string; question: string; options: string[]; multiple: boolean; allow_text: boolean }>;
};
export type UserResponse = {
  interaction_id: string; decision: "answer" | "approve" | "reject" | "revise" | "cancel";
  answers: Array<{ question_id: string; selected: number[]; text: string }>;
  feedback: string;
};
export type InteractionDraft = { answers: Record<string, { selected: number[]; text: string }>; feedback: string };
export function canAnswer(interaction: UserInteraction, draft: InteractionDraft): boolean {
  return interaction.questions.every(q => {
    const a = draft.answers[q.id];
    return !!a && (a.selected.length > 0 || (q.allow_text && !!a.text.trim()));
  });
}
export function makeResponse(interaction: UserInteraction, draft: InteractionDraft, decision: UserResponse["decision"]): UserResponse {
  return { interaction_id: interaction.id, decision, feedback: draft.feedback.trim(),
    answers: decision === "answer" ? interaction.questions.map(q => ({ question_id: q.id, selected: draft.answers[q.id]?.selected || [], text: draft.answers[q.id]?.text.trim() || "" })) : [] };
}
