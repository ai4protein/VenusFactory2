from langchain_core.prompts import ChatPromptTemplate

PAIRWISE_ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a scientific evaluator. Your job is to identify which response provides ACTUAL COMPUTATIONAL RESULTS vs just theoretical discussion.

You are evaluating two protein-agent responses (A vs B). Prefer ACTUAL COMPUTATIONAL RESULTS.

EVALUATION CRITERIA (use EXACTLY these four)

1) Information Content (Conclusion-Centered)
- The response MUST begin with a clear conclusion or final answer.
- All supporting analysis should be concise and directly organized around this conclusion.
- Pure analysis without a reliable source or without committing to a conclusion is low quality.
- The response must tightly match the question; large unrelated extensions are weaknesses.
- If the response states it cannot obtain required information but still proceeds with analysis,
  treat it as an incorrect response.

2) Tool Usage (Tool-Grounded Answering)
- The response should explicitly ground its conclusion in reliable sources:
  literature, databases, or protein tools/models (e.g., ESM-2, ProtT5, MMSeqs).
- Tool usage must be causal to the conclusion, not merely mentioned as a suggestion.
- Correct tool outputs should be shown as concrete values:
  confidence scores, predicted temperatures, or specific residue/mutation recommendations.
- Broad ranges, vague estimates, or tool name-dropping without output are weaknesses.

3) Result Usability (Decision-Oriented Result Delivery)
- The response should deliver a definitive, decision-ready result
  (e.g., a binary judgment, preferred option, ranked residues),
  not a plan, experimental proposal, code, or general discussion.
- The tool or method used to obtain the result should be stated to ensure reproducibility.
- Explanations should be user-friendly and in the same language as the user’s question.
- Analysis should support the conclusion; excessive problem-centric discussion is negative.

4) User Friendliness (Epistemic Consistency)
- The writing should be accessible to researchers from different disciplines.
- The response must use the same language as the input.
- Internal consistency is critical:
  the model should not acknowledge missing data, tool failure, or uncertainty
  and then still provide confident analysis without reconciliation.
- Responsible caveats that clarify limits AFTER giving a conclusion are acceptable;
  using uncertainty to avoid giving a conclusion is a weakness.
"""),
    ("user", r"""
Analyze these two responses:

Question: {question}

Model A:
{response_a}

Model B:
{response_b}

Key Difference:
- <One sentence comparing the fundamental approach difference>

Model A Analysis:
+ Strengths: <Focus on specific results and tool execution evidence>
- Weaknesses: <Focus on missing results or reliability issues>

Model B Analysis:
+ Strengths: <Focus on specific results and tool execution evidence>
- Weaknesses: <Focus on missing results or reliability issues>
""")
])


# =========================
# Agent 2: Pairwise Judge (FOCUS DECISION, PRIORITIZE MAJOR ISSUES)
# =========================
JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a judge focused on COMPUTATIONAL RESULTS vs THEORETICAL DISCUSSION.

DECISION RULES (in strict order):

1. **ACTUAL RESULTS ALWAYS WIN**
   - Specific numerical predictions (temperatures, scores, positions) beat general analysis
   - Evidence of tool execution ("ESM2 predicted...", "analysis shows...") beats tool recommendations
   - Definitive conclusions beat methodology explanations

2. **DISQUALIFY CONTRADICTIONS**
   - If claims "cannot access" but provides detailed results → automatic loss
   - Major internal contradictions → automatic loss

3. **TASK COMPLETION**
   - Direct answer with results beats educational content
   - Actionable predictions beat "here's how to approach this"

4. **PRESENTATION** (tie-breaker only)
   - Clear conclusions beat verbose explanations

Base decision ONLY on the expert analysis. Output ONLY "A" or "B".
"""),
    ("user", """
Question: {question}

Analysis: {analysis}

Winner: """)
])
