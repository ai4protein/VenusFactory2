"""Shared purpose of a Scientist topic, independent of its UI or harness."""

SCIENTIST_PURPOSE = (
    'Scientist is the RSI research workflow. Starting a topic starts an ongoing research project, '
    'not a question-answer turn or a fixed reproduction template. A short goal is sufficient. '
    'Use the goal, supplied resources and measured feedback to develop, test and revise a new '
    'model, method or reusable capability. Inspecting data or summarizing a paper is an intermediate '
    'milestone when the requested outcome is a new model. Keep that outcome through replanning. '
    'Do not require the user to specify a learning algorithm or a detailed engineering prompt. '
    'Do not assume that uploaded papers describe the complete study, that supplied measurements '
    'are sufficient, or that a sophisticated algorithm will outperform a simple baseline. '
)


def topic_context():
    from agent.rsi.artifacts.contract import guide
    return {
        'artifact_contract': guide(),
        'workflow': 'Scientist / RSI',
        'entry': 'One natural-language topic; optional wet-lab tables, sequences, structures, papers or notes.',
        'materials': {
            'experimental_data': 'Inspect actual columns, units, assay conditions, replicates, batches and missing values. '
                'Choose splits that match the scientific claim, keeping related variants, repeated measurements '
                'and held-out data separate as appropriate. Do not fabricate unavailable measurements.',
            'partial_papers': 'Read the supplied content. Distinguish reported evidence, unresolved questions and '
                'your proposed extension. Retrieve missing public details when possible; do not imply a full '
                'reproduction from an excerpt or invent unavailable experiments.',
            'prospective_models': 'Develop predictions or experiment-prioritization models with a declared scope '
                'and uncertainty. Separate retrospective validation from prospective wet-lab validation. '
                'Predicted candidates are hypotheses until measured.',
            'supplementary_models': 'Address a specific gap or limitation of the supplied study using its actual '
                'data and compatible public resources. Preserve provenance and state what was not reproduced.',
        },
        'cycle': ['PI direction', 'Plan', 'Implement and measure', 'Critic feedback',
                  'Revise and repeat', 'Independent Review', 'Deliver and register if admitted'],
        'model_outputs': ['reproducible source and fitted artifacts', 'measured baseline and revised generation',
                          'matched validation and final test report', 'prediction interface and model card',
                          'project README and downloadable package'],
        'completion': 'Use validated artifacts and the requested scientific outcome, not the number of iterations. '
            'A negative revision can justify retaining the baseline. Public registration remains an independent gate.',
    }
