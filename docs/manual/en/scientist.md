# Scientist: start an RSI research topic

Scientist is the platform's RSI workflow. Sending a research goal starts a topic that continues through planning, implementation, experiments, criticism and revision. The web interface shows this same process and its outputs.

## Bring a goal and your materials

A short description is enough. You can start from a research idea, wet-lab measurements, a paper or an excerpt. Files are optional; supply them when they help define the target.

Examples:

- “Develop an enzyme activity predictor from these measurements.”
- “Use this paper and my data to develop a complementary prediction model.”
- “Build a small-data model to prioritize the next variants to test.”

Attach tables, sequences, structures, papers or notes together. If known, describe measurement units, assay conditions, batches, replicates and samples reserved for testing. Scientist inspects the resources and proposes a feasible strategy; you do not need to choose an algorithm, task template or a predefined protein case.

Paper-based work can reproduce a defined part of a study, address a limitation or develop a complementary model. The plan distinguishes what the supplied material establishes from the new hypothesis. Prospective predictions remain candidates for wet-lab testing; retrospective validation does not establish experimental success.

## Start and guide the research

1. Select Scientist, enter the goal and attach any materials. Choose an available model/harness and a budget.
2. Send the goal to start the RSI topic. Follow **PI → Plan → Implement → Critic**, with revised plans and repeated experiments as needed.
3. Use the revision controls or composer to add constraints, correct assumptions or supply new data in the same topic.
4. Inspect the plan, intermediate results, model iterations, activity and project files. Model iterations show measured baselines, revisions, validation scores and their parent experiments.
5. Review the final project and test report. Independent **Review** checks reusable candidates before admission.

Enter submits; Shift+Enter adds a line. A topic retains its own context when you switch modes. If it asks for essential missing information, reply in the same topic. A budget limit is a checkpoint, not evidence that research has finished.

## Develop and evaluate models

Scientist can implement statistical models, custom neural models, few-shot adaptation, semi-supervised learning, meta-learning or other justified methods. It may use enabled training MCP tools or develop a custom model in the session sandbox. PLM training is not required; online PLM training is restricted to ESM2-8M and all work follows deployment budgets and tool availability.

Model research requires a measured baseline and a revised experiment on matched validation data. The system can retain a baseline when a revision does not help. Advanced method names, repeated iterations or a better validation score alone do not establish novelty, statistical significance or biological effectiveness. Final testing and independent review are distinct from feedback used to revise a model.

## Receive and reuse the results

**Research outputs** shows the selected models or methods with their README, usage card, test report and download. Each release includes runnable code, a JSON interface, example input/output, interface tests and recorded dependencies. Learned models also include their fitted state and training code. Raw evaluation datasets stay in the research workspace.

Use **Download project (.zip)** for the selected releases and research report. Each release has its own README with use and reproduction commands. A draft without a reviewed release downloads the working project. Experiments, ablations and previous versions remain in the topic; expand their records when you need the details. Revisions of the same purpose appear as versions of one capability instead of filling the tool list with every trial.

Research completion and public-tool admission are separate. Code execution or Critic agreement alone does not publish a model. Independent checks review interfaces, scope, reproducibility and evaluation evidence. Rejected candidates remain available for revision. Only reviewed, published and enabled capabilities can be discovered and called by [Agent](manual:agent), Model Plaza or workflows.


## Share a model with the platform

Scientist outputs remain private in both local and online modes, even after independent Review passes. Download your project and test report for your own use. To request publication, choose **Request publication by email** and include your name, author email, scope, code/weights, test report and permission to share the data and model. Send the email to **tanyang.august@sjtu.edu.cn**. There is no in-app submission queue.

The platform operator selects and publishes reviewed models. Published entries display the author email and become discoverable by Agent, Model Plaza and Workflow when enabled. Research completion, technical review and operator publication are three separate decisions.
