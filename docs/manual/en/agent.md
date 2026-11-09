# Agent: answer and act

## Ask a useful question

Include the protein identifier or sequence, species, the property of interest and the output you need. Add relevant files. For example:

> Find UniProt annotations for human TP53, summarize its domains, and cite the records you use.

> Read my sequences.csv, calculate sequence properties, and save a CSV. Ask me if the sequence column is ambiguous.

Agent can search available tools, retrieve web evidence, query protein databases and run enabled prediction tools. For platform questions such as “How do I train with my data?”, it searches the platform Manual rather than assuming an older interface.

## Choose Model and Harness

Use the selectors next to the composer. The available choices depend on deployment configuration and credentials. Online mode offers configured free models. Harnesses can include Kimi Code, Codex, Claude Code and DeepSeek Harness; choose an available combination. A harness name does not specify the protein model used for a prediction.

## Follow a conversation

Tool activity appears within the conversation. Running sessions have an indicator in the history. You can switch modes or conversations while a submitted job continues. Return through the history or the session URL. Stop is an explicit action; changing the page is not a stop request.

Pin frequently used conversations; hover over a history item to see its time. The context indicator near the composer shows context use. Long histories may be compacted: restate important constraints if they are missing from a continuation.

## Code and files

For custom analysis, Agent writes source files and runs them in the session sandbox. Only that conversation's workspace is exposed to the program. Network searches and downloads use registered tools. Open outputs from the file icon; supported structures open in the structure viewer.

Do not paste API keys into a conversation. Deployment credentials are managed in Administration.

## Reuse Scientist work

Ask for a tool by protein, property or experimental scenario. Agent searches the registered catalog and inspects a matching model's interface and scope before running it. A research output becomes a shared tool only after independent review and publication, and it must be enabled in the current deployment.

Registered Scientist model predictions appear in a result card with their units. Expand **Applicability** to check the supported conditions. Open **View report**, **CSV** or **JSON** to preview and download the complete result; long tables show a short preview in the conversation.

See [Scientist](manual:scientist) to develop a capability and [Library](manual:library) to manage results.
