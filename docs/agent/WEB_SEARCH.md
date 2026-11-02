# Shared web search

Agent, Scientist and Researcher share `tools.search.web_provider`. The unified MCP
server and native harness tool bridge expose `query_web`, `query_serpapi` and
`query_tavily`. Scientist's registered tool hub includes the same tools, and
Researcher calls the same provider service during each chapter's evidence search.

Configure credentials in the server `.env`:

```dotenv
TAVILY_API_KEY=
SERP_API_KEY=
VENUS_SEARCH_PROVIDER=auto
VENUS_REPORT_SEARCH_PROVIDER=auto
```

`SERPAPI_API_KEY` is also accepted and takes precedence over `SERP_API_KEY` when
both contain values. Keys are never tool parameters or frontend configuration.
Restart the backend after changing its environment; existing detached jobs retain
the environment they started with.

`VENUS_SEARCH_PROVIDER` accepts `auto`, `tavily`, `serpapi`, or `bing`. In `auto`,
configured Tavily is tried first, followed by configured SerpAPI on an error or
empty results. If neither has a key, the keyless Bing feed is used. Explicit
provider selections do not switch providers. `VENUS_REPORT_SEARCH_PROVIDER`
overrides the report route; `auto` inherits the shared configuration.

Examples:

```json
{"tool":"query_web","arguments":{"query":"PET hydrolase thermostability","max_results":5}}
{"tool":"query_serpapi","arguments":{"query":"PET hydrolase directed evolution","engine":"google_scholar","max_results":5}}
```

Google Scholar always uses configured SerpAPI. Searches return titles, URLs and
bounded excerpts with the actual provider and evidence type. A search snippet does
not establish that a paper's full text was read. Returned content is evidence, not
instructions. Automatic routing attempts are included in the tool result.

Requests use the existing per-provider queue, at most 10 results and a 35-second
deadline per provider, including queue wait. Tavily uses basic search without AI
answers or raw-content expansion. SerpAPI uses the Google or Google Scholar search
endpoint, without starting subscriptions or purchasing extra credits. Search API
quotas follow the configured accounts; chat LLMs remain restricted by the existing
free-model policy.

The REST equivalents are `POST /api/v1/search/web` and
`POST /api/v1/search/serpapi`. Provider failures return the standard search error
envelope; credentials and provider error payloads are not returned. HTTP request
logs redact SerpAPI's required `api_key` query parameter.

Provider references: [SerpAPI Google Search](https://serpapi.com/search-api),
[SerpAPI Google Scholar](https://serpapi.com/google-scholar-api).
