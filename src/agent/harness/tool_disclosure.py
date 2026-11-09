"""Bounded discovery index; schemas are disclosed only for a selected tool.

This is an MCP transport surface, not a second execution backend. Resolved
calls must still pass the host's normal policy, argument and queue checks.
"""
from __future__ import annotations

import re
from mcp_tools.tools.discovery.matching import query_groups, relevance
from .capability_boundary import CAPABILITY_BOUNDARY


SEARCH = 'search_tools'
DESCRIBE = 'describe_tool'
CALL = 'call_tool'
CONTROLS = frozenset({SEARCH, DESCRIBE, CALL})
GUIDANCE = CAPABILITY_BOUNDARY + '\n\n' + (
    'Discover progressively: categories first; then search_tools(category=ID, query=task_or_model) '
    'for names and brief descriptions; then describe_tool(name) only for a selected capability. '
    'call_tool uses that declared interface. Use matching_models arguments for a specifically requested model. '
    'search_tools, describe_tool and call_tool are already exposed; call these navigation functions directly. '
    'Available session file operations are also directly exposed; invoke them by name to read, write and run code. '
    'The initial category overview contains no individual models or skills. Summarize it for general '
    'capability questions in a few sentences; avoid tool counts or an exhaustive category list. '
    'A specific task normally needs one targeted search. '
    'For help or workflows choose guides; read the selected manual section or skill only when needed. '
    'Discovery results are pointers, not retrieved evidence: only claim to have read a manual, skill '
    'or database record after calling its reader. Treat guidance as reference, not permission. '
    'Follow up using earlier tool observations, '
    'refreshing availability before execution. Ask for missing scientific inputs, never server keys. '
    'An empty search is not proof the platform lacks the capability; try another relevant category or query. '
    'Explain user-facing requirements concisely; omit raw tool IDs, server configuration and optional '
    'parameters unless requested. Only enabled tools are discoverable. '
    'ask_user and request_confirmation remain directly available.'
)

class DisclosureError(ValueError):
    """A safe, actionable message that contains no host or credential details."""


def brief_description(value, limit=180):
    text = str(value or '').strip()
    # Tool docstrings often embed full argument/reference sections.
    text = re.split(r'\n\s*\n|\n\s*(?:Args|Arguments|Parameters|Returns|Examples?)\s*:', text, maxsplit=1)[0]
    text = re.sub(r'\s+', ' ', text).strip()
    first = re.split(r'(?<=[.!?。])\s+', text, maxsplit=1)[0]
    return first if len(first) <= limit else first[:limit-1].rstrip() + '…'


def _schema(properties, required=()):
    return {'type': 'object', 'properties': properties, 'required': list(required), 'additionalProperties': False}


class ToolDisclosure:
    def __init__(self, mapping, *, schema, allowed, direct=(), detailed=()):
        if CONTROLS & mapping.keys():
            raise ValueError('Tool name conflicts with discovery transport')
        self.mapping = mapping
        self.schema = schema
        self.allowed = allowed
        self.direct = set(direct) & mapping.keys()
        self.detailed = set(detailed)
        if len(self.detailed) > 3 or not self.detailed <= self.direct:
            raise ValueError("Detailed disclosure requires at most three admitted direct tools")
        self.inspected = set()
        self._model_matches = {}
        self._model_selectors = {}
        self._navigation_queries = {}


    def _categories(self, name):
        from mcp_tools.tools.capability_policy import canonical
        from mcp_tools.tools.discovery.categories import tool_categories
        tool = self.mapping[name]
        return tool_categories(canonical(getattr(tool, 'name', name)), tool)

    def overview(self, names=None):
        """Only category labels are disclosed before a category is selected."""
        from mcp_tools.tools.discovery.categories import CATEGORIES
        names = [n for n in (self.mapping if names is None else names)
                 if n not in self.direct and self._available(n)]
        counts = {}
        for name in names:
            for category in self._categories(name):
                counts[category] = counts.get(category, 0) + 1
        return {'level': 'categories',
                'categories': [{'id': key, 'name': title, 'description': description,
                                'tool_count': counts[key]}
                               for key, (title, description) in CATEGORIES.items() if counts.get(key)],
                'tools': [], 'total': len(names), 'next_offset': None,
                'next_step': 'Select a category ID, then search_tools(category=ID, query=task_or_model). '
                             'For a broad introduction, summarize these categories directly.'}

    @property
    def names(self):
        return CONTROLS | self.direct

    def _name(self, name):
        # Harnesses sometimes repeat our MCP transport namespace inside the
        # discovery arguments. Normalize only this server, then apply the same
        # catalog, disclosure and enablement checks.
        for prefix in ('functions.mcp__venusfactory__', 'mcp__venusfactory__'):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        if name in self.mapping:
            return name
        from mcp_tools.tools.capability_policy import canonical
        matches = [key for key, tool in self.mapping.items()
                   if canonical(getattr(tool, 'name', key)) == canonical(name)]
        return matches[0] if len(matches) == 1 else name

    def _available(self, name):
        return name in self.mapping and bool(self.allowed(self.mapping[name]))

    def surface(self):
        name = {'type': 'string', 'description': 'Exact name returned by search_tools.'}
        result = [
            {'name': SEARCH, 'description': 'List categories, or search one category for enabled capability names and brief descriptions.',
             'inputSchema': _schema({'category': {'type': 'string', 'default': '', 'description': 'Category ID from the initial overview; omit to list categories.',
                 'enum': ['', *[row['id'] for row in self.overview()['categories']]]}, 'query': {'type': 'string', 'default': '', 'maxLength': 200},
                                    'limit': {'type': 'integer', 'default': 8, 'minimum': 1, 'maximum': 20},
                                    'offset': {'type': 'integer', 'default': 0, 'minimum': 0}})},
            {'name': DESCRIBE, 'description': 'Read one tool’s full input contract before calling it.',
             'inputSchema': _schema({'name': name}, ['name'])},
            {'name': CALL, 'description': 'Execute after describe_tool(name), or use an already exposed direct tool. Supply its inputSchema arguments. Uses normal session permissions and queue.',
             'inputSchema': _schema({'name': name, 'arguments': {'type': 'object', 'additionalProperties': True}}, ['name', 'arguments'])},
        ]
        result += [{'name': name, 'description': (self.mapping[name].description if name in self.detailed else brief_description(self.mapping[name].description)),
                    'inputSchema': self.schema(self.mapping[name])} for name in sorted(self.direct) if self._available(name)]
        return result

    def search(self, arguments):
        from mcp_tools.tools.discovery.categories import CATEGORIES, category_id
        if set(arguments) - {'category', 'query', 'limit', 'offset'}:
            raise DisclosureError('search_tools accepts category, query, limit and offset only.')
        query, limit, offset = arguments.get('query', ''), arguments.get('limit', 8), arguments.get('offset', 0)
        if not isinstance(query, str) or len(query) > 200 or type(limit) is not int or not 1 <= limit <= 20 or type(offset) is not int or offset < 0:
            raise DisclosureError('Use query up to 200 characters, limit 1-20, and nonnegative offset.')
        # Search hints belong to this query; inspected interfaces remain turn-scoped.
        self._model_matches.clear()
        self._navigation_queries.clear()
        try:
            category = category_id(arguments.get('category', ''))
        except ValueError as exc:
            raise DisclosureError(str(exc)) from exc
        if category and category not in CATEGORIES:
            if query.strip():
                # Keep the query useful even if a model guessed a category name.
                # Normal search still enforces visibility and progressive disclosure.
                result = self.search({key:value for key,value in arguments.items() if key != 'category'})
                result['requested_category'] = category
                return result
            # A misspelled/guessed facet is a navigation mistake, not a failed
            # scientific operation. Return only the current allowed categories
            # so smaller models can recover without repeating the same call.
            result = self.overview()
            result['selection_error'] = 'Unknown category.'
            result['next_step'] = 'Use an exact category ID from categories below, or omit category and search by query.'
            return result
        if not query.strip() and not category:
            return self.overview()
        candidates = [n for n in self.mapping if n not in self.direct and self._available(n)
                      and (not category or category in self._categories(n))]
        # If the caller already knows an exact name but selected a wrong facet,
        # return its category only; never disclose a disabled tool or its schema.
        if category and query.strip():
            exact = self._name(query.strip())
            if exact not in self.direct and self._available(exact) and exact not in candidates:
                result = self.overview([exact])
                result['selection_error'] = 'This named capability belongs to another category.'
                result['next_step'] = 'Select its category below, or inspect the known name with describe_tool(name).'
                return result
        known_name = self._name(query.strip())
        if known_name in self.mapping:
            query = known_name
        groups = query_groups(query)
        from mcp_tools.tools.discovery.platform_models import platform_models
        from mcp_tools.tools.capability_policy import canonical, ADAPTER_TOOLS
        native = {canonical(getattr(self.mapping[n], 'name', n)) for n in candidates}
        known = {v for values in ADAPTER_TOOLS.values() for v in values}
        models = platform_models(query, native) if query.strip() and native & known else []
        aliases = {}
        for _, _, card in models:
            for tool_name in card['tools']:
                aliases.setdefault(tool_name, []).append(card['name'])
        for tool_name in aliases:
            self._model_matches[tool_name] = [card for _, _, card in models if tool_name in card['tools']][:6]
        matches, partial = [], []
        for name in candidates:
            tool = self.mapping[name]
            description = brief_description(tool.description)
            title = getattr(tool, 'name', name)
            # Search uses just the public name and summary, never schemas or skill bodies.
            score = relevance(groups, title.replace('_', ' '), f"{name} {title} {title.replace('_', ' ')} {description}")
            model_names = aliases.get(canonical(title), [])
            if model_names:
                score = max(score or 0, 10)
                description = brief_description(', '.join(model_names[:4]) + ' - ' + description)
            if score is None and category and len(groups) > 1:
                hits = sum(relevance([group], title, f"{name} {title} {description}") is not None
                           for group in groups)
                if hits >= max(2, (len(groups) + 1) // 2):
                    partial.append((False, hits, name, description))
            if score is not None:
                matches.append((name.casefold() == query.strip().casefold(), score, name, description))
        match_mode = 'name_and_summary'
        if not matches and partial:
            best = max(row[1] for row in partial)
            matches = [row for row in partial if row[1] == best]
            match_mode = 'partial_summary'
        if not matches and category:
            from mcp_tools.tools.discovery.categories import ROUTERS
            matches = [(False, 0, name, brief_description(self.mapping[name].description))
                       for name in candidates
                       if canonical(getattr(self.mapping[name], 'name', name)) in ROUTERS.get(category, set())]
            if matches:
                match_mode = 'category_navigation'
        matches.sort(key=lambda row: (-row[0], -row[1], row[2]))
        if not category:
            categories = {c for _, _, n, _ in matches for c in self._categories(n)}
            if len(categories) != 1:
                result = self.overview([n for _, _, n, _ in matches] if matches else None)
                # A category-only response must not look like an empty tool
                # search: small models interpreted tools=[] as unavailability.
                result.pop('tools', None)
                result['query'] = query
                result['matched_capabilities'] = len(matches)
                result['match_status'] = 'matches_found' if matches else 'no_direct_match'
                result['summary'] = (
                    f'Found {len(matches)} enabled capabilities matching {query!r}. '
                    'Their names and interfaces have not been loaded yet.'
                    if matches else
                    'No direct name or summary match. Browse a relevant category; '
                    'this does not establish that the requested capability is unavailable.')
                result['suggested_searches'] = [
                    {'category': row['id'], 'query': query}
                    for row in result['categories'][:6]]
                result['next_step'] = (
                    'Call search_tools with a suggested category and the SAME query '
                    'to load matching names before answering availability.'
                    if matches else
                    'Choose a relevant category and broaden the query.')
                return result
            category = next(iter(categories))
        page = matches[offset:offset+limit]
        result = {'level': 'capabilities', 'category': category, 'match_mode': match_mode,
                'tools': [{'name': name, 'description': desc} for _, _, name, desc in page],
                'total': len(matches),
                'next_offset': offset+len(page) if offset+len(page) < len(matches) else None,
                'next_step': ('No name or summary matched. These are category search entry points; '
                              'inspect one and pass your topic to it. No underlying records or guides were read.'
                              if match_mode == 'category_navigation' else
                              'Inspect a relevant capability with describe_tool(name) before calling it. '
                              'If no match, broaden the query or choose another category.')}

        if match_mode == 'category_navigation' and page:
            # A reader is a pointer to records, never evidence that a model is
            # absent. Keep its query until the selected schema is disclosed.
            self._navigation_queries.update({name: query for _, _, name, _ in page})
            result.update(
                records_searched=False,
                next_action={'name': DESCRIBE, 'arguments': {'name': page[0][2]}},
                next_step='Call next_action, then call_tool with the selected reader and its declared arguments. '
                          'No model, database or guide records have been searched; repeating search_tools repeats only this index.',
            )
        return result

    def metadata(self, method, arguments):
        if not isinstance(arguments, dict):
            raise DisclosureError('Use a JSON object for tool arguments.')
        if method == SEARCH:
            return self.search(arguments)
        if method == DESCRIBE:
            if set(arguments) != {'name'} or not isinstance(arguments['name'], str):
                raise DisclosureError('describe_tool requires one tool name.')
            name = self._name(arguments['name'])
            if name in CONTROLS:
                exposed = next(tool for tool in self.surface() if tool['name'] == name)
                return {**exposed, 'call_via':name,
                        'next_step':'This navigation tool is already exposed. Call it directly with these arguments.'}
            if not self._available(name):
                raise DisclosureError('Tool unavailable. Search for an enabled tool.')
            tool = self.mapping[name]
            result = {'name': name, 'description': tool.description,
                      'call_via': name if name in self.direct else 'call_tool',
                      'next_step': ('This tool is directly exposed; call it by name with its arguments.' if name in self.direct else 'Use call_tool(name, arguments). This capability is not a directly exposed native function.'),
                      'inputSchema': self.schema(tool)}
            query_field = result['inputSchema'].get('properties', {}).get('query', {})
            if name in self._navigation_queries and query_field.get('type') == 'string':
                result['suggested_arguments'] = {'query': self._navigation_queries[name]}
            from mcp_tools.tools.capability_policy import canonical
            from mcp_tools.tools.discovery.platform_models import describe_model_choices
            tool_name = getattr(tool, 'name', name)
            model_details, selectors = describe_model_choices(
                tool_name, result['inputSchema'], self._model_matches.get(canonical(tool_name), [])
            )
            result.update(model_details)
            if model_details.get('matching_models'):
                result['description'] = re.split(r'\n\s*\n|\n\s*(?:Args|Arguments|Parameters|Returns)\s*:', str(tool.description), maxsplit=1)[0].strip()[:600]
                self._model_selectors[name] = selectors
            self.inspected.add(name)
            return result
        raise DisclosureError('Unknown discovery operation.')

    def resolve(self, method, arguments):
        if method in self.direct:
            name = method
        elif method == CALL:
            if not isinstance(arguments, dict) or set(arguments) != {'name', 'arguments'} or not isinstance(arguments['name'], str) or not isinstance(arguments['arguments'], dict):
                raise DisclosureError('call_tool requires a name and an arguments object.')
            name, arguments = self._name(arguments['name']), arguments['arguments']
            if not self._available(name):
                raise DisclosureError('Tool unavailable. Search for an enabled tool; do not guess its name.')
            if name not in self.inspected and name not in self.direct:
                raise DisclosureError('Read describe_tool for this tool before calling it.')
        else:
            raise DisclosureError('Use search_tools, describe_tool, then call_tool.')
        if not self._available(name):
            raise DisclosureError('Tool unavailable. Search for an enabled tool.')
        if not isinstance(arguments, dict):
            raise DisclosureError('Use a JSON object for tool arguments.')
        for key in self._model_selectors.get(name, ()):
            if not arguments.get(key):
                raise DisclosureError('Specify the requested model explicitly using matching_models from describe_tool; do not use a generic default.')
        return name, self.mapping[name], arguments
