from agent.harness.web_native_worker import web_tool_name


def test_kimi_long_tools_have_distinct_stable_runtime_names():
    names = [
        'download_brenda_compare_organisms_by_ec_number',
        'download_brenda_environmental_parameters_by_ec_number',
        'download_brenda_compare_organisms_by_gene_name',
        'calculate_physchem_from_sequence',
    ]
    aliases = [web_tool_name(n, 'kimi-code') for n in names]
    assert len(set(aliases)) == len(names)
    assert all(len('mcp__venusfactory__' + n) <= 64 for n in aliases)
    assert aliases == [web_tool_name(n, 'kimi-code') for n in names]
    assert aliases[-1] == names[-1]
    assert [web_tool_name(n, 'deepseek-harness') for n in names] == names
