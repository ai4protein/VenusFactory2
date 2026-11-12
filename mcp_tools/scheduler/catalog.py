"""Functional taxonomy and admission policy, shared by discovery and execution."""
from collections import Counter

CATEGORIES = {
    'platform_help': ('Platform guides', '平台使用指南', 'cpu', 30),
    'web_search': ('Web & literature', '网络与文献检索', 'api:web-search', 180),
    'protein_database': ('Protein databases', '蛋白质与生物数据库', 'api:protein-database', 300),
    'remote_analysis': ('Remote analyses', '远程序列与结构分析', 'api:remote-analysis', 600),
    'sequence_analysis': ('Sequence calculations', '序列计算', 'cpu', 600),
    'structure_analysis': ('Structure calculations', '结构计算', 'cpu', 900),
    'molecular_docking': ('Molecular docking', '分子对接', 'cpu', 3600),
    'molecular_admet': ('Molecular ADMET', '小分子 ADMET', 'cpu', 600),
    'model_prediction': ('Model prediction & design', '模型预测与设计', 'gpu', 3600),
    'model_training': ('Model training & development', '模型训练与开发', 'gpu', 7200),
    'workspace': ('Files & capability management', '文件与能力管理', 'cpu', 180),
    'sandbox_execution': ('Session sandbox', '会话沙箱执行', 'cpu', 300),
}
INLINE = {'query_platform_manual', 'search_research_tools', 'list_evolved_models', 'describe_evolved_model', 'list_trained_models',
          'search_generated_capabilities', 'describe_generated_capability', 'read_skill', 'list_skills'}
REMOTE = {'query_interproscan', 'query_ncbi_blast', 'query_foldseek_remote', 'query_public_protein_service',
          'download_ncbi_blast', 'download_blast_homologs_by_sequence', 'download_mmseqs2_homologs_by_sequence',
          'download_clustalo_msa_by_fasta'}
WEB = {'query_europepmc','query_arxiv','query_biorxiv','query_duckduckgo','query_fda','query_github','query_hugging_face','query_pubmed',
       'query_semantic_scholar','query_tavily','query_web','query_serpapi','download_openalex_entries_by_query',
       'download_openalex_entry_by_id','download_arxiv_paper_by_id','download_biorxiv_by_doi'}
STRUCTURE = {'calculate_rsa_from_pdb','calculate_sasa_from_pdb','calculate_ss_from_pdb','render_protein_structure',
             'superpose_two_structures','analyze_alphafold_plddt_by_metadata_file','analyze_alphafold_pae_by_pae_file',
             'pdb_chain_sequences','get_seq_from_pdb_chain_a','pdb_dir_to_fasta','check_pdb_apo','maxit_structure_convert'}
PREDICTION = {'predict_protein_function','predict_residue_function','predict_structure_esmfold','protein_model_predict',
              'predict_with_evolved_model','run_generated_capability','zero_shot_mutation_sequence_prediction',
              'zero_shot_mutation_structure_prediction','proteinmpnn_sequence_design_from_structure','proteinmpnn_sequence_scoring_from_structure'}


def classify(name, metadata=None):
    """Plugins may explicitly declare tool_category, queue_resource, timeout_seconds."""
    meta = metadata or {}
    if meta.get('tool_category'):
        category = meta['tool_category']
    elif name == 'query_platform_manual':
        category = 'platform_help'
    elif name in {'run_session_python','run_session_shell','run_research_python'}:
        category = 'sandbox_execution'
    elif name == 'predict_admet':
        category = 'molecular_admet'
    elif name in WEB or name.startswith(('query_literature','query_dataset','query_web_by','download_openalex','download_arxiv','download_pubmed','download_biorxiv')):
        category = 'web_search'
    elif name in REMOTE or name.startswith(('download_clustalo','download_mmseqs2','download_blast')):
        category = 'remote_analysis'
    elif name in STRUCTURE or name.startswith(('calculate_rsa','calculate_sasa','calculate_ss','analyze_alphafold')):
        category = 'structure_analysis'
    elif name in PREDICTION or 'foldseek' in name or 'venusmine' in name or name.startswith('proteinmpnn_'):
        category = 'model_prediction'
    elif name in {'train_protein_model','generate_training_config'}:
        category = 'model_training'
    elif name=='predict_s4pred':
        category = 'sequence_analysis'
    elif name.startswith(('calculate_physchem','translate_ncbi','read_fasta')):
        category = 'sequence_analysis'
    elif name in {'query_ncbi','query_gget','query_opentargets','query_ensembl','query_pubchem','query_kegg'} or name.startswith(('download_','query_sequence_','query_uniprot','search_ncbi','search_uniprot')):
        category = 'protein_database'
    else:
        category = 'workspace'
    if category not in CATEGORIES:
        raise ValueError('Unknown tool category: '+str(category))
    en, zh, resource, timeout = CATEGORIES[category]
    if name in {'query_gget','query_opentargets','query_ensembl','query_europepmc','query_pubchem','query_kegg'}:resource='api:'+name.removeprefix('query_')
    if name in {'query_ncbi','query_ncbi_blast'} or name.startswith(('download_ncbi','search_ncbi')):resource='api:ncbi'
    # Registered numerical Scientist models use the CPU sandbox, not a PLM/GPU runtime.
    if name in {'generate_training_config', 'predict_with_evolved_model', 'run_generated_capability'}:resource='cpu'
    return {'category': category, 'category_label': en, 'category_label_zh': zh,
            'resource': meta.get('queue_resource', resource), 'timeout_seconds': min(7200,max(5,int(meta.get('timeout_seconds',timeout)))),
            'execution': 'inline' if name in INLINE or meta.get('workspace_relative_paths') else 'queued'}


def native_tools():
    from mcp_tools.tools.tools_agent_hub import get_tools
    return {t.name:t for t in get_tools()}


def catalog():
    from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
    rows=[]
    for name, tool in native_tools().items():
        rows.append({'name':name,'description':tool.description,**classify(name,tool.metadata),
                     'model_family': tool_family(name, tool), 'available':is_tool_allowed(name),'input_schema':tool.get_input_schema().model_json_schema()})
    counts=Counter(x['category'] for x in rows)
    return {'categories':[{'id':k,'label':v[0],'label_zh':v[1],'count':counts[k]} for k,v in CATEGORIES.items()],
            'tools':rows,'total':len(rows)}


def tool_family(name, tool):
    """Provenance group; resource/category remain independent of model origin."""
    if name in {'predict_protein_function','predict_residue_function','protein_model_predict','train_protein_model','generate_training_config','list_trained_models','register_trained_model'}:
        return 'factory'
    if name in {'predict_with_evolved_model','list_evolved_models','describe_evolved_model','run_generated_capability','search_generated_capabilities','describe_generated_capability'}:
        return 'rsi'
    if name in {'predict_structure_esmfold','predict_s4pred','predict_admet','zero_shot_mutation_sequence_prediction','zero_shot_mutation_structure_prediction','proteinmpnn_sequence_design_from_structure','proteinmpnn_sequence_scoring_from_structure'}:
        return 'pretrained'
    function = getattr(tool, 'func', None) or getattr(tool, 'coroutine', None)
    module = getattr(function, '__module__', '')
    for package, family in [('pretrained','pretrained'),('factory','factory'),('evolved','rsi')]:
        if module.startswith('mcp_tools.models.' + package + '.'):
            return family
    return 'utility'
