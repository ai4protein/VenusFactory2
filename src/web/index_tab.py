import gradio as gr
def create_index_tab(constant):
    # Read gjf file content
    try:
        with open("img/Show1.gjf", "r") as f:
            gjf_content = f.read()
    except Exception:
        gjf_content = "(Show1.gjf not found)"
    with gr.Blocks() as index_tab:
        gr.HTML(
            f'''            
                <script>
                // 页面加载时自动统计访问量
                (function() {{
                    fetch('/api/stats/track', {
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{module: 'total_visits'}})
                    });
                }})();
                </script>
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
                body, .gradio-container {{
                    font-family: 'Inter', 'Roboto', Arial, sans-serif !important;
                    background: #f4f7fa;
                    color: #222;
                }}
                .main-content {{
                    margin-top: 80px;
                    max-width: 1100px;
                    margin-left: auto;
                    margin-right: auto;
                    background: #fff;
                    border-radius: 16px;
                    box-shadow: 0 4px 24px rgba(30,41,59,0.07);
                    padding: 40px 36px 32px 36px;
                }}
                h1, h2, h3 {{
                    color: #2563eb;
                    font-weight: 700;
                    margin-bottom: 0.5em;
                }}
                p, li, ul {{
                    font-size: 1.08em;
                    line-height: 1.7;
                }}
                .card {{
                    background: #f4f7fa;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(30,41,59,0.04);
                    padding: 18px 22px;
                    margin-bottom: 18px;
                }}
                .stats-container {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 16px;
                    padding: 32px;
                    margin-top: 40px;
                    color: white;
                    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 32px;
                    margin-top: 32px;
                    max-width: 800px;
                    margin-left: auto;
                    margin-right: auto;
                }}
                .stat-item {{
                    background: rgba(255, 255, 255, 0.15);
                    border-radius: 16px;
                    padding: 28px 24px;
                    text-align: center;
                    backdrop-filter: blur(15px);
                    border: 1px solid rgba(255, 255, 255, 0.25);
                    transition: all 0.3s ease;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                }}
                .stat-item:hover {{
                    transform: translateY(-4px);
                    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
                    background: rgba(255, 255, 255, 0.2);
                }}
                .stat-number {{
                    font-size: 2.5em;
                    font-weight: 900;
                    margin-bottom: 8px;
                    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                }}
                .stat-label {{
                    font-size: 1.1em;
                    opacity: 0.9;
                    font-weight: 500;
                }}
                .stat-icon {{
                    font-size: 2em;
                    margin-bottom: 12px;
                    display: block;
                }}
                </style>
            <div class="main-content">
                <!-- 上半部分:VenusFactory 介绍 -->
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5em;">
                <img src="/file=img/venus_head.png" alt="Venus Head" style="height: 120px; margin-left: 10px;" />
                <img src="/file=img/venus_icon.png" alt="Venus Logo" style="height: 70px; margin-right: 10px;" />
            </div>
            <div style="text-align: center; margin-top: -60px; margin-bottom: 20px;">
                <h1 style="font-size:3.5em; font-weight:900;">Welcome to <span style='font-weight:900;'>VenusFactory</span> !</h1>
            </div>
            <div style="max-width: 1100px; margin: 0 auto; font-size: 1.2em; text-align: left;">
                <p style="font-size:1.2em; margin-bottom: 0.7em;"><b>VenusFactory</b> is a unified open-source platform for protein engineering, designed to simplify data acquisition, model fine-tuning, and functional analysis for both biologists and AI researchers.<br>
                The Web UI features four core modules:</p>
                <ul style="font-size:1.0em;">
                    <li>🤖 <b>VenusAgent-0.1</b> is an integrated AI assistant that answers questions related to the platform and protein AI.</li>
                    <li>🛠️ <b>Quick Tools</b> offers one-click protein analysis tools designed as a convenient method, making common tasks easy and accessible.</li>
                    <li>⚡ <b>Advanced Tools</b> enables zero-shot prediction, function analysis, and advanced data options for experienced users.</li>
                    <li>💾 <b>Download</b> allows you to get various protein data like AlphaFold2 Structures, RCSB PDB and InterPro.</li>
                </ul>
            </div>
            <hr style="margin: 40px 0; border: 1px solid #eee;">
            <!-- 中间部分:How to Use VenusFactory -->
            <div style="text-align: left; max-width: 1100px; margin: 0 auto;">
                <h1 style="font-size:2.2em; font-weight:900; color:#222; margin-bottom: 0.7em;">
                    <img src="/file=img/venus_icon.png" style="height: 48px; vertical-align: middle; margin-right: 10px;">
                    How to Use VenusFactory ?
                </h1>
                <div style="font-size:1.2em;">
                    <p style="font-size:1.2em;">Depending on your needs, VenusFactory can provide different services.</p>
                    <p style="font-size:1.2em;">If you want a quick answer about protein mutations, use VenusAgent-0.1. Upload your file, and the AI Assistant will give you a helpful reply.</p>
                    <p style="font-size:1.2em;">If you want to know possible mutation methods or protein functions, go to Quick Tools, choose the task you need, and you will get the result in a few minutes.</p>
                    <p style="font-size:1.2em;">If you have some knowledge about different protein models, you can use the Advanced Tools tab. All major models are available to meet your needs.</p>
                    <p style="font-size:1.2em;">If you want to get some protein data files, click the download tab, input the PDB ID, to download and use it for further research.</p>
                    <p style="font-size:1.2em; font-weight:bold; margin-top:2em;">Example GIF:</p>
                    <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/MovieGen/fig3.png" alt="Example GIF" style="max-width: 100%; height: auto; border-radius: 8px; margin-bottom: 1em;" />
                </div>
            </div>
            <!-- 其余内容保持不变 -->
            <hr style="margin: 40px 0; border: 1px solid #eee;">
            <!-- Citation 板块 -->
            <div style="text-align: left; max-width: 1100px; margin: 0 auto; font-size: 1.2em;">
                <h1 style="font-size:2.2em; font-weight:900; color:#222; margin-bottom: 0.7em;">Citation</h1>
                <div style="font-size:1.2em; margin-bottom: 0.7em;">✏️ Please cite our work if you have used VenusFactory.</div>
                <pre style="background:#f8f8f8; border-radius:8px; padding:18px; font-size:1.2em; overflow-x:auto;"><code>@article{{tan2025venusfactory,
  title={{VenusFactory: A Unified Platform for Protein Engineering Data Retrieval and Language Model Fine-Tuning}},
  author={{Tan, Yang and Liu, Chen and Gao, Jingyuan and Wu, Banghao and Li, Mingchen and Wang, Ruilin and Zhang, Lingrong and Yu, Huiqun and Fan, Guisheng and Hong, Liang and Zhou, Bingxin}},
  journal={{arXiv preprint arXiv:2503.15438}},
  year={{2025}}
}}</code></pre>
                <!-- 合作平台等内容, 作为Citation一部分 -->
                <div style="display: flex; flex-wrap: wrap; justify-content: space-between; max-width: 1100px; margin: 30px auto 0 auto; color: #666; font-size: 1.2em; gap: 32px;">
                    <div style="flex:1; min-width: 320px;">
                        <b>🤝 Cooperate Platform:</b> <a href="https://openbayes.com/" target="_blank">HyberAI</a><br>
                        <b>🧬 Small-sample mutation prediction tool:</b> <a href="https://github.com/ai4protein/Pro-FSFP" target="_blank">Pro-FSFP</a><br>
                        <b>⚡ The most advanced zero-shot protein prediction tool:</b> <a href="https://github.com/ai4protein/VenusREM" target="_blank">VenusREM</a><br>
                        <b>🏠 Developer homepage:</b> <a href="https://tyang816.github.io/" target="_blank">https://tyang816.github.io/</a><br>
                        <b>✉️ Developer contact information:</b> <a href="mailto:tanyang.august@sjtu.edu.cn">tanyang.august@sjtu.edu.cn</a>
                    </div>
                    <div style="flex:1; min-width: 320px;">
                        <b>🏢 Joint unit:</b>
                        <ul style="margin-left: 20px; list-style: none; padding: 0;">
                            <li style="display:inline-flex; align-items:center; margin-bottom: 10px;">
                                <a href="https://www.sjtu.edu.cn/" target="_blank" style="font-size:0.7em;">Shanghai Jiao Tong University</a>
                            </li>
                            <li style="display:inline-flex; align-items:center; margin-bottom: 10px;">
                                <a href="https://www.ecust.edu.cn/" target="_blank" style="font-size:0.7em;">East China University of Science and Technology</a>
                            </li>
                            <li style="display:inline-flex; align-items:center; margin-bottom: 10px;">
                                <a href="https://www.ecnu.edu.cn/" target="_blank" style="font-size:0.7em;">East China Normal University</a>
                            </li>
                            <li style="display:inline-flex; align-items:center; margin-bottom: 10px;">
                                <a href="https://www.shlab.org.cn/" target="_blank" style="font-size:0.7em;">Shanghai Artificial Intelligence Laboratory</a>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
            <hr style="margin: 40px 0; border: 1px solid #eee;">
            <!-- Additional Information 板块 -->
            <div style="background: #f7fafd; border-radius: 14px; box-shadow: 0 2px 12px rgba(30,41,59,0.06); padding: 32px 28px 24px 28px; max-width: 1100px; margin: 40px auto 0 auto;">
                <h1 style="font-size:2em; font-weight:900; color:#2563eb; margin-bottom: 0.5em; border-bottom: 2px solid #e0e7ef; padding-bottom: 0.2em; letter-spacing: 1px;">Additional Information</h1>
                <div style="display: flex; flex-wrap: wrap; gap: 40px;">
                    <!-- Supported Models -->
                    <div style="flex:1; min-width: 320px;">
                        <b style="font-size:1.1em; color:#1e293b;">Supported Models:</b>
                        <ul style="margin: 18px 0 0 0; padding: 0; list-style: none;">
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">ESM-1v/ESM-1b/ESM-650M</span>
                                <span style="font-size:0.87em; color:#444;"> – State-of-the-art protein language models from Meta AI for sequence-based prediction.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/622803v4.full" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/facebookresearch/esm" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">SaProt</span>
                                <span style="font-size:0.87em; color:#444;"> – A model for protein sequence analysis and function prediction.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2023.10.01.560349v5" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/westlake-repl/SaProt" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">MIF-ST</span>
                                <span style="font-size:0.87em; color:#444;"> – Structure-informed models for protein fitness and mutation effect prediction.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2022.05.25.493516v1.full" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/microsoft/protein-sequence-models" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">ProSST-2048</span>
                                <span style="font-size:0.87em; color:#444;"> – Large-scale protein sequence-structure models for zero-shot and supervised tasks.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2024.04.15.589672v2.full" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/ai4protein/ProSST" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">ProtSSN</span>
                                <span style="font-size:0.87em; color:#444;"> – Protein structure and sequence network for protein structure prediction.</span>
                                <a href="https://elifesciences.org/reviewed-preprints/98033" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/ai4protein/ProtSSN" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">Ankh-large</span>
                                <span style="font-size:0.87em; color:#444;"> – Transformer-based protein language model for structure and function tasks.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2023.01.16.524265v1.full" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/agemagician/Ankh" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">ProtBert-uniref50</span>
                                <span style="font-size:0.87em; color:#444;"> – BERT-based protein model trained on UniRef50.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2021.05.24.445464v1" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/agemagician/ProtTrans" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">ProtT5-xl-uniref50</span>
                                <span style="font-size:0.87em; color:#444;"> – T5-based protein model for sequence and structure prediction.</span>
                                <a href="https://arxiv.org/abs/2007.06225" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/agemagician/ProtTrans" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Code]</a>
                            </li>
                        </ul>
                    </div>
                    <!-- Supported Datasets -->
                    <div style="flex:1; min-width: 320px;">
                        <b style="font-size:1.1em; color:#1e293b;">Supported Datasets:</b>
                        <ul style="margin: 18px 0 0 0; padding: 0; list-style: none;">
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">DeepSol</span>
                                <span style="font-size:0.87em; color:#444;"> – Dataset for protein solubility prediction.</span>
                                <a href="https://academic.oup.com/bioinformatics/article/34/15/2605/4938490" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://zenodo.org/records/1162886" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">DeepSoluE</span>
                                <span style="font-size:0.87em; color:#444;"> – Enhanced solubility dataset for benchmarking.</span>
                                <a href="https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-023-01510-8" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://github.com/wangchao-malab/DeepSoluE" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">ProtSolM</span>
                                <span style="font-size:0.87em; color:#444;"> – Solubility dataset for machine learning tasks.</span>
                                <a href="https://arxiv.org/abs/2406.19744" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/ProtSolM_ESMFold_PDB" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">DeepLocBinary</span>
                                <span style="font-size:0.87em; color:#444;"> – Dataset for binary protein subcellular localization prediction.</span>
                                <a href="https://academic.oup.com/bioinformatics/article/33/21/3387/4099600" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/DeepLocBinary" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">DeepLocMulti</span>
                                <span style="font-size:0.87em; color:#444;"> – Dataset for multi-class protein subcellular localization prediction.</span>
                                <a href="https://academic.oup.com/bioinformatics/article/33/21/3387/4099600" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/DeepLocMulti" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">MetallonBinding</span>
                                <span style="font-size:0.87em; color:#444;"> – Dataset for protein metal ion binding site prediction.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2023.10.01.560349v5" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/MetallonBinding" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">Thermostability</span>
                                <span style="font-size:0.87em; color:#444;"> – Dataset for protein thermostability prediction.</span>
                                <a href="https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/2b44928ae11fb9384c4cf38708677c48-Paper-round2.pdf" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/Thermostability" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li style="margin-bottom: 12px;">
                                <span style="font-weight:bold; color:#2563eb;">SortingSignal</span>
                                <span style="font-size:0.87em; color:#444;"> – Dataset for protein sorting signal prediction.</span>
                                <a href="https://www.nature.com/articles/s41587-019-0036-z" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/SortingSignal" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                            <li>
                                <span style="font-weight:bold; color:#2563eb;">DeepET_Topt</span>
                                <span style="font-size:0.87em; color:#444;"> – Dataset for optimal growth temperature (Topt) prediction.</span>
                                <a href="https://academic.oup.com/bib/article/26/2/bbaf114/8074761" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/DeepET_Topt" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Datasets]</a>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <!-- Usage Statistics Section -->
            <div class="stats-container">
                <h1 style="font-size:2.2em; font-weight:900; color:white; margin-bottom: 0.5em; text-align: center;">
                    📊 Platform Usage Statistics
                </h1>
                <p style="text-align: center; font-size: 1.2em; opacity: 0.9; margin-bottom: 0;">
                    Real-time platform usage statistics
                </p>
                
                <div class="stats-grid">
                    <!-- Total Website Visits -->
                    <div class="stat-item">
                        <span class="stat-icon">🌐</span>
                        <div class="stat-number" id="total-visits">-</div>
                        <div class="stat-label">Total Website Visits</div>
                    </div>
                    
                    <!-- Agent Usage Count -->
                    <div class="stat-item">
                        <span class="stat-icon">🤖</span>
                        <div class="stat-number" id="agent-usage">-</div>
                        <div class="stat-label">VenusAgent Usage</div>
                    </div>
                    
                    <!-- Mutation Prediction Usage Count -->
                    <div class="stat-item">
                        <span class="stat-icon">🧬</span>
                        <div class="stat-number" id="evolution-usage">-</div>
                        <div class="stat-label">Mutation Prediction</div>
                    </div>
                    
                    <!-- Function Prediction Usage Count -->
                    <div class="stat-item">
                        <span class="stat-icon">⚡</span>
                        <div class="stat-number" id="prediction-usage">-</div>
                        <div class="stat-label">Function Prediction</div>
                    </div>
                </div>
                
                <!-- Statistics Script -->
                
                <!-- Custom Statistics Logic -->
                <script>
                    class RealTimeStatsManager {{
                        constructor() {{
                            this.updateInterval = null;
                               this.apiBaseUrl = '/api';
                            this.init();
                        }}
                        
                        async init() {{
                            await this.loadStats();
                            this.startAutoRefresh();
                            this.bindEvents();
                        }}
                        
                        async loadStats() {{
                            try {{
                                const response = await fetch(`${{this.apiBaseUrl}}/stats`);
                                if (response.ok) {{
                                    const data = await response.json();
                                    this.updateDisplay(data);
                                }} else {{
                                    console.error('Failed to load stats:', response.status);
                                    this.loadMockData();
                                }}
                            }} catch (error) {{
                                console.error('Failed to load stats:', error);
                                this.loadMockData();
                            }}
                        }}
                        
                        updateDisplay(data) {{
                               const evolutionElement = document.getElementById('evolution-usage');
                            if (evolutionElement) {{
                                evolutionElement.textContent = this.formatNumber(data.mutation_prediction || 0);
                            }}
                            
                               const predictionElement = document.getElementById('prediction-usage');
                            if (predictionElement) {{
                                predictionElement.textContent = this.formatNumber(data.function_analysis || 0);
                            }}
                            
                               setTimeout(() => {{
                                const totalElement = document.getElementById('total-visits');
                                if (totalElement) {{
                                    totalElement.textContent = this.formatNumber(data.total_visits || 0);
                                }}
                            }}, 2000);
                        }}
                        
                        loadMockData() {{
                               document.getElementById('evolution-usage').textContent = '1,234';
                            document.getElementById('prediction-usage').textContent = '2,341';
                            document.getElementById('total-visits').textContent = '5,678';
                        }}
                        
                        formatNumber(num) {{
                            return new Intl.NumberFormat().format(num);
                        }}
                        
                        startAutoRefresh() {{
                            this.updateInterval = setInterval(() => {{
                                this.loadStats();
                            }}, 30000); 
                        }}
                        
                        bindEvents() {{
                            // 监听页面可见性变化
                            document.addEventListener('visibilitychange', () => {{
                                if (!document.hidden) {{
                                    this.loadStats();
                                }}
                            }});
                        }}
                        
                        async trackUsage(module) {{
                            try {{
                                const response = await fetch(`${{this.apiBaseUrl}}/stats/track`, {{
                                    method: 'POST',
                                    headers: {{
                                        'Content-Type': 'application/json',
                                    }},
                                    body: JSON.stringify({{
                                        module: module,
                                        timestamp: new Date().toISOString()
                                    }})
                                }});
                                
                                if (response.ok) {{
                                    console.log(`Tracked ${{module}} usage successfully`); 
                                    await this.loadStats();
                                }} else {{
                                    console.error('Failed to track usage:', response.status);
                                }}
                            }} catch (error) {{
                                console.error('Failed to track usage:', error);
                            }}
                        }}
                    }}
                    
                       document.addEventListener('DOMContentLoaded', () => {{
                        window.statsManager = new RealTimeStatsManager();
                    }});
                    
                       function trackUsage(module) {{
                        if (window.statsManager) {{
                            window.statsManager.trackUsage(module);
                        }}
                    }}
                </script>
            </div>
            '''
        )
        
    return {"index_tab": index_tab}
