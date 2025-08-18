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
                // Track page visits asynchronously without blocking page load
                (function() {{
                    // Use setTimeout to ensure page is fully loaded before tracking
                    setTimeout(() => {{
                        fetch('/api/stats/track', {{
                            method: 'POST',
                            headers: {{'Content-Type': 'application/json'}},
                            body: JSON.stringify({{module: 'total_visits'}})
                        }}).catch(error => {{
                            console.log('Stats API call failed, using mock data');
                        }});
                    }}, 1000);
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
                    max-width: 1400px;
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
                    background: linear-gradient(135deg, #f6fbff 0%, #e3f3fb 100%);
                    border-radius: 16px;
                    padding: 32px;
                    margin-top: 40px;
                    color: #1e293b;
                    box-shadow: 0 8px 32px rgba(37, 99, 235, 0.07);
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 32px;
                    margin-top: 24px;
                    max-width: 1200px;
                    margin-left: auto;
                    margin-right: auto;
                }}
                .stat-item {{
                    background: rgba(255, 255, 255, 0.8);
                    border-radius: 12px;
                    padding: 16px 10px;
                    text-align: center;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(37, 99, 235, 0.15);
                    transition: all 0.3s ease;
                    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.08);
                }}
                .stat-item:hover {{
                    transform: translateY(-4px);
                    box-shadow: 0 12px 40px rgba(37, 99, 235, 0.15);
                    background: rgba(255, 255, 255, 0.9);
                }}
                .stat-number {{
                    font-size: 1.5em;
                    font-weight: 900;
                    margin-bottom: 4px;
                    text-shadow: 0 1px 2px rgba(37, 99, 235, 0.12);
                }}
                .stat-label {{
                    font-size: 0.95em;
                    opacity: 0.9;
                    font-weight: 500;
                }}
                .stat-icon {{
                    font-size: 1.3em;
                    margin-bottom: 6px;
                    display: block;
                }}
                </style>
                <div class="main-content">
            <!-- Top section: VenusFactory introduction -->
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5em;">
                <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venusfactory_logo.png" alt="Venus Head" style="height: 150px; margin-left: 10px;" />
                <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/venus_logo.png" alt="Venus Logo" style="height: 100px; margin-right: 10px; margin-top: 20px;" />
            </div>
            <div style="text-align: center; margin-top: -80px; margin-bottom: 40px;">
                <h1 style="font-size:3.5em; font-weight:900;">Welcome to <span style='font-weight:900;'>VenusFactory</span> !</h1>
            </div>
            <div style="max-width: 1400px; margin: 0 auto; font-size: 1.2em; text-align: left;">
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
            <!-- Middle section: How to Use VenusFactory -->
            <div style="text-align: left; max-width: 1400px; margin: 0 auto;">
                <h1 style="font-size:2.2em; font-weight:900; color:#222; margin-bottom: 0.7em;">
                    How to Use VenusFactory
                </h1>
                <div style="font-size:1.2em;">
                    <p style="font-size:1.2em;">Depending on your needs, VenusFactory can provide different services.</p>
                    <p style="font-size:1.2em;">If you want a quick answer about protein mutations, use VenusAgent-0.1. Upload your file, and the AI Assistant will give you a helpful reply.</p>
                    <p style="font-size:1.2em;">If you want to know possible mutation methods or protein functions, go to Quick Tools, choose the task you need, and you will get the result in a few minutes.</p>
                    <p style="font-size:1.2em;">If you have some knowledge about different protein models, you can use the Advanced Tools tab. All major models are available to meet your needs.</p>
                    <p style="font-size:1.2em;">If you want to get some protein data files, click the download tab, input the PDB ID or UniProt ID, and download it for further research.</p>
                    
                    <p style="font-size:1.2em; font-weight:bold; margin-top:2em;">Module Demonstrations:</p>
                    
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 1em;">
                        <!-- VenusAgent-0.1 GIF -->
                        <div style="text-align: center;">
                            <h3 style="color: #2563eb; margin-bottom: 10px;">🤖 VenusAgent-0.1</h3>
                            <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/agent.gif" alt="VenusAgent-0.1 Demo" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
                        </div>
                        
                        <!-- Quick Tools GIF -->
                        <div style="text-align: center;">
                            <h3 style="color: #2563eb; margin-bottom: 10px;">🛠️ Quick Tools</h3>
                            <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/quick_tool.gif" alt="Quick Tools Demo" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
                        </div>
                        
                        <!-- Advanced Tools GIF -->
                        <div style="text-align: center;">
                            <h3 style="color: #2563eb; margin-bottom: 10px;">⚡ Advanced Tools</h3>
                            <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/advanced_tool.gif" alt="Advanced Tools Demo" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
                        </div>
                        
                        <!-- Download GIF -->
                        <div style="text-align: center;">
                            <h3 style="color: #2563eb; margin-bottom: 10px;">💾 Download</h3>
                            <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/gif/download.gif" alt="Download Demo" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" />
                        </div>
                    </div>
                </div>
            </div>
            <hr style="margin: 40px 0; border: 1px solid #eee;">

            <div style="text-align: left; max-width: 1400px; margin: 0 auto; font-size: 1.2em;">
                <h1 style="font-size:2.2em; font-weight:900; color:#222; margin-bottom: 0.7em;">Citation</h1>
                <div style="font-size:1.2em; margin-bottom: 0.7em;">✏️ Please cite our work if you have used VenusFactory.</div>
                <pre style="background:#f8f8f8; border-radius:8px; padding:18px; font-size:1.2em; overflow-x:auto;"><code>@inproceedings{{tan-etal-2025-venusfactory,
            title = "{{V}}enus{{F}}actory: An Integrated System for Protein Engineering with Data Retrieval and Language Model Fine-Tuning",
            author = "Tan, Yang and Liu, Chen and Gao, Jingyuan and Wu, Banghao and Li, Mingchen and Wang, Ruilin and Zhang, Lingrong and Yu, Huiqun and Fan, Guisheng and Hong, Liang and Zhou, Bingxin",
            editor = "Mishra, Pushkar and Muresan, Smaranda and Yu, Tao",
            booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
            month = jul,
            year = "2025",
            address = "Vienna, Austria",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2025.acl-demo.23/",
            doi = "10.18653/v1/2025.acl-demo.23",
            pages = "230--241",
            ISBN = "979-8-89176-253-4",
        }}</code></pre>

                <!-- Partner Institutions Display Area -->
                <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 16px; padding: 40px; margin: 40px auto; max-width: 1400px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                    <div style="text-align: center; margin-bottom: 30px;">
                        <h2 style="font-size: 1.5em; font-weight: 700; color: #1e293b; margin-bottom: 10px;">Partner Institutions</h2>
                    </div>
                    
                    <!-- Four Institutions -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <a href="https://www.sjtu.edu.cn/" target="_blank" style="text-decoration: none; color: inherit;">
                            <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); transition: all 0.3s ease; border: 1px solid #e2e8f0; display: flex; align-items: center; justify-content: center; gap: 15px;">
                                <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/sjtu_logo.jpg" alt="SJTU" style="height: 40px;">
                                <div style="font-size: 1.2em; font-weight: 600; color: #1e293b;">Shanghai Jiao Tong University</div>
                            </div>
                        </a>
                        <a href="https://www.ecust.edu.cn/" target="_blank" style="text-decoration: none; color: inherit;">
                            <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); transition: all 0.3s ease; border: 1px solid #e2e8f0; display: flex; align-items: center; justify-content: center; gap: 15px;">
                                <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/ecust_logo.jpg" alt="ECUST" style="height: 40px;">
                                <div style="font-size: 1.2em; font-weight: 600; color: #1e293b;">East China University of Science and Technology</div>
                            </div>
                        </a>
                        <a href="https://www.ecnu.edu.cn/" target="_blank" style="text-decoration: none; color: inherit;">
                            <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); transition: all 0.3s ease; border: 1px solid #e2e8f0; display: flex; align-items: center; justify-content: center; gap: 15px;">
                                <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/ecnu_logo.jpg" alt="ECNU" style="height: 40px;">
                                <div style="font-size: 1.2em; font-weight: 600; color: #1e293b;">East China Normal University</div>
                            </div>
                        </a>
                        <a href="https://www.shlab.org.cn/" target="_blank" style="text-decoration: none; color: inherit;">
                            <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); transition: all 0.3s ease; border: 1px solid #e2e8f0; display: flex; align-items: center; justify-content: center; gap: 15px;">
                                <img src="https://blog-img-1259433191.cos.ap-shanghai.myqcloud.com/venus/img/shailab_logo.jpg" alt="SHAILab" style="height: 40px;">
                                <div style="font-size: 1.2em; font-weight: 600; color: #1e293b;">Shanghai AI Laboratory</div>
                            </div>
                        </a>
                    </div>
                </div>
                
                <!-- Developer Information Area -->
                <div style="background: #f8fafc; border-radius: 12px; padding: 30px; margin: 30px auto; max-width: 1400px;">
                    <h3 style="font-size: 1.2em; font-weight: 700; color: #1e293b; margin-bottom: 20px;">Cooperation Platform & Developer Information</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; font-size: 1.1em;">
                        <div>  
                            <div style="margin-bottom: 12px;"><b>🤝 Cooperation Platform:</b> <a href="https://hyper.ai/cn/tutorials/38568" target="_blank" style="color: #2563eb;">HyberAI</a></div>
                            <div style="margin-bottom: 12px;"><b>🧬 Few-shot mutation prediction tool:</b> <a href="https://github.com/ai4protein/Pro-FSFP" target="_blank" style="color: #2563eb;">Pro-FSFP</a></div>
                            <div style="margin-bottom: 12px;"><b>⚡ The most advanced zero-shot protein prediction tool:</b> <a href="https://github.com/ai4protein/VenusREM" target="_blank" style="color: #2563eb;">VenusREM</a></div>
                        </div>
                        <div>
                            <div style="margin-bottom: 12px;"><b>🏠 Developer homepage:</b> <a href="https://tyang816.github.io/" target="_blank" style="color: #2563eb;">https://tyang816.github.io/</a></div>
                            <div style="margin-bottom: 12px;"><b>✉️ Developer contact information:</b> <a href="mailto:tanyang.august@sjtu.edu.cn" style="color: #2563eb;">tanyang.august@sjtu.edu.cn</a>, <a href="mailto:zlr_zmm@163.com" style="color: #2563eb;">zlr_zmm@163.com</a></div>
                        </div>
                    </div>
                </div>
            </div>
            <hr style="margin: 40px 0; border: 1px solid #eee;">
            <!-- Additional Information Section -->
            <div style="background: #f7fafd; border-radius: 14px; box-shadow: 0 2px 12px rgba(30,41,59,0.06); padding: 32px 28px 24px 28px; max-width: 1400px; margin: 40px auto 0 auto;">
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
                                <span style="font-weight:bold; color:#2563eb;">MetalIonBinding</span>
                                <span style="font-size:0.87em; color:#444;"> – Dataset for protein metal ion binding site prediction.</span>
                                <a href="https://www.biorxiv.org/content/10.1101/2023.10.01.560349v5" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Paper]</a>
                                <a href="https://huggingface.co/datasets/AI4Protein/MetalIonBinding" target="_blank" style="margin-left:8px; color:#2563eb; text-decoration:underline; font-size:0.87em;">[Datasets]</a>
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
                <h1 style="font-size:2.2em; font-weight:900; color:#1e293b; margin-bottom: 0.5em; text-align: center;">
                    📊 Platform Usage Statistics
                </h1>
                <p style="text-align: center; font-size: 1.2em; opacity: 0.8; margin-bottom: 0; color:#1e293b;">
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
                            this.init();
                        }}
                        
                        async init() {{
                            this.loadMockData();
                            this.startAutoRefresh();
                            this.bindEvents();
                        }}
                        
                        loadMockData() {{
                            try {{
                                fetch('/file/stats_data.json')
                                    .then(response => response.json())
                                    .then(data => {{
                                        const mutationTotal = (data.mutation_prediction_quick || 0) + (data.mutation_prediction_advanced || 0);
                                        const evolutionElement = document.getElementById('evolution-usage');
                                        if (evolutionElement) {{
                                            evolutionElement.textContent = this.formatNumber(mutationTotal);
                                        }}
                                        
                                        const functionTotal = (data.function_prediction_quick || 0) + (data.function_prediction_advanced || 0);
                                        const predictionElement = document.getElementById('prediction-usage');
                                        if (predictionElement) {{
                                            predictionElement.textContent = this.formatNumber(functionTotal);
                                        }}
                                        
                                        const agentElement = document.getElementById('agent-usage');
                                        if (agentElement) {{
                                            agentElement.textContent = this.formatNumber(data.agent_usage || 0);
                                        }}
                                        
                                        const totalElement = document.getElementById('total-visits');
                                        if (totalElement) {{
                                            totalElement.textContent = this.formatNumber(data.total_visits || 0);
                                        }}
                                    }})
                                    .catch(error => {{
                                        console.log('Failed to load stats file, using fallback data');
                                        document.getElementById('evolution-usage').textContent = '0';
                                        document.getElementById('prediction-usage').textContent = '0';
                                        document.getElementById('agent-usage').textContent = '0';
                                        document.getElementById('total-visits').textContent = '0';
                                    }});
                            }} catch (error) {{
                                console.log('Error loading stats:', error);
                                document.getElementById('evolution-usage').textContent = '0';
                                document.getElementById('prediction-usage').textContent = '0';
                                document.getElementById('agent-usage').textContent = '0';
                                document.getElementById('total-visits').textContent = '0';
                            }}
                        }}
                        
                        formatNumber(num) {{
                            return new Intl.NumberFormat().format(num);
                        }}
                        
                        startAutoRefresh() {{
                            this.updateInterval = setInterval(() => {{
                                this.loadMockData();
                            }}, 60000); 
                        }}
                        
                        bindEvents() {{
                            let visibilityTimeout;
                            document.addEventListener('visibilitychange', () => {{
                                if (!document.hidden) {{
                                    clearTimeout(visibilityTimeout);
                                    visibilityTimeout = setTimeout(() => {{
                                        this.loadMockData();
                                    }}, 2000);
                                }}
                            }});
                        }}
                        
                        async trackUsage(module) {{
                            try {{
                                fetch('/api/predict/', {{
                                    method: 'POST',
                                    headers: {{
                                        'Content-Type': 'application/json',
                                    }},
                                    body: JSON.stringify({{
                                        fn_index: 1,
                                        data: [module]
                                    }})
                                }}).catch(error => {{
                                    console.log('Failed to track usage:', error);
                                }});
                            }} catch (error) {{
                                console.log('Failed to track usage:', error);
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
