from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import json
import zipfile
import base64
import io
import itertools
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from app.graph_builder import (
    build_rule_graph,
    detect_communities_leiden,
    accumulate_communities_to_master,
    compute_layout_weights,
    build_rules,
    convert_graph_to_json,
)
from app.graph_engine import (
    CommunityDetector,
    CommunityExplainer,
    GraphRenderer,
    NodeEmbedder,
)

STATE_DIR = "/tmp/bankgraphai_state"
os.makedirs(STATE_DIR, exist_ok=True)

def _save(key: str, obj):
    import pickle
    path = os.path.join(STATE_DIR, f"{key}.pkl")
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        pass

def _load(key: str):
    import pickle
    path = os.path.join(STATE_DIR, f"{key}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_resource
def get_detector():
    return CommunityDetector()

@st.cache_resource
def get_explainer():
    return CommunityExplainer()

@st.cache_resource
def get_renderer():
    return GraphRenderer()

@st.cache_resource
def get_embedder():
    return NodeEmbedder()

# ---------------------------------------------------------------------------
# Feature Engineering (colleague's logic)
# ---------------------------------------------------------------------------
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    already = all(c in df.columns for c in ['creation_hour', 'creation_day_shift', 'faixa_movement', 'creation_month'])
    if already:
        return df
    if 'creation_time' in df.columns and 'creation_hour' not in df.columns:
        temp_time = pd.to_datetime(df['creation_time'], format='%H:%M:%S', errors='coerce')
        df['creation_hour'] = temp_time.dt.hour
        limites_horas = [0, 6, 12, 18, 24]
        nomes_turnos = ['Madrugada', 'Manhã', 'Tarde', 'Noite']
        df['creation_day_shift'] = pd.cut(df['creation_hour'], bins=limites_horas, labels=nomes_turnos, right=False, include_lowest=True)
    if 'movement' in df.columns and 'faixa_movement' not in df.columns:
        qbins = 150
        _, limites = pd.qcut(df['movement'], q=qbins, retbins=True, duplicates='drop')
        labels = [f"{limites[i]:.2f}<x<={limites[i+1]:.2f}" for i in range(len(limites)-1)]
        df['faixa_movement'] = pd.qcut(df['movement'], q=qbins, labels=labels, duplicates='drop')
    for date_col in ['entry_date', 'creation_date', 'date']:
        if date_col in df.columns and 'creation_month' not in df.columns:
            try:
                td = pd.to_datetime(df[date_col], errors='coerce')
                df['creation_month'] = td.dt.month
                df['entry_month_number'] = td.dt.month
                df['creation_week'] = td.dt.isocalendar().week.astype(int)
                df['entry_week'] = td.dt.isocalendar().week.astype(int)
                df['creation_day_of_month'] = td.dt.day
                df['entry_day_of_month'] = td.dt.day
                break
            except Exception:
                continue
    return df

# ---------------------------------------------------------------------------
# Sankey
# ---------------------------------------------------------------------------
def build_sankey(G, top_k=10):
    tw = {}
    for u, v, d in G.edges(data=True):
        et = d.get("edge_type", "unknown")
        tw[et] = tw.get(et, 0) + d.get("weight", 1)
    top = sorted(tw.items(), key=lambda x: -x[1])[:top_k]
    labels = [t[0].replace("shared_", "") for t in top]
    values = [t[1] for t in top]
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="rgba(255,255,255,0.1)", width=0.5),
                  label=["Users"] + labels, color=["#818cf8"] * (len(labels) + 1)),
        link=dict(source=[0]*len(labels), target=list(range(1,len(labels)+1)), value=values,
                  color=["rgba(129,140,248,0.6)"]*len(labels))
    )])
    fig.update_layout(title="Connection Types by Weight", font=dict(color="#e0e0e0", size=12),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=500)
    return fig

# ---------------------------------------------------------------------------
# vis.js HTML generator (interactive graph)
# ---------------------------------------------------------------------------
def generate_vis_html(graph_data: dict) -> str:
    """Generate an HTML file with vis.js force-directed graph."""
    nodes_json = json.dumps([
        {"id": n["id"], "label": n["id"], "x": n.get("x")*800, "y": n.get("y")*600, 
         "color": ["#818cf8","#ef553b","#00cc96","#ab63fa","#ffa15a","#19d3f3","#ff6692"][n.get("community",0)%7],
         "value": 1 + (n.get("community",0)%3)}
        for n in graph_data.get("nodes", [])
    ])
    edges_json = json.dumps([
        {"from": e["source"], "to": e["target"], "value": e.get("weight",1),
         "title": f"weight: {e.get('weight',1)}"}
        for e in graph_data.get("edges", [])
    ])
    
    html = f"""<!DOCTYPE html>
<html><head><title>Interactive Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>body{{margin:0;background:#0a0a1a;height:100vh}}#network{{width:100%;height:100vh}}</style>
</head><body>
<div id="network"></div>
<script>
var nodes = new vis.DataSet({nodes_json});
var edges = new vis.DataSet({edges_json});
var container = document.getElementById('network');
var data = {{nodes: nodes, edges: edges}};
var options = {{
    physics: {{
        enabled: true,
        solver: "barnesHut",
        barnesHut: {{gravitationalConstant: -3000, springLength: 500, springConstant: 0.005, damping: 0.92}},
        stabilization: {{iterations: 500, updateInterval: 25, onlyDynamicEdges: false}}
    }},
    interaction: {{hover: true, tooltipDelay: 100, dragNodes: true, dragView: true, zoomView: true}},
    nodes: {{font: {{color: '#e8e8f0', size: 13}}, borderWidth: 2, scaling: {{min: 8, max: 35}}}},
    edges: {{color: {{color: 'rgba(255,255,255,0.15)', highlight: 'rgba(129,140,248,0.8)'}}, width: 1, smooth: {{type: 'continuous'}}}},
    background: '#0a0a1a'
}};
var network = new vis.Network(container, data, options);
network.once('stabilizationIterationsDone', function() {{
    network.setOptions({{physics: {{enabled: false}}}});
}});
</script></body></html>"""
    return html

def get_zip_download_link(all_graphs: dict) -> str:
    """Create a ZIP with all JSON files + all HTML files and return a download link."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, (gdata, communities) in all_graphs.items():
            # JSON
            zf.writestr(f"{name}.json", json.dumps(gdata, ensure_ascii=False, indent=2))
            # Vis.js HTML
            html = generate_vis_html(gdata)
            zf.writestr(f"{name}.html", html)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:application/zip;base64,{b64}" download="graph_exports.zip">📥 Download ZIP (JSON + HTML)</a>'

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
* { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background: #0a0a1a; color: #e8e8f0; }
h1 { font-size: 2.2rem !important; font-weight: 800 !important; background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
section[data-testid="stSidebar"] { background: #0d0d24 !important; }
section[data-testid="stSidebar"] .stButton button { width: 100%; background: rgba(255,255,255,0.04); color: #c8c8e0; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px 16px; transition: all 0.2s ease; text-align: left; }
section[data-testid="stSidebar"] .stButton button:hover { background: rgba(129,140,248,0.15); border-color: rgba(129,140,248,0.3); transform: translateX(3px); }
div[data-testid="metric-container"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 18px 20px; }
.stButton button[kind="primary"] { background: linear-gradient(135deg, #818cf8, #a78bfa) !important; color: white !important; border-radius: 10px !important; }
.community-card { background: rgba(255,255,255,0.04); border-radius: 14px; padding: 22px; margin: 12px 0; border: 1px solid rgba(255,255,255,0.08); }
.community-card h4 { color: #818cf8; margin: 0 0 10px 0; }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="BankGraphAI", layout="wide")
st.title("Plataforma Teste")
st.markdown("Análise de redes financeiras com grafos inteligentes")

def init_state():
    defaults = {
        "page": "home", "df": None, "graph_built": False, "G": None,
        "graph_nodes": 0, "graph_edges": 0, "feature_names": [],
        "community_results": {}, "explanation_report": None, "embedding_result": None,
        "user_df": None, "edge_method": "",
        "last_filter": [], "last_color": "none",
        "rules_metadata": None, "exported_graphs": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            disk = _load(k)
            st.session_state[k] = disk if disk is not None else v

init_state()
detector = get_detector()
explainer = get_explainer()
renderer = get_renderer()
embedder = get_embedder()

st.sidebar.header("Pipeline Controls")
uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"], key="uploader")
st.sidebar.markdown("---### Actions")
for c, l in zip(["build", "communities", "explain", "visualize", "embeddings", "export"],
                 ["1. Build User Graph", "2. Detect Communities", "3. Explain Communities",
                  "4. Visualize Graph", "5. Generate Embeddings", "6. Export Graphs"]):
    if st.sidebar.button(l, width='stretch'):
        st.session_state.page = c
st.sidebar.markdown("---**Status:**")
if st.session_state.graph_built: st.sidebar.success("✅ User graph built")
if st.session_state.community_results: st.sidebar.success("✅ Communities")
if st.session_state.explanation_report: st.sidebar.success("✅ Explained")
if st.session_state.embedding_result: st.sidebar.success("✅ Embeddings")

# HOME
if uploaded_file is not None:
    path = f"/tmp/{uploaded_file.name}"
    with open(path, "wb") as f: f.write(uploaded_file.read())
    with st.spinner("Loading CSV…"):
        t0 = time.time()
        pdf = pd.read_csv(path, low_memory=False)
        st.session_state.df = pdf; _save("df", pdf)
        c1,c2,c3 = st.columns(3)
        c1.metric("Columns", len(pdf.columns)); c2.metric("Rows", f"{pdf.shape[0]:,}")
        c3.metric("Time", f"{time.time()-t0:.1f}s")
    st.dataframe(pdf.head(10), width='stretch')
    if "creator_user_id" in pdf.columns:
        st.info(f"📊 {pdf.shape[0]:,} transactions — {pdf['creator_user_id'].nunique()} unique users")

# BUILD
if st.session_state.page == "build":
    st.subheader("Build User Graph")
    st.markdown("**Colleague's approach:** Counter-based co-occurrence → Leiden → Master graph.")
    if st.session_state.df is None:
        st.warning("Upload a CSV first.")
    else:
        df: pd.DataFrame = st.session_state.df
        with st.spinner("Applying feature engineering…"):
            df_fe = apply_feature_engineering(df.copy())
            st.session_state.df = df_fe; _save("df", df_fe)

        possible_nodes = ['creator_user_id']
        default_edges = ['erp_book_account_name','profit_center_name','cost_center','journal_entry_origin',
            'exchange_id','movement_description','faixa_movement','id','supplier_number','asset_number',
            'auxiliary_id','journal_entry_id','status_origin','partner_name','erp_book_account_type',
            'movement_type','entry_credit_or_debit','erp_transaction_code','sap_journal_entry_id',
            'creation_day_of_month','creation_day_shift','creator_user_id']
        available_edges = [c for c in default_edges if c in df_fe.columns]
        if not available_edges: available_edges = [c for c in df_fe.columns if c != 'creator_user_id'][:10]
        default_analysis = ['full_period','creation_month','entry_month_number','creation_day_of_month',
            'entry_day_of_month','entry_week','creation_week','creation_hour','creation_day_shift']
        available_analysis = [a for a in default_analysis if a in df_fe.columns or a == 'full_period']

        if "creator_user_id" not in df_fe.columns:
            st.error("Missing 'creator_user_id' column.")
        else:
            unique_users = df_fe["creator_user_id"].nunique()
            st.success(f"📊 {unique_users} users × {df_fe.shape[0]:,} rows")
            new_cols = [c for c in df_fe.columns if c not in df.columns]
            if new_cols: st.info(f"✅ Feature engineering: {', '.join(new_cols)}")
            
            c1,c2 = st.columns(2)
            with c1: selected_nodes = st.multiselect("Node(s):", options=possible_nodes, default=possible_nodes, key="bld_nodes")
            with c2: selected_edges = st.multiselect("Edge cols:", options=available_edges, default=available_edges[:6], key="bld_edges")
            selected_analysis = st.multiselect("Analysis periods:", options=available_analysis,
                default=[a for a in ['full_period','creation_month'] if a in available_analysis], key="bld_analysis")

            if st.button("Build Master Graph Now", type="primary", key="build_go") and selected_nodes and selected_edges:
                rules = build_rules(nodes=selected_nodes, edges=selected_edges, analysis=selected_analysis)
                st.info(f"Processing {len(rules)} rules…")
                status=st.empty(); bar=st.progress(0)
                master = nx.Graph(); all_meta = {}
                for idx, rule in enumerate(rules):
                    nc, ec = rule['node'], rule['connect_with']
                    status.info(f"Rule {idx+1}/{len(rules)}: {nc}×{ec}")
                    rm = {'edge_col': ec, 'periods': {}}
                    for a in rule['analysis']:
                        if ec == a or nc == ec: continue
                        if a == 'full_period':
                            G_r = build_rule_graph(df_fe, nc, ec, a)
                            comms, _ = detect_communities_leiden(G_r)
                            # Accumulate EDGES from rule graph (not communities) for better scaling
                            for u, v, w in G_r.edges(data='weight', default=1):
                                if master.has_edge(u, v):
                                    master[u][v]['weight'] += w
                                else:
                                    master.add_edge(u, v, weight=w, edge_type=f"shared_{ec}")
                            rm['periods'][f'{a}__all'] = {'n_communities': len(comms),'n_edges': G_r.number_of_edges(),'n_nodes': G_r.number_of_nodes()}
                        else:
                            if a not in df_fe.columns: continue
                            for p, pdf in df_fe.groupby(a, sort=False):
                                G_r = build_rule_graph(pdf, nc, ec, a, p)
                                if G_r.number_of_nodes() <= 1: continue
                                comms, _ = detect_communities_leiden(G_r)
                                for u, v, w in G_r.edges(data='weight', default=1):
                                    if master.has_edge(u, v):
                                        master[u][v]['weight'] += w
                                    else:
                                        master.add_edge(u, v, weight=w, edge_type=f"shared_{ec}")
                                rm['periods'][f'{a}__{str(p)}'] = {'n_communities': len(comms),'n_edges': G_r.number_of_edges(),'n_nodes': G_r.number_of_nodes()}
                    all_meta[f"rule_{idx}"] = rm
                    bar.progress((idx+1)/len(rules))
                status.empty(); bar.empty()
                st.session_state.G = master; st.session_state.graph_nodes = master.number_of_nodes()
                st.session_state.graph_edges = master.number_of_edges(); st.session_state.graph_built = True
                st.session_state.feature_names = list(df_fe.columns)
                st.session_state.edge_method = f"colleague: {len(rules)} rules"
                st.session_state.rules_metadata = all_meta
                _save("G", master); _save("graph_nodes", master.number_of_nodes()); _save("graph_edges", master.number_of_edges())
                _save("graph_built", True); _save("rules_metadata", all_meta)
                tw = sum(d.get("weight",1) for _,_,d in master.edges(data=True))
                max_possible = master.number_of_nodes() * (master.number_of_nodes() - 1) // 2
                pct = 100 * master.number_of_edges() / max_possible if max_possible > 0 else 0
                st.success(f"✅ {master.number_of_nodes()} usuários — {master.number_of_edges():,} conexões — peso total: {tw:,}")
                
                with st.expander("📊 Entenda os números", expanded=False):
                    st.markdown(f"""
                    **Por que poucas arestas?**
                    - Com {master.number_of_nodes()} usuários, o máximo teórico de conexões é {max_possible:,} (grafo completo)
                    - Cada aresta existe apenas se 2+ usuários compartilham **pelo menos 1 valor** em uma coluna
                    - Se usuário A e B compartilham `erp_book_account_name` E `cost_center`, ainda é **1 aresta** (não 2)
                    
                    **Por que o peso é alto ({tw:,} = média {tw/master.number_of_edges():.1f} por aresta)?**
                    - Para cada regra (coluna selecionada), contamos quantas vezes os usuários co-ocorrem
                    - Se A e B compartilham `erp_book_account_name` em 50 transações → peso += 50
                    - Se também compartilham `cost_center` em 30 transações → peso += 30
                    - **Peso final = 80** (1 aresta com peso 80)
                    
                    **Exemplo prático:**
                    - 2 colunas selecionadas: ~500 arestas, peso total ~20.000
                    - 6 colunas selecionadas: ~1.176 arestas, peso total ~115.000
                    
                    ✅ Mais colunas = **mais peso** nas mesmas arestas, não mais arestas diferentes!
                    """)
                rows=[{"Rule":rk,"Edge":rv['edge_col'],"Periods":len(rv['periods']),"Edges":sum(p['n_edges'] for p in rv['periods'].values())} for rk,rv in all_meta.items()]
                st.dataframe(pd.DataFrame(rows), width='stretch')

    if st.session_state.graph_built and st.session_state.G:
        G=st.session_state.G; tw=sum(d.get("weight",1) for _,_,d in G.edges(data=True))
        max_p = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2
        pct = 100 * G.number_of_edges() / max_p if max_p > 0 else 0
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Usuários", G.number_of_nodes())
        c2.metric("Arestas", f"{G.number_of_edges():,} ({pct:.0f}%)")
        c3.metric("Peso Total", f"{tw:,}")
        c4.metric("Peso Médio", f"{tw/max(G.number_of_edges(),1):.1f}")

# DETECT
if st.session_state.page == "communities":
    st.subheader("Community Detection")
    if not st.session_state.graph_built or st.session_state.G is None:
        st.warning("Build first.")
    else:
        G=st.session_state.G; st.info(f"{G.number_of_nodes()} nodes, {G.number_of_edges():,} edges")
        method=st.selectbox("Algorithm",["leiden","louvain","label_propagation","spectral","all"],key="comm_method")
        k=st.slider("k",2,20,8,key="comm_k")
        if st.button("Run",type="primary",key="comm_run"):
            with st.spinner("Running…"):
                if method=="all": results=detector.all_methods(G,k_spectral=k,k_kmeans=8)
                elif method=="leiden": results={"leiden":detector.leiden(G)}
                elif method=="louvain": results={"louvain":detector.louvain(G)}
                elif method=="label_propagation": results={"label_propagation":detector.label_propagation(G)}
                elif method=="spectral": results={"spectral":detector.spectral(G,k=k)}
            st.session_state.community_results=results; _save("community_results",results)
            rows=[{"Method":n,"Communities":r.n_communities,"Score":f"{r.score:.4f}"} for n,r in results.items()]
            st.dataframe(pd.DataFrame(rows),width='stretch')
            if results:
                best=max(results.values(),key=lambda r:r.score)
                st.success(f"Best: {best.method} — {best.n_communities} groups")
                fig=renderer.community_distribution(best.labels)
                st.plotly_chart(fig,width='stretch')

# EXPLAIN
if st.session_state.page == "explain":
    st.subheader("Community Explainability")
    if not st.session_state.graph_built or not st.session_state.community_results:
        st.warning("Build and detect first.")
    else:
        G=st.session_state.G; uf=st.session_state.get("user_df"); fn=st.session_state.feature_names
        mn=list(st.session_state.community_results.keys())
        sm=st.selectbox("Method",mn,key="expl_method")
        r=st.session_state.community_results[sm]
        if st.button("Generate",type="primary",key="expl_go"):
            with st.spinner("Analyzing…"):
                report=explainer.explain(G,r.labels,fn,original_df=uf,method=sm,quality_score=r.score)
                st.session_state.explanation_report=report; _save("explanation_report",report)
            st.success(f"Report: {report.n_communities} groups")
            for cid in sorted(report.communities.keys()):
                p=report.communities[cid]; top=p.top_features[:5]
                desc="; ".join([f"**{f.replace('_',' ').replace('nunique','unique').replace('sum','total').replace('mean','avg')}** ({'higher' if d>0 else 'lower'})" for f,d in top]) or "None"
                st.markdown(f"<div class='community-card'><h4>👥 Group {cid} — {p.size} users</h4><p>{desc}</p><p>Sample: {', '.join(p.sample_nodes[:5])}</p></div>",unsafe_allow_html=True)

# VISUALIZE
if st.session_state.page == "visualize":
    st.subheader("User Graph Visualization")
    if not st.session_state.graph_built or st.session_state.G is None:
        st.warning("Build first.")
    else:
        G=st.session_state.G; tw=sum(d.get("weight",1) for _,_,d in G.edges(data=True))
        st.info(f"**{G.number_of_nodes()} users** — **{G.number_of_edges():,} edges** — weight: {tw:,}")
        st.plotly_chart(build_sankey(G),width='stretch')
        etc={}
        for u,v,d in G.edges(data=True):
            et=d.get("edge_type","unknown"); etc[et]=etc.get(et,0)+d.get("weight",1)
        st.dataframe(pd.DataFrame([{"Type":k,"Weight":f"{v:,}"} for k,v in sorted(etc.items(),key=lambda x:-x[1])]),width='stretch')
        
        # vis.js interactive viewer
        st.markdown("### 🕸️ Interactive Graph Viewer")
        # Detect communities on master for coloring
        with st.spinner("Computing layout for interactive view…"):
            try:
                from networkx.algorithms.community import louvain_communities
                comms_raw = list(louvain_communities(G, weight='weight'))
                louvain_comms = [list(c) for c in comms_raw]
            except Exception:
                louvain_comms = [[n] for n in G.nodes()]
            pos = nx.spring_layout(G, seed=42, weight='weight' if G.number_of_edges()>0 else None)
            graph_json = convert_graph_to_json(G, louvain_comms, pos)
        
        vis_html = generate_vis_html(graph_json)
        st.components.v1.html(vis_html, height=600, scrolling=False)
        st.caption("🖱️ Drag nodes, scroll to zoom, hover for details")

        ecols=[c for c in st.session_state.get("feature_names",[]) if c!="creator_user_id"]
        eopts=[f"shared_{c}" for c in ecols]
        sf=st.session_state.get("last_filter",[])
        sel=st.multiselect("Filter:",options=eopts,default=sf,key="viz_edge_filter")
        st.session_state.last_filter=sel
        if sel:
            Gv=nx.Graph()
            for u,v,d in G.edges(data=True):
                if d.get("edge_type") in sel: Gv.add_edge(u,v,**d)
            for n in G.nodes():
                if n in Gv:
                    for k,v in G.nodes[n].items(): Gv.nodes[n][k]=v
        else: Gv=G
        labels=None
        if st.session_state.community_results:
            names=list(st.session_state.community_results.keys())
            sel2=st.selectbox("Color by:",["none"]+names,key="viz_color")
            if sel2!="none":
                fl=st.session_state.community_results[sel2].labels; fn=list(G.nodes())
                lm={}
                for i,n in enumerate(fn):
                    if n in Gv and i<len(fl): lm[n]=int(fl[i])
                if lm: labels=np.array([lm[n] for n in Gv.nodes()])
        with st.spinner("Rendering Plotly…"):
            fig=renderer.plotly(Gv,node_labels=labels,title=f"User Graph — {sel or 'all'}")
            st.plotly_chart(fig,width='stretch')
        c1,c2,c3=st.columns(3)
        c1.metric("Users",Gv.number_of_nodes()); c2.metric("Edges",Gv.number_of_edges())
        degs=[d for _,d in Gv.degree()]; c3.metric("Avg Degree",f"{np.mean(degs):.1f}" if degs else "0")

# EXPORT
if st.session_state.page == "export":
    st.subheader("Export Graphs")
    st.markdown("Export all graphs as JSON (colleague's format) + interactive HTML + download ZIP.")
    if not st.session_state.graph_built or st.session_state.G is None:
        st.warning("Build a user graph first.")
    else:
        G = st.session_state.G
        st.info(f"Exporting master graph: {G.number_of_nodes()} users, {G.number_of_edges():,} edges")
        
        # Detect communities for coloring
        with st.spinner("Computing layout…"):
            try:
                from networkx.algorithms.community import louvain_communities
                comms_raw = list(louvain_communities(G, weight='weight'))
                communities = [list(c) for c in comms_raw]
            except Exception:
                communities = [[n] for n in G.nodes()]
            pos = nx.spring_layout(G, seed=42, weight='weight' if G.number_of_edges()>0 else None)
        
        graph_json = convert_graph_to_json(G, communities, pos)
        
        st.markdown("#### 📄 JSON Preview")
        st.json(graph_json, expanded=False)
        
        st.markdown("#### 🕸️ Interactive HTML Preview")
        vis_html = generate_vis_html(graph_json)
        st.components.v1.html(vis_html, height=600, scrolling=False)
        
        # Download individual files
        st.markdown("#### 📥 Downloads")
        json_str = json.dumps(graph_json, ensure_ascii=False, indent=2)
        b64_json = base64.b64encode(json_str.encode()).decode()
        b64_html = base64.b64encode(vis_html.encode()).decode()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<a href="data:application/json;base64,{b64_json}" download="graph.json">📄 Download JSON</a>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<a href="text/html;base64,{b64_html}" download="graph.html">🌐 Download HTML (vis.js)</a>', unsafe_allow_html=True)
        with col3:
            all_graphs = {"master_graph": (graph_json, communities)}
            st.markdown(get_zip_download_link(all_graphs), unsafe_allow_html=True)

# EMBEDDINGS
if st.session_state.page == "embeddings":
    st.subheader("User Embeddings")
    if not st.session_state.graph_built or st.session_state.G is None:
        st.warning("Build first.")
    else:
        G=st.session_state.G; n=G.number_of_nodes()
        md=max(8,min(128,n-1)) if n>1 else 8
        m=st.selectbox("Method",["laplacian_eigenmaps","node2vec","deepwalk","graphsage","gcn"],key="emb_method")
        d=st.slider("Dimensions",8,md,min(64,md),8,key="emb_dims")
        if st.button("Generate",type="primary",key="emb_go"):
            with st.spinner("Running…"):
                try:
                    ef={"laplacian_eigenmaps":embedder.laplacian,"node2vec":embedder.node2vec,
                        "deepwalk":embedder.deepwalk,"graphsage":embedder.graphsage,"gcn":embedder.gcn}[m]
                    emb=ef(G,dimensions=d)
                    st.session_state.embedding_result=emb; _save("embedding_result",emb)
                    c1,c2,c3=st.columns(3)
                    c1.metric("Shape",f"{emb.embeddings.shape[0]}×{emb.embeddings.shape[1]}")
                    c2.metric("Method",emb.method); c3.metric("Dim",emb.dimension)
                    labs=None
                    if st.session_state.community_results:
                        best=max(st.session_state.community_results.values(),key=lambda r:r.score); labs=best.labels
                    pj=st.selectbox("Projection",["umap","pca","tsne"],key="emb_proj")
                    fig=renderer.embeddings_2d(emb.embeddings,labels=labs,method=pj,title=f"{m} ({pj})")
                    st.plotly_chart(fig,width='stretch')
                except Exception as e: st.error(f"{m} failed: {e}"); st.info("pip install node2vec torch-geometric")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#888'>Plataforma Teste — Análise de Redes Financeiras</div>", unsafe_allow_html=True)
