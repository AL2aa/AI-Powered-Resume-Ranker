import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ai_pipeline import (
    load_model,
    extract_text_from_file,
    process_cv_batch
)

from ai_pipeline import load_model, process_cv_batch

# تهيئة الـ session_state عشان الموديل يكون جاهز في الذاكرة
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None

# وظيفة التأكد من تحميل الموديل
def ensure_model_loaded():
    if st.session_state.tokenizer is None or st.session_state.model is None:
        with st.spinner("🧠 AI is waking up... Loading Model"):
            t, m = load_model()
            st.session_state.tokenizer = t
            st.session_state.model = m



st.set_page_config(page_title="Smart ATS AI", page_icon="✨", layout="wide")

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'landing'
if 'results' not in st.session_state:
    st.session_state.results = None
if 'job_title' not in st.session_state:
    st.session_state.job_title = ''
if 'uploaded_files_names' not in st.session_state:
    st.session_state.uploaded_files_names = []

def switch_to_input():
    st.session_state.current_page = 'input'

def switch_to_landing():
    st.session_state.current_page = 'landing'

def switch_to_results():
    st.session_state.current_page = 'results'

# ─── SHARED CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    header { visibility: hidden; }

    @keyframes slideUp {
        0%   { opacity: 0; transform: translateY(40px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .animate-1 { animation: slideUp 1.2s ease-out forwards; }
    .animate-2 { animation: slideUp 1.2s ease-out 0.3s forwards; opacity: 0; }
    .animate-3 { animation: slideUp 1.2s ease-out 0.6s forwards; opacity: 0; }

    [data-testid="stHeader"] {
        visibility: visible;
        background-color: #f3e8ff !important;
        background-image: linear-gradient(rgba(243,232,255,0.7), rgba(243,232,255,0.7)),
                          url("https://i.postimg.cc/mD8zM9fS/9w-Fx-L2r4.jpg") !important;
        background-size: cover;
        height: 60px !important;
        border-bottom: 1px solid #e9d5ff;
    }

    .stApp {
        background: linear-gradient(rgba(255,255,255,0.6), rgba(255,255,255,0.6)),
                    url("https://i.postimg.cc/k57CDHgB/tnzyl.jpg") !important;
        background-size: cover !important;
        background-attachment: fixed !important;
    }

    .block-container { padding-top: 0rem !important; max-width: 100% !important; }

    div.stButton > button {
        border-radius: 50px !important;
        padding: 12px 45px !important;
        font-weight: bold !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-5px) scale(1.05) !important;
        box-shadow: 0 10px 20px rgba(168, 85, 247, 0.3) !important;
    }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #a855f7 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
    }

    div[data-baseweb="input"], div[data-baseweb="textarea"] {
        background-color: white !important;
        border: 2px solid #e9d5ff !important;
        border-radius: 10px !important;
    }

    .input-card {
        display: none;
        background: rgba(255,255,255,0.9) !important;
        padding: 35px;
        border-radius: 25px;
        border: 1px solid #f3e8ff;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        backdrop-filter: blur(10px);
    }

    .smart-banner {
        background: rgba(168,85,247,0.1) !important;
        border-left: 5px solid #a855f7;
        padding: 15px 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: #5b21b6;
        font-size: 14px;
        backdrop-filter: blur(5px);
    }

    /* ── Results page styles ── */
    .res-wrap {
        background: rgba(255,255,255,0.92);
        border-radius: 24px;
        padding: 36px 40px;
        margin: 30px auto;
        max-width: 1100px;
        box-shadow: 0 8px 32px rgba(124,58,237,0.08);
    }
    .page-title { font-size: 22px; font-weight: 700; color: #1e293b; margin: 0; }
    .page-sub   { font-size: 14px; color: #64748b; margin: 4px 0 0; }

    .stat-grid { display: flex; gap: 14px; margin: 20px 0 28px; }
    .stat-card {
        flex: 1;
        background: #f8f5ff;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #e9d5ff;
    }
    .stat-val { font-size: 28px; font-weight: 700; color: #1e293b; }
    .stat-val.green { color: #16a34a; }
    .stat-val.red   { color: #dc2626; }
    .stat-val.purple{ color: #7c3aed; }
    .stat-lbl { font-size: 12px; color: #64748b; margin-top: 2px; }

    .section-label {
        font-size: 11px; font-weight: 700; color: #7c3aed;
        letter-spacing: 1.5px; text-transform: uppercase;
        margin: 0 0 14px;
    }

    .top5-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 32px; }
    .top-card {
        flex: 1 1 160px;
        background: white;
        border: 1px solid #e9d5ff;
        border-radius: 18px;
        padding: 18px 16px;
        position: relative;
        min-width: 150px;
    }
    .top-card.rank1 { border: 2px solid #a855f7; }

    .rank-badge {
        position: absolute; top: 10px; right: 10px;
        width: 24px; height: 24px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 700;
    }
    .r1 { background: #ede9fe; color: #7c3aed; }
    .r2 { background: #e2e8f0; color: #475569; }
    .r3 { background: #fef3c7; color: #b45309; }
    .r4,.r5 { background: #f1f5f9; color: #64748b; }

    .avatar {
        width: 42px; height: 42px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 13px; font-weight: 700; margin-bottom: 10px;
    }
    .av1{background:#ede9fe;color:#7c3aed;}
    .av2{background:#d1fae5;color:#065f46;}
    .av3{background:#fef3c7;color:#b45309;}
    .av4{background:#fee2e2;color:#b91c1c;}
    .av5{background:#dbeafe;color:#1d4ed8;}

    .card-name  { font-size: 14px; font-weight: 600; color: #1e293b; }
    .card-role  { font-size: 11px; color: #64748b; margin-bottom: 10px; }
    .card-score-num { font-size: 22px; font-weight: 700; color: #1e293b; }
    .card-score-lbl { font-size: 11px; color: #64748b; }
    .skills-row { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
    .skill-pill {
        background: #f3e8ff; color: #7c3aed;
        border-radius: 20px; font-size: 10px; padding: 2px 8px;
    }

    .res-table {
        width: 100%; border-collapse: collapse;
        font-size: 13px; margin-bottom: 0;
    }
    .res-table th {
        background: #f8f5ff;
        padding: 14px 18px; text-align: left;
        font-size: 12px; font-weight: 700; color: #64748b;
        border-bottom: 1px solid #e9d5ff;
    }
    .res-table td {
        padding: 16px 18px;
        border-bottom: 1px solid #f1f5f9;
        color: #1e293b; vertical-align: middle;
    }
    .res-table tr:last-child td { border-bottom: none; }

    .badge {
        display: inline-block; padding: 6px 14px;
        border-radius: 20px; font-size: 11px; font-weight: 600;
    }
    .badge-accept { background: #dcfce7; color: #16a34a; }
    .badge-reject { background: #fee2e2; color: #b91c1c; }

    .bar-wrap {
        width: 100px; height: 6px;
        background: #e9d5ff; border-radius: 3px;
        display: inline-block; vertical-align: middle; margin-right: 12px;
    }
    .bar-fill { height: 6px; border-radius: 3px; }

    /* CEO modal */
    .ceo-overlay {
        position: fixed; inset: 0;
        background: rgba(0,0,0,0.45);
        display: flex; align-items: center; justify-content: center;
        z-index: 9999;
    }
    .ceo-modal {
        background: white; border-radius: 20px;
        padding: 36px 40px; max-width: 420px; width: 90%;
        text-align: center; box-shadow: 0 24px 64px rgba(124,58,237,0.18);
    }
    .ceo-crown-circle {
        width: 56px; height: 56px; background: #fef3c7;
        border-radius: 50%; display: flex; align-items: center;
        justify-content: center; margin: 0 auto 14px; font-size: 26px;
    }
    .ceo-tag {
        font-size: 10px; font-weight: 800; color: #b45309;
        letter-spacing: 2px; margin-bottom: 8px;
    }
    .ceo-name   { font-size: 22px; font-weight: 700; color: #1e293b; }
    .ceo-score  { font-size: 13px; color: #64748b; margin-bottom: 16px; }
    .ceo-reason {
        font-size: 13px; color: #475569; line-height: 1.7;
        background: #f8f5ff; border-radius: 12px;
        padding: 14px 18px; margin-bottom: 20px; text-align: left;
    }
    .ceo-close-btn {
        background: linear-gradient(135deg, #a855f7, #7c3aed);
        color: white; border: none; border-radius: 25px;
        padding: 10px 32px; font-size: 13px; font-weight: 700;
        cursor: pointer;
    }

    .back-btn-res {
        display: inline-flex; align-items: center; gap: 6px;
        background: white; border: 1px solid #e9d5ff;
        border-radius: 20px; padding: 7px 18px;
        font-size: 12px; color: #7c3aed; font-weight: 600;
        cursor: pointer; margin-bottom: 6px; text-decoration: none;
    }
    .section-header {
        display: flex; align-items: center;
        justify-content: space-between; margin-bottom: 14px;
    }
    .ceo-open-btn {
        background: linear-gradient(135deg, #a855f7, #7c3aed);
        color: white; border: none; border-radius: 25px;
        padding: 9px 22px; font-size: 13px; font-weight: 700;
        cursor: pointer; display: flex; align-items: center; gap: 6px;
    }
    .table-card {
        background: white; border: 1px solid #e9d5ff;
        border-radius: 18px; overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — LANDING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.current_page == 'landing':
    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.markdown('<div style="padding: 8% 5% 5% 10%;">', unsafe_allow_html=True)
        st.markdown('<p class="animate-1" style="color:#a855f7;font-weight:bold;letter-spacing:1.5px;">✨ THE NEXT GEN RECRUITMENT</p>', unsafe_allow_html=True)
        st.markdown('<h1 class="animate-1" style="font-size:52px;font-weight:800;color:#1e293b;line-height:1.1;margin-bottom:20px;">YOUR AI PARTNER<br>FOR SMART<br>HIRING DECISIONS</h1>', unsafe_allow_html=True)
        st.markdown('<p class="animate-2" style="font-size:18px;color:#4a5568;line-height:1.6;">Automate resume screening, discover top talent, and boost hiring accuracy using Multi-Agent AI technology.</p>', unsafe_allow_html=True)
        st.markdown('<div class="animate-3">', unsafe_allow_html=True)
        st.button("START AI ANALYSIS", on_click=switch_to_input)
        st.markdown('</div></div>', unsafe_allow_html=True)
    with col2:
        side_img = "https://i.postimg.cc/c4nJwQxz/Recruiter-looks-at-a-perfect-candidate-cv-illustration.jpg"
        st.markdown(f'<div class="animate-2" style="margin-top:100px;display:flex;justify-content:center;"><img src="{side_img}" style="width:82%;border-radius:30px;"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — INPUT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == 'input':
    st.write("")
    col_main, col_side = st.columns([1.3, 0.7], gap="medium")

    with col_main:
        st.markdown('<div style="padding-left:10%;padding-top:50px;">', unsafe_allow_html=True)

        if st.button("← Back to Home"):
            switch_to_landing()
            st.rerun()

        st.markdown("""
            <div class="smart-banner">
                <b>Pro Tip:</b> Make sure your Job Description includes key skills.
                Our AI agents use this to rank candidates more accurately!
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="color:#1e293b;margin-top:0;">Add Your Details</h2>', unsafe_allow_html=True)

        job_title   = st.text_input("Desired Job Title", placeholder="e.g: Data Scientist")
        job_desc    = st.text_area("Job Description", placeholder="Paste job requirements here...", height=120)
        uploaded    = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

        if st.button("Proceed to Analysis"):
            if not job_title or not job_desc or not uploaded:
                st.warning("Please complete all fields and upload CVs.")
            else:
                # 1. تحميل الموديل لو مش متحمل
                ensure_model_loaded()
                
                with st.spinner("AI is evaluating CVs... This may take a minute."):
                    # 2. تحضير الملفات (اسم الملف : محتواه)
                    file_dict = {file.name: file.getvalue() for file in uploaded}
                    
                    # 3. تشغيل الموديل الحقيقي
                    results_df = process_cv_batch(
                        file_dict, job_title, job_desc, 
                        st.session_state.tokenizer, 
                        st.session_state.model
                    )
                    
                    # 4. تحويل النتائج للجدول (بدون راندوم)
                    candidates = []
                    for _, row in results_df.iterrows():
                        filename = row['Candidate File']
                        display_name = filename.split('.')[0].replace('_', ' ').title()
                        
                        candidates.append({
                            "name": display_name,
                            "score": int(row['AI Score']),
                            "role": job_title,
                            "status": "Accepted" if row['AI Score'] >= 65 else "Rejected",
                            "exp": "Check CV", 
                            "skills": ["Python", "AI", "ML"],
                            "initials": "".join([w[0] for w in display_name.split()][:2]).upper()
                        })

                    st.session_state.results = candidates
                    st.session_state.job_title = job_title
                    st.session_state.current_page = 'results'
                    st.rerun()
                

        st.markdown('</div></div>', unsafe_allow_html=True)

    with col_side:
        analysis_img = "https://i.postimg.cc/7ZXfX0cW/Analyze-Customizable-Flat-Illustrations-Rafiki-Style-removebg-preview.png"
        st.markdown('<div style="margin-top:100px;padding-right:10%;">', unsafe_allow_html=True)
        st.image(analysis_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.current_page == 'results':
    candidates = st.session_state.results or []
    job_title = st.session_state.job_title or "Position"

    df = pd.DataFrame(candidates)

    total_candidates = len(df)
    accepted_count = len(df[df['status'] == 'Accepted'])
    rejected_count = len(df[df['status'] == 'Rejected'])
    avg_score = int(df['score'].mean()) if total_candidates > 0 else 0
    top_score = int(df['score'].max()) if total_candidates > 0 else 0

    st.markdown("""
    <style>
    .dashboard-container {
        padding: 20px;
    }

    .dashboard-title {
        font-size: 34px;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 5px;
    }

    .dashboard-sub {
        color: #64748b;
        margin-bottom: 25px;
        font-size: 15px;
    }

    .card {
        display: none;
        background: white;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #f1f5f9;
    }

    .mini-card {
        background: linear-gradient(135deg,#f8f5ff,#ffffff);
        border-radius: 18px;
        padding: 20px;
        border: 1px solid #ede9fe;
    }

    .mini-title {
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
    }

    .mini-value {
        font-size: 32px;
        font-weight: 800;
        color: #7c3aed;
        margin-top: 8px;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 15px;
    }

    .small-stat {
        background: white;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #f1f5f9;
        text-align: center;
    }

    .small-num {
        font-size: 28px;
        font-weight: 800;
        color: #7c3aed;
    }

    .small-label {
        color: #64748b;
        font-size: 13px;
    }

    .top-candidate {
        background: linear-gradient(135deg,#7c3aed,#a855f7);
        color: white;
        padding: 20px;
        border-radius: 22px;
        margin-top: 15px;
    }

    .candidate-name {
        font-size: 24px;
        font-weight: 700;
    }

    .candidate-role {
        opacity: 0.9;
        margin-bottom: 15px;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)

    # HEADER

    left_head, right_head = st.columns([4,1])

    with left_head:
        st.markdown(f'<div class="dashboard-title">Analytics Dashboard</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="dashboard-sub">AI Recruitment Intelligence for {job_title}</div>', unsafe_allow_html=True)

    with right_head:
        st.write('')
        st.write('')
        if st.button('← New Analysis'):
            st.session_state.current_page = 'input'
            st.rerun()

    # TOP CARDS

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f'''
        <div class="mini-card">
            <div class="mini-title">TOTAL CANDIDATES</div>
            <div class="mini-value">{total_candidates}</div>
        </div>
        ''', unsafe_allow_html=True)

    with c2:
        st.markdown(f'''
        <div class="mini-card">
            <div class="mini-title">ACCEPTED</div>
            <div class="mini-value">{accepted_count}</div>
        </div>
        ''', unsafe_allow_html=True)

    with c3:
        st.markdown(f'''
        <div class="mini-card">
            <div class="mini-title">REJECTED</div>
            <div class="mini-value">{rejected_count}</div>
        </div>
        ''', unsafe_allow_html=True)

    with c4:
        st.markdown(f'''
        <div class="mini-card">
            <div class="mini-title">AVG SCORE</div>
            <div class="mini-value">{avg_score}%</div>
        </div>
        ''', unsafe_allow_html=True)

    st.write('')

    # MAIN SECTION

    left_side, right_side = st.columns([2.5,1])

    with left_side:

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Candidate Score Analysis</div>', unsafe_allow_html=True)

        if total_candidates > 0:
            fig_line = px.area(
                df,
                x='name',
                y='score',
                markers=True,
                line_shape='spline'
            )

            fig_line.update_traces(
                line_color='#7c3aed',
                fillcolor='rgba(124,58,237,0.2)'
            )

            fig_line.update_layout(
                height=350,
                paper_bgcolor='white',
                plot_bgcolor='white',
                margin=dict(l=0,r=0,t=0,b=0),
                xaxis_title='Candidates',
                yaxis_title='Score',
                xaxis_tickangle=-25,
                font=dict(color='#1e293b')
            )

            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.write('')

        bottom_left1, bottom_left2 = st.columns(2)

        with bottom_left1:

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Accepted vs Rejected</div>', unsafe_allow_html=True)

            if total_candidates > 0:
                fig_pie = px.pie(
                    df,
                    names='status',
                    hole=0.7,
                    color='status',
                    color_discrete_map={
                        'Accepted':'#7c3aed',
                        'Rejected':'#e9d5ff'
                    }
                )

                fig_pie.update_layout(
                    height=350,
                    margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor='white'
                )

                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with bottom_left2:

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top Candidate</div>', unsafe_allow_html=True)

            if total_candidates > 0:
                top_candidate = df.sort_values(by='score', ascending=False).iloc[0]

                st.markdown(f'''
                <div class="top-candidate">
                    <div class="candidate-name">{top_candidate['name']}</div>
                    <div class="candidate-role">{top_candidate['role']}</div>
                    <h1>{top_candidate['score']}%</h1>
                    <p>Best Match For {job_title}</p>
                </div>
                ''', unsafe_allow_html=True)

                st.write('')

                skills = top_candidate['skills']

                skills_html = """
                                <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:10px;">
                                                """

                for skill in skills:
                    skills_html += f'''
                    <span style="
                    background:#f3e8ff;
                    color:#7c3aed;
                    padding:8px 14px;
                    border-radius:20px;
                    font-size:13px;
                    font-weight:600;
                    display:inline-block;">
                    {skill}
                    </span>
                    '''

                skills_html += '</div>'

                st.markdown(skills_html, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with right_side:

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Score</div>', unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = top_score,
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': '#7c3aed'},
                'bgcolor': 'white',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 50], 'color': '#ede9fe'},
                    {'range': [50, 100], 'color': '#ddd6fe'}
                ]
            }
        ))

        fig_gauge.update_layout(
            height=280,
            margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor='white'
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.write('')

        small1, small2 = st.columns(2)

        with small1:
            st.markdown(f'''
            <div class="small-stat">
                <div class="small-num">{accepted_count}</div>
                <div class="small-label">Shortlisted</div>
            </div>
            ''', unsafe_allow_html=True)

        with small2:
            st.markdown(f'''
            <div class="small-stat">
                <div class="small-num">{rejected_count}</div>
                <div class="small-label">Rejected</div>
            </div>
            ''', unsafe_allow_html=True)

        st.write('')

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">AI Recommendation</div>', unsafe_allow_html=True)

        if total_candidates > 0:
            st.success(f"{top_candidate['name']} is the best candidate for this role.")

            st.markdown(f"""
            ### Why?

            - Highest matching score.
            - Strong skill alignment.
            - Suitable experience level.
            - Passed AI screening successfully.
            """)

        st.markdown('</div>', unsafe_allow_html=True)

    st.write('')

    # ──────────────────────────────────────────────────────────────────────────
    #  MODIFIED SECTION: Detailed Candidate Matrix
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top: 30px;">Detailed Candidate Matrix</div>', unsafe_allow_html=True)

    table_html = '<div class="table-card" style="overflow-x: auto;">'
    table_html += '<table class="res-table">'
    table_html += '<thead><tr>'
    table_html += '<th>Rank</th><th>Candidate</th><th>Score</th><th>Skills match</th><th>Experience</th><th>Status</th>'
    table_html += '</tr></thead><tbody>'

    for idx, row in enumerate(candidates):
        rank = idx + 1
        
        # Determine rank badge class
        if rank == 1:
            r_class = "r1"
        elif rank == 2:
            r_class = "r2"
        elif rank == 3:
            r_class = "r3"
        elif rank == 4:
            r_class = "r4"
        else:
            r_class = "r5"

        name = row['name']
        role = row['role']
        score = row['score']
        exp = row['exp']
        status = row['status']
        n_skills = len(row['skills'])

        status_class = "badge-accept" if status == "Accepted" else "badge-reject"
        
        # Appending HTML without newlines/indentation to prevent Markdown code-block bugs
        table_html += "<tr>"
        table_html += f"<td><div style='position: relative; width: 30px; height: 30px;'><div class='rank-badge {r_class}' style='position: absolute; top: 0; left: 0; right: auto;'>#{rank}</div></div></td>"
        table_html += f"<td><div class='card-name' style='margin-bottom: 2px;'>{name}</div><div class='card-role' style='margin-bottom: 0;'>{role}</div></td>"
        table_html += f"<td><div class='bar-wrap'><div class='bar-fill' style='width: {score}%; background: #a855f7;'></div></div><span style='font-size: 13px; font-weight: 700; color: #64748b;'>{score}%</span></td>"
        table_html += f"<td><span style='font-size: 13px; color: #475569; font-weight: 500;'>{n_skills} / 10</span></td>"
        table_html += f"<td><span style='font-size: 13px; color: #475569; font-weight: 500;'>{exp} yrs</span></td>"
        table_html += f"<td><span class='badge {status_class}'>{status}</span></td>"
        table_html += "</tr>"

    table_html += '</tbody></table></div>'

    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)