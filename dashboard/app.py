"""
Predictive Policing Decision Support System - Premium Dashboard
================================================================
A modern, visually stunning Streamlit dashboard for crime analytics

Features:
- Dark/Light theme support
- Glassmorphism design elements
- Smooth animations
- Interactive visualizations
- Professional color palette

Run with: python -m streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Crime Analytics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS Styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
</style>
<link href="https://cdn.jsdelivr.net/npm/remixicon@3.5.0/fonts/remixicon.css" rel="stylesheet">
<style>
    
    /* Root variables for theming */
    :root {
        --primary-color: #6366f1;
        --primary-hover: #4f46e5;
        --secondary-color: #10b981;
        --accent-color: #f59e0b;
        --danger-color: #ef4444;
        --background-dark: #0f172a;
        --background-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border-color: #334155;
        --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-2: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-3: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --glass-bg: rgba(30, 41, 59, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Custom header */
    .premium-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 2rem 3rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-1);
    }
    
    .premium-header h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .premium-header p {
        font-family: 'Inter', sans-serif;
        color: var(--text-secondary);
        font-size: 1rem;
        margin: 0;
    }
    
    /* Metric cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2);
        border-color: var(--primary-color);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        border-radius: 4px 0 0 4px;
    }
    
    .metric-card.blue::before { background: var(--gradient-3); }
    .metric-card.purple::before { background: var(--gradient-1); }
    .metric-card.green::before { background: linear-gradient(135deg, #10b981, #34d399); }
    .metric-card.orange::before { background: linear-gradient(135deg, #f59e0b, #fbbf24); }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border-color) 0%, transparent 100%);
    }
    
    /* Glass cards */
    .glass-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
    }
    
    /* Alert boxes */
    .alert-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 4px solid var(--accent-color);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .alert-info {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(99, 102, 241, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-left: 4px solid var(--primary-color);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 4px solid var(--secondary-color);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 4px solid var(--danger-color);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        background: transparent;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: var(--primary-color);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--glass-bg);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        padding: 0.75rem 1.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: var(--glass-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--gradient-1);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--glass-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-low { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .risk-medium { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .risk-high { background: rgba(239, 68, 68, 0.2); color: #f87171; }
    
    /* Animated gradient text */
    .gradient-text {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 5s ease infinite;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Pulse animation for live indicator */
    .live-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: var(--secondary-color);
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        border-top: 1px solid var(--border-color);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Color scheme for Plotly charts
CHART_COLORS = {
    'primary': '#6366f1',
    'secondary': '#10b981',
    'accent': '#f59e0b',
    'danger': '#ef4444',
    'purple': '#a855f7',
    'pink': '#ec4899',
    'blue': '#3b82f6',
    'cyan': '#06b6d4',
}

PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'family': 'Inter, sans-serif', 'color': '#f1f5f9'},
        'xaxis': {
            'gridcolor': 'rgba(51, 65, 85, 0.5)',
            'linecolor': 'rgba(51, 65, 85, 0.5)',
            'tickfont': {'color': '#94a3b8'}
        },
        'yaxis': {
            'gridcolor': 'rgba(51, 65, 85, 0.5)',
            'linecolor': 'rgba(51, 65, 85, 0.5)',
            'tickfont': {'color': '#94a3b8'}
        },
        'colorway': ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#a855f7', '#ec4899', '#3b82f6', '#06b6d4']
    }
}


@st.cache_data
def load_data():
    """Load and cache the crime data"""
    try:
        df = pd.read_csv("data/raw/dstrIPC_2013.csv")
        df = df[~df['DISTRICT'].str.contains('TOTAL|RLY|G.R.P|CID|STF', case=False, na=False)]
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure data/raw/dstrIPC_2013.csv exists.")
        return None


def render_header():
    """Render the premium header"""
    st.markdown("""
    <div class="premium-header">
        <h1><i class="ri-shield-check-line" style="margin-right: 0.5rem;"></i>Crime Analytics Dashboard</h1>
        <p><span class="live-indicator"></span>Predictive Policing Decision Support System • Data Year: 2013</p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(df):
    """Render metric cards"""
    total_states = df['STATE/UT'].nunique()
    total_districts = len(df)
    total_crimes = df['TOTAL IPC CRIMES'].sum()
    avg_crimes = df['TOTAL IPC CRIMES'].mean()
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card blue">
            <div class="metric-icon"><i class="ri-government-line"></i></div>
            <div class="metric-value">{total_states}</div>
            <div class="metric-label">States & UTs</div>
        </div>
        <div class="metric-card purple">
            <div class="metric-icon"><i class="ri-map-pin-line"></i></div>
            <div class="metric-value">{total_districts:,}</div>
            <div class="metric-label">Districts</div>
        </div>
        <div class="metric-card green">
            <div class="metric-icon"><i class="ri-bar-chart-box-line"></i></div>
            <div class="metric-value">{total_crimes:,.0f}</div>
            <div class="metric-label">Total Reported Crimes</div>
        </div>
        <div class="metric-card orange">
            <div class="metric-icon"><i class="ri-line-chart-line"></i></div>
            <div class="metric-value">{avg_crimes:,.0f}</div>
            <div class="metric-label">Avg per District</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_ethical_disclaimer():
    """Render ethical disclaimer"""
    with st.expander("⚠️ Important: Ethical Guidelines & Data Limitations", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="alert-success">
                <strong>✅ Appropriate Uses</strong><br>
                • Resource allocation planning<br>
                • Academic research<br>
                • Policy analysis<br>
                • Social program targeting
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="alert-danger">
                <strong>❌ Inappropriate Uses</strong><br>
                • Individual targeting<br>
                • Community profiling<br>
                • Automated policing<br>
                • Discriminatory practices
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-warning">
            <strong>⚠️ Key Limitation:</strong> This data reflects <em>reported</em> crime only. 
            Higher numbers may indicate better reporting infrastructure, not necessarily higher actual crime rates.
        </div>
        """, unsafe_allow_html=True)


def render_state_analysis(df):
    """Render state-wise analysis with premium charts"""
    st.markdown('<div class="section-header"><i class="ri-pie-chart-2-line"></i> State-wise Crime Distribution</div>', unsafe_allow_html=True)
    
    # Aggregate by state
    state_df = df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().reset_index()
    state_df = state_df.sort_values('TOTAL IPC CRIMES', ascending=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Horizontal bar chart with gradient
        fig = go.Figure()
        
        # Add bars with color gradient
        colors = px.colors.sequential.Viridis[::-1]
        n_colors = len(colors)
        bar_colors = [colors[int(i * n_colors / len(state_df.tail(15)))] for i in range(len(state_df.tail(15)))]
        
        fig.add_trace(go.Bar(
            y=state_df.tail(15)['STATE/UT'],
            x=state_df.tail(15)['TOTAL IPC CRIMES'],
            orientation='h',
            marker=dict(
                color=state_df.tail(15)['TOTAL IPC CRIMES'],
                colorscale='Viridis',
                line=dict(width=0)
            ),
            hovertemplate='<b>%{y}</b><br>Crimes: %{x:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            **PLOTLY_TEMPLATE['layout'],
            title=dict(text='Top 15 States by Total Reported Crimes', x=0.5),
            height=500,
            showlegend=False,
            xaxis_title='Total IPC Crimes',
            yaxis_title='',
            margin=dict(l=20, r=20, t=60, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 Top 5 States")
        
        top_5 = state_df.tail(5).iloc[::-1]
        for i, (_, row) in enumerate(top_5.iterrows()):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            st.markdown(f"""
            <div style="padding: 0.75rem; background: rgba(99, 102, 241, {0.2 - i*0.03}); 
                        border-radius: 8px; margin-bottom: 0.5rem;">
                <span style="font-size: 1.25rem;">{medal}</span>
                <strong style="color: #f1f5f9;">{row['STATE/UT']}</strong><br>
                <span style="color: #94a3b8;">{row['TOTAL IPC CRIMES']:,.0f} crimes</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_crime_categories(df):
    """Render crime category analysis"""
    st.markdown('<div class="section-header"><i class="ri-search-line"></i> Crime Category Breakdown</div>', unsafe_allow_html=True)
    
    categories = {
        'MURDER': ('Murder', '#ef4444'),
        'RAPE': ('Rape', '#f97316'),
        'KIDNAPPING & ABDUCTION': ('Kidnapping', '#eab308'),
        'ROBBERY': ('Robbery', '#22c55e'),
        'BURGLARY': ('Burglary', '#14b8a6'),
        'THEFT': ('Theft', '#06b6d4'),
        'RIOTS': ('Riots', '#3b82f6'),
        'CHEATING': ('Cheating', '#6366f1'),
        'DOWRY DEATHS': ('Dowry Deaths', '#a855f7'),
        'CRUELTY BY HUSBAND OR HIS RELATIVES': ('Domestic Violence', '#ec4899')
    }
    
    category_data = []
    for col, (name, color) in categories.items():
        if col in df.columns:
            category_data.append({
                'Category': name,
                'Total': df[col].sum(),
                'Color': color
            })
    
    cat_df = pd.DataFrame(category_data).sort_values('Total', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Donut chart
        fig = go.Figure(data=[go.Pie(
            labels=cat_df['Category'],
            values=cat_df['Total'],
            hole=0.6,
            marker=dict(colors=cat_df['Color']),
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>Cases: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            **PLOTLY_TEMPLATE['layout'],
            title=dict(text='Crime Distribution by Category', x=0.5),
            height=450,
            showlegend=False,
            annotations=[dict(
                text=f"<b>{cat_df['Total'].sum():,.0f}</b><br>Total",
                x=0.5, y=0.5,
                font_size=16,
                font_color='#f1f5f9',
                showarrow=False
            )]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Treemap
        fig = px.treemap(
            cat_df,
            path=['Category'],
            values='Total',
            color='Total',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            **PLOTLY_TEMPLATE['layout'],
            title=dict(text='Crime Categories Treemap', x=0.5),
            height=450,
            margin=dict(l=10, r=10, t=60, b=10)
        )
        
        fig.update_traces(
            textinfo='label+value',
            hovertemplate='<b>%{label}</b><br>Cases: %{value:,.0f}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_bias_analysis(df):
    """Render reporting bias analysis"""
    st.markdown('<div class="section-header"><i class="ri-scales-3-line"></i> Reporting Bias Indicators</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-info">
        <strong>🔬 Why This Matters:</strong> High variance in crime reporting across districts 
        within a state may indicate inconsistent reporting practices rather than actual crime differences.
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate CV by state
    state_stats = df.groupby('STATE/UT')['TOTAL IPC CRIMES'].agg(['mean', 'std', 'count'])
    state_stats['cv'] = (state_stats['std'] / state_stats['mean']) * 100
    state_stats = state_stats.sort_values('cv', ascending=False).reset_index()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # CV chart
        fig = go.Figure()
        
        top_cv = state_stats.head(15)
        colors = ['#ef4444' if cv > 100 else '#f59e0b' if cv > 50 else '#10b981' 
                  for cv in top_cv['cv']]
        
        fig.add_trace(go.Bar(
            x=top_cv['STATE/UT'],
            y=top_cv['cv'],
            marker=dict(color=colors),
            hovertemplate='<b>%{x}</b><br>CV: %{y:.1f}%<extra></extra>'
        ))
        
        # Add threshold lines
        fig.add_hline(y=100, line_dash="dash", line_color="#ef4444", 
                      annotation_text="High Risk Threshold", annotation_position="right")
        fig.add_hline(y=50, line_dash="dash", line_color="#f59e0b",
                      annotation_text="Medium Risk Threshold", annotation_position="right")
        
        fig.update_layout(
            **PLOTLY_TEMPLATE['layout'],
            title=dict(text='Coefficient of Variation by State', x=0.5),
            height=400,
            xaxis_tickangle=-45,
            yaxis_title='CV (%)',
            margin=dict(l=40, r=40, t=60, b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Risk Distribution")
        
        high_risk = len(state_stats[state_stats['cv'] > 100])
        medium_risk = len(state_stats[(state_stats['cv'] > 50) & (state_stats['cv'] <= 100)])
        low_risk = len(state_stats[state_stats['cv'] <= 50])
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=high_risk,
            title={'text': "High-Risk States", 'font': {'color': '#f1f5f9', 'size': 16}},
            delta={'reference': 0, 'increasing': {'color': "#ef4444"}},
            gauge={
                'axis': {'range': [0, len(state_stats)], 'tickcolor': '#94a3b8'},
                'bar': {'color': "#ef4444"},
                'bgcolor': "rgba(30, 41, 59, 0.8)",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, low_risk], 'color': 'rgba(16, 185, 129, 0.3)'},
                    {'range': [low_risk, low_risk + medium_risk], 'color': 'rgba(245, 158, 11, 0.3)'},
                    {'range': [low_risk + medium_risk, len(state_stats)], 'color': 'rgba(239, 68, 68, 0.3)'}
                ]
            }
        ))
        
        fig.update_layout(
            **PLOTLY_TEMPLATE['layout'],
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div style="text-align: center;">
            <span class="risk-badge risk-high">🔴 High: {high_risk}</span>
            <span class="risk-badge risk-medium">🟡 Medium: {medium_risk}</span>
            <span class="risk-badge risk-low">🟢 Low: {low_risk}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_risk_predictions(df):
    """Render risk predictions with interactive state selector"""
    st.markdown('<div class="section-header"><i class="ri-focus-3-line"></i> Crime Risk Predictions</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-warning">
        <strong>⚠️ Prediction Disclaimer:</strong> These risk categories are based on 
        historical reported crime data. They are for resource planning purposes only.
    </div>
    """, unsafe_allow_html=True)
    
    # State selector
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_state = st.selectbox(
            "🏛️ Select State for Analysis",
            options=sorted(df['STATE/UT'].unique()),
            index=0
        )
    
    # Filter and categorize
    state_districts = df[df['STATE/UT'] == selected_state].copy()
    state_districts['Risk Category'] = pd.qcut(
        state_districts['TOTAL IPC CRIMES'].rank(method='first'),
        q=3,
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    state_districts = state_districts.sort_values('TOTAL IPC CRIMES', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # District chart
        colors = {'LOW': '#10b981', 'MEDIUM': '#f59e0b', 'HIGH': '#ef4444'}
        
        fig = go.Figure()
        
        for risk in ['HIGH', 'MEDIUM', 'LOW']:
            subset = state_districts[state_districts['Risk Category'] == risk]
            fig.add_trace(go.Bar(
                x=subset['DISTRICT'],
                y=subset['TOTAL IPC CRIMES'],
                name=risk,
                marker_color=colors[risk],
                hovertemplate='<b>%{x}</b><br>Crimes: %{y:,.0f}<br>Risk: ' + risk + '<extra></extra>'
            ))
        
        fig.update_layout(
            **PLOTLY_TEMPLATE['layout'],
            title=dict(text=f'District Risk Assessment: {selected_state}', x=0.5),
            height=400,
            barmode='group',
            xaxis_tickangle=-90,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=40, r=40, t=80, b=120)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"### 📋 {selected_state} Summary")
        
        for risk, color, icon in [('HIGH', '#ef4444', '🔴'), ('MEDIUM', '#f59e0b', '🟡'), ('LOW', '#10b981', '🟢')]:
            count = len(state_districts[state_districts['Risk Category'] == risk])
            pct = count / len(state_districts) * 100
            st.markdown(f"""
            <div style="padding: 0.75rem; background: {color}22; 
                        border-left: 3px solid {color}; border-radius: 8px; margin-bottom: 0.5rem;">
                <strong style="color: {color};">{icon} {risk}</strong>
                <span style="float: right; color: #f1f5f9;">{count} districts ({pct:.0f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Interpretation")
        st.markdown("""
        - **🔴 HIGH**: Higher resource allocation needed
        - **🟡 MEDIUM**: Standard monitoring required  
        - **🟢 LOW**: May indicate under-reporting
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_ethics_page():
    """Render comprehensive ethics page"""
    st.markdown('<div class="section-header"><i class="ri-file-text-line"></i> Ethical Framework & Guidelines</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Guidelines", "Risks", "References"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #10b981;">✅ Responsible Use</h3>
                <ul style="color: #94a3b8;">
                    <li>Use for <strong>resource allocation</strong> planning</li>
                    <li>Support <strong>policy research</strong> and analysis</li>
                    <li>Identify areas needing <strong>social services</strong></li>
                    <li>Plan infrastructure <strong>development</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #ef4444;">❌ Prohibited Use</h3>
                <ul style="color: #94a3b8;">
                    <li>Never use for <strong>individual profiling</strong></li>
                    <li>Never automate <strong>law enforcement</strong> decisions</li>
                    <li>Never target <strong>communities</strong> or groups</li>
                    <li>Never ignore <strong>socioeconomic context</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="alert-danger">
            <h4>🔴 Feedback Loops</h4>
            <p>More policing → More arrests → "Higher crime" → More policing</p>
            <p><strong>Mitigation:</strong> Use crime reports, not arrests; regular bias audits</p>
        </div>
        
        <div class="alert-warning">
            <h4>🟡 Bias Amplification</h4>
            <p>Historical bias in data gets encoded into ML predictions</p>
            <p><strong>Mitigation:</strong> State-level analysis only; no demographic features</p>
        </div>
        
        <div class="alert-info">
            <h4>🔵 Over-reliance Risk</h4>
            <p>Decision-makers may trust ML predictions over human judgment</p>
            <p><strong>Mitigation:</strong> Mandatory human review; confidence scores displayed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="glass-card">
            <h3>📚 Academic References</h3>
            <ol style="color: #94a3b8;">
                <li>Richardson, R., Schultz, J., & Crawford, K. (2019). <em>Dirty Data, Bad Predictions</em>. NYU Law Review.</li>
                <li>Lum, K., & Isaac, W. (2016). <em>To predict and serve?</em> Significance, 13(5).</li>
                <li>Ferguson, A. G. (2017). <em>The Rise of Big Data Policing</em>. NYU Press.</li>
                <li>NCRB India. (2013). <em>Crime in India Statistics</em>.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """Render premium sidebar"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <i class="ri-shield-check-fill" style="font-size: 3rem; color: #6366f1;"></i>
            <h2 style="color: #f1f5f9; margin: 0.5rem 0;">Crime Analytics</h2>
            <p style="color: #94a3b8; font-size: 0.875rem;">Decision Support System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Overview", "State Analysis", "Crime Categories", 
             "Bias Analysis", "Risk Predictions", "Ethics"],
            label_visibility="collapsed",
            format_func=lambda x: {
                "Overview": "◈  Overview",
                "State Analysis": "◈  State Analysis", 
                "Crime Categories": "◈  Crime Categories",
                "Bias Analysis": "◈  Bias Analysis",
                "Risk Predictions": "◈  Risk Predictions",
                "Ethics": "◈  Ethics"
            }.get(x, x)
        )
        
        st.markdown("---")
        
        st.markdown("""
        <div style="padding: 1rem; background: rgba(99, 102, 241, 0.1); 
                    border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.3);">
            <p style="color: #a5b4fc; font-size: 0.75rem; margin: 0;">
                <i class="ri-database-2-line"></i> <strong>Data Source</strong><br>
                NCRB India 2013
            </p>
            <p style="color: #94a3b8; font-size: 0.75rem; margin: 0.5rem 0 0 0;">
                <i class="ri-graduation-cap-line"></i> <strong>Purpose</strong><br>
                Academic Research Only
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return page


def render_footer():
    """Render footer"""
    st.markdown("""
    <div class="custom-footer">
        <p>Built with <i class="ri-heart-fill" style="color: #ef4444;"></i> for Academic Research</p>
        <p style="font-size: 0.75rem; color: #64748b;">
            © 2026 Predictive Policing Decision Support System • 
            Data: NCRB India 2013 • 
            <span style="color: #ef4444;">Not for Law Enforcement Use</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application"""
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Render sidebar and get page
    page = render_sidebar()
    
    # Render header
    render_header()
    
    # Render ethical disclaimer
    render_ethical_disclaimer()
    
    # Render metrics
    render_metrics(df)
    
    # Route to page content
    if page == "Overview":
        render_state_analysis(df)
        render_crime_categories(df)
    elif page == "State Analysis":
        render_state_analysis(df)
    elif page == "Crime Categories":
        render_crime_categories(df)
    elif page == "Bias Analysis":
        render_bias_analysis(df)
    elif page == "Risk Predictions":
        render_risk_predictions(df)
    elif page == "Ethics":
        render_ethics_page()
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
