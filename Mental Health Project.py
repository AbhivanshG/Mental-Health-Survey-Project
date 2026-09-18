import streamlit as st
import pandas as pd
import numpy as np

# Import plotly separately to ensure it's loaded
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    st.error("Plotly not installed. Please check requirements.txt")
    st.stop()

import seaborn as sns
from scipy import stats

# Set page configuration
st.set_page_config(
    page_title="Mental Health Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 30px;
    }
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Mental Health in Tech - Data Analysis Dashboard")
st.markdown("---")

# Load the CSV file
try:
    df = pd.read_csv('survey.csv')
    
    # ===== DATA CLEANING =====
    # Age Column fixed
    df = df[(df['Age'] > 0) & (df['Age'] <= 120)]
    
    # Fixing Gender column
    df['Gender'] = df['Gender'].str.lower().str.strip()
    
    df['Gender'] = df['Gender'].map({
        'male': 'Male',
        'm': 'Male',
        'man': 'Male',
        'female': 'Female',
        'f': 'Female',
        'woman': 'Female',
    })
    
    df['Gender'] = df['Gender'].fillna('Other')
    
    # Drop columns
    if 'state' in df.columns:
        df = df.drop('state', axis=1)
    if 'comments' in df.columns:
        df = df.drop('comments', axis=1)
    
    # Fill missing values
    df['work_interfere'] = df['work_interfere'].fillna('Unknown')
    df['self_employed'] = df['self_employed'].fillna('Unknown')
    
    # ===== DISPLAY BASIC METRICS =====
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Respondents", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Age Range", f"{df['Age'].min():.0f} - {df['Age'].max():.0f}")
    
    st.markdown("---")
    
    # Create tabs
    tab_home, tab_overview, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Home",
        "📊 Data Overview",
        "👤 Demographics", 
        "💼 Work & Employment", 
        "🏥 Health & Treatment", 
        "📊 Cross-Analysis",
        "📈 Summary"
    ])
    
    # ===== TAB HOME =====
    with tab_home:
        st.header("🏠 Welcome to Mental Health in Tech Dashboard")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 📋 Project Overview
            
            This dashboard presents a comprehensive analysis of **mental health in the technology industry**. 
            The data was collected through an extensive survey targeting tech professionals from around the world.
            
            - 🧠 How mental health issues affect tech workers
            - 💼 The impact of work environment on mental wellness
            - 👥 Demographic patterns in mental health
            - 🏥 Treatment-seeking behaviors and family history
            - 🌍 Global perspectives on mental health in tech
            """)
        
        with col2:
            st.info(f"""
            ### 📊 Dataset Stats
            **Total Respondents:** {len(df):,}
            
            **Survey Features:** {len(df.columns)}
            
            **Data Coverage:** Global
            
            **Analysis Type:** EDA
            """)
    
    # ===== TAB DATA OVERVIEW =====
    with tab_overview:
        st.header("📊 Data Overview")
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(15), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("Data Types & Info")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count()
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📐 Dataset Dimensions")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
    
    # ===== TAB 1: DEMOGRAPHICS =====
    with tab1:
        st.header("👤 Demographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            fig_age = go.Figure(data=[
                go.Histogram(
                    x=df['Age'],
                    nbinsx=30,
                    marker_color='skyblue',
                    marker_line_color='black',
                    marker_line_width=1
                )
            ])
            fig_age.update_layout(
                title_text="Age Distribution",
                xaxis_title="Age",
                yaxis_title="Count",
                template='plotly_white',
                height=500
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.subheader("Gender Distribution")
            gender_counts = df['Gender'].value_counts()
            fig_gender = go.Figure(data=[
                go.Bar(
                    x=gender_counts.index,
                    y=gender_counts.values,
                    marker_color=['coral', 'skyblue', 'lightcoral'],
                    text=gender_counts.values,
                    textposition='auto',
                )
            ])
            fig_gender.update_layout(
                title_text="Gender Distribution",
                xaxis_title="Gender",
                yaxis_title="Count",
                template='plotly_white',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        
        st.subheader("📊 Age Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Age", f"{df['Age'].mean():.1f}")
        with col2:
            st.metric("Median Age", f"{df['Age'].median():.1f}")
        with col3:
            st.metric("Std Dev", f"{df['Age'].std():.1f}")
        with col4:
            st.metric("Most Common Age", f"{df['Age'].mode()[0]:.0f}")
    
    # ===== TAB 2: WORK & EMPLOYMENT =====
    with tab2:
        st.header("💼 Work & Employment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Mental Health Interference at Work")
            work_counts = df['work_interfere'].value_counts()
            fig_work = go.Figure(data=[
                go.Bar(
                    x=work_counts.index,
                    y=work_counts.values,
                    marker_color='lightgreen',
                    text=work_counts.values,
                    textposition='auto',
                )
            ])
            fig_work.update_layout(
                title_text="Mental Health Interference at Work",
                xaxis_title="Interference Level",
                yaxis_title="Count",
                template='plotly_white',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_work, use_container_width=True)
        
        with col2:
            st.subheader("Self Employed Status")
            self_emp_counts = df['self_employed'].value_counts()
            fig_self = go.Figure(data=[
                go.Bar(
                    x=self_emp_counts.index,
                    y=self_emp_counts.values,
                    marker_color='skyblue',
                    text=self_emp_counts.values,
                    textposition='auto',
                )
            ])
            fig_self.update_layout(
                title_text="Self Employed Distribution",
                xaxis_title="Self Employed",
                yaxis_title="Count",
                template='plotly_white',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_self, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Remote Work Opportunity")
            remote_counts = df['remote_work'].value_counts()
            fig_remote = go.Figure(data=[
                go.Bar(
                    x=remote_counts.index,
                    y=remote_counts.values,
                    marker_color='cyan',
                    text=remote_counts.values,
                    textposition='auto',
                )
            ])
            fig_remote.update_layout(
                title_text="Remote Work Availability",
                xaxis_title="Remote Work Available",
                yaxis_title="Count",
                template='plotly_white',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_remote, use_container_width=True)
        
        with col2:
            st.subheader("Company Size Distribution")
            comp_size = df['no_employees'].value_counts()
            fig_comp = go.Figure(data=[
                go.Bar(
                    x=comp_size.index.astype(str),
                    y=comp_size.values,
                    marker_color='teal',
                    text=comp_size.values,
                    textposition='auto',
                )
            ])
            fig_comp.update_layout(
                title_text="Company Size Distribution",
                xaxis_title="Number of Employees",
                yaxis_title="Count",
                template='plotly_white',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_comp, use_container_width=True)
    
    # ===== TAB 3: HEALTH & TREATMENT =====
    with tab3:
        st.header("🏥 Health & Treatment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Family History of Mental Health Issues")
            family_counts = df['family_history'].value_counts()
            fig_family = go.Figure(data=[
                go.Bar(
                    x=family_counts.index,
                    y=family_counts.values,
                    marker_color=['#9370DB', '#BA55D3'],
                    text=family_counts.values,
                    textposition='auto',
                )
            ])
            fig_family.update_layout(
                title_text="Family History Distribution",
                xaxis_title="Family History",
                yaxis_title="Count",
                template='plotly_white',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_family, use_container_width=True)
        
        with col2:
            st.subheader("Seeking Treatment")
            treatment_counts = df['treatment'].value_counts()
            fig_treatment = go.Figure(data=[
                go.Bar(
                    x=treatment_counts.index,
                    y=treatment_counts.values,
                    marker_color=['#ff9999', '#66b3ff'],
                    text=treatment_counts.values,
                    textposition='auto',
                )
            ])
            fig_treatment.update_layout(
                title_text="Seeking Treatment for Mental Health",
                xaxis_title="Treatment",
                yaxis_title="Count",
                template='plotly_white',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_treatment, use_container_width=True)
        
        st.subheader("Work in Tech Company")
        tech_counts = df['tech_company'].value_counts()
        fig_tech = go.Figure(data=[
            go.Bar(
                x=tech_counts.index,
                y=tech_counts.values,
                marker_color=['#00FF00', '#FF6B6B'],
                text=tech_counts.values,
                textposition='auto',
            )
        ])
        fig_tech.update_layout(
            title_text="Work in Tech Company",
            xaxis_title="Tech Company",
            yaxis_title="Count",
            template='plotly_white',
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig_tech, use_container_width=True)
    
    # ===== TAB 4: CROSS-ANALYSIS WITH BOX PLOT =====
    with tab4:
        st.header("📊 Cross-Variable Analysis: Age & Work Interference")
        
        st.markdown("""
        ## 🎬 The Story Behind the Numbers
        
        **How does age relate to mental health interference at work?**
        """)
        
        # Box plot
        st.subheader("🔹 Age vs Mental Health Work Interference")
        
        fig_box = go.Figure()
        
        interference_order = ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown']
        colors_box = ['#2ecc71', '#e74c3c', '#3498db', '#e91e63', '#95a5a6']
        
        for i, interference in enumerate(interference_order):
            age_data = df[df['work_interfere'] == interference]['Age']
            fig_box.add_trace(go.Box(
                y=age_data,
                name=interference,
                boxmean='sd',
                marker_color=colors_box[i],
                boxpoints=False
            ))
        
        fig_box.update_layout(
            title_text="Age vs Mental Health Work Interference",
            yaxis_title="Age (years)",
            xaxis_title="Work Interference Level",
            template='plotly_white',
            height=600,
            hovermode='y unified'
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("---")
        
        # Statistics table
        st.subheader("📊 Statistical Breakdown")
        
        interference_stats = []
        for interference in interference_order:
            group_data = df[df['work_interfere'] == interference]['Age']
            interference_stats.append({
                'Interference': interference,
                'Median': f"{group_data.median():.0f}",
                'Q1 (25%)': f"{group_data.quantile(0.25):.0f}",
                'Q3 (75%)': f"{group_data.quantile(0.75):.0f}",
                'Min': f"{group_data.min():.0f}",
                'Max': f"{group_data.max():.0f}",
                'Count': len(group_data)
            })
        
        stats_df = pd.DataFrame(interference_stats)
        st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("---")
        
        # Key insights
        st.subheader("💡 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("🎯 **Age Doesn't Strongly Predict Work Interference** - All groups have similar median ages (~30-32)")
        
        with col2:
            st.success("🌟 **Hope at Any Age** - People from 18-71 experience minimal work interference")
        
        with col3:
            st.warning("🚨 **Young People Struggle Too** - Similar age distribution in 'Often' and 'Never' groups")
        
        # Cross-analysis charts
        st.markdown("---")
        st.subheader("Self Employed vs Work Interference")
        
        crosstab1 = pd.crosstab(df['self_employed'], df['work_interfere'])
        fig_cross1 = go.Figure()
        for col in crosstab1.columns:
            fig_cross1.add_trace(go.Bar(
                name=col,
                x=crosstab1.index,
                y=crosstab1[col],
                text=crosstab1[col],
                textposition='auto',
            ))
        
        fig_cross1.update_layout(
            title_text="Self Employed vs Work Interference",
            xaxis_title="Self Employed",
            yaxis_title="Count",
            barmode='group',
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig_cross1, use_container_width=True)
        
        st.subheader("Family History vs Seeking Treatment")
        
        crosstab2 = pd.crosstab(df['family_history'], df['treatment'])
        fig_cross2 = go.Figure()
        for col in crosstab2.columns:
            fig_cross2.add_trace(go.Bar(
                name=col,
                x=crosstab2.index,
                y=crosstab2[col],
                text=crosstab2[col],
                textposition='auto',
            ))
        
        fig_cross2.update_layout(
            title_text="Family History vs Seeking Treatment",
            xaxis_title="Family History",
            yaxis_title="Count",
            barmode='group',
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig_cross2, use_container_width=True)
    
    # ===== TAB 5: SUMMARY =====
    with tab5:
        st.header("📈 Summary & Insights")
        
        st.markdown("""
        ## 🎯 Project Summary
        
        This analysis examined mental health patterns in the technology industry through global survey data.
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Respondents", f"{len(df):,}")
        with col2:
            st.metric("Average Age", f"{df['Age'].mean():.1f} years")
        with col3:
            st.metric("Gender Categories", df['Gender'].nunique())
        with col4:
            st.metric("Survey Features", len(df.columns))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Demographics Insights:")
            st.write(f"- **Age Range**: {int(df['Age'].min())} to {int(df['Age'].max())} years")
            st.write(f"- **Average Age**: {df['Age'].mean():.1f} years")
            st.write(f"- **Median Age**: {df['Age'].median():.1f} years")
            
            gender_dist = df['Gender'].value_counts()
            for gender, count in gender_dist.items():
                pct = (count / len(df)) * 100
                st.write(f"- **{gender}**: {count:,} ({pct:.1f}%)")
        
        with col2:
            st.markdown("### Mental Health Insights:")
            
            treatment_yes = (df['treatment'] == 'Yes').sum()
            treatment_pct = (treatment_yes / len(df)) * 100
            st.write(f"- **Seeking Treatment**: {treatment_yes:,} ({treatment_pct:.1f}%)")
            
            family_yes = (df['family_history'] == 'Yes').sum()
            family_pct = (family_yes / len(df)) * 100
            st.write(f"- **Family History**: {family_yes:,} ({family_pct:.1f}%)")
            
            work_interference = df['work_interfere'].value_counts()
            st.write("**Work Interference Levels:**")
            for interference, count in work_interference.items():
                pct = (count / len(df)) * 100
                st.write(f"- **{interference}**: {count:,} ({pct:.1f}%)")
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>📊 Mental Health in Tech Dashboard | Built with Streamlit & Plotly</p>", unsafe_allow_html=True)

except FileNotFoundError:
    st.error("❌ Error: 'survey.csv' not found!")
    st.info("Please ensure 'survey.csv' is in the same directory as this script.")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.info("Please check your CSV file and try again.")