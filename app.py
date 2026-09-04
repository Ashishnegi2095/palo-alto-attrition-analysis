import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

#  configuration
st.set_page_config(
    page_title="Palo Alto Networks - Workforce Attrition Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# styling
st.markdown("""
    <style>
    .main-header { font-size: 26px; font-weight: 700; color: blue; margin-bottom: 2px; }
    .sub-header { font-size: 14px; color: #64748B; margin-bottom: 20px; }
    .card-metric { background-color: #F8FAFC; border-left: 4px solid #0284C7; padding: 16px; border-radius: 6px; }
    .risk-high { background-color: #FEF2F2; border-left: 4px solid #EF4444; padding: 14px; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("Palo Alto Networks.csv")
    
    # attrition
    if df['Attrition'].dtype == object:
        df['Attrition_Numeric'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        df['Attrition_Label'] = df['Attrition']
    else:
        df['Attrition_Numeric'] = df['Attrition']
        df['Attrition_Label'] = df['Attrition'].map({1: 'Exited', 0: 'Retained'})
        
    df['Tenure Bucket'] = pd.cut(
        df['YearsAtCompany'], 
        bins=[-1, 2, 5, 10, 40], 
        labels=['0-2 Years (Early)', '3-5 Years (Mid)', '6-10 Years (Experienced)', '10+ Years (Senior)']
    )
    df['Age Group'] = pd.cut(
        df['Age'], 
        bins=[17, 25, 35, 45, 65], 
        labels=['18-25 Years', '26-35 Years', '36-45 Years', '46+ Years']
    )
    return df

df_raw = load_and_preprocess_data()

# header
st.markdown("<div class='main-header'>Palo Alto Networks - Workforce Attrition & Risk Hotspot Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Foundational HR Intelligence & Diagnostic Analytics Platform</div>", unsafe_allow_html=True)

# sidebar filter control
st.sidebar.header("Interactive Filters")

departments = ["All"] + sorted(df_raw['Department'].unique().tolist())
selected_dept = st.sidebar.selectbox("Select Department", departments)

if selected_dept != "All":
    available_roles = ["All"] + sorted(df_raw[df_raw['Department'] == selected_dept]['JobRole'].unique().tolist())
else:
    available_roles = ["All"] + sorted(df_raw['JobRole'].unique().tolist())

selected_role = st.sidebar.selectbox("Select Job Role", available_roles)

tenure_min, tenure_max = int(df_raw['YearsAtCompany'].min()), int(df_raw['YearsAtCompany'].max())
selected_tenure = st.sidebar.slider("Tenure at Company (Years)", tenure_min, tenure_max, (tenure_min, tenure_max))

selected_overtime = st.sidebar.radio("Overtime Work Filter", ["All", "Yes", "No"], horizontal=True)

travel_options = ["All"] + sorted(df_raw['BusinessTravel'].unique().tolist())
selected_travel = st.sidebar.selectbox("Business Travel Frequency", travel_options)



filtered_df = df_raw.copy()

if selected_dept != "All":
    filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
if selected_role != "All":
    filtered_df = filtered_df[filtered_df['JobRole'] == selected_role]

filtered_df = filtered_df[
    (filtered_df['YearsAtCompany'] >= selected_tenure[0]) & 
    (filtered_df['YearsAtCompany'] <= selected_tenure[1])
]

if selected_overtime != "All":
    filtered_df = filtered_df[filtered_df['OverTime'] == selected_overtime]

if selected_travel != "All":
    filtered_df = filtered_df[filtered_df['BusinessTravel'] == selected_travel]

# Navi
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Attrition Overview", 
    "2. Department & Role Heatmaps", 
    "3. Demographic Explorer", 
    "4. Tenure & Workload Analysis"
])

# tab1
with tab1:
    st.subheader("Organizational Attrition Overview")
    
    total_emp = len(filtered_df)
    exited_emp = filtered_df['Attrition_Numeric'].sum()
    retained_emp = total_emp - exited_emp
    attrition_rate = (exited_emp / total_emp * 100) if total_emp > 0 else 0
    avg_monthly_income = filtered_df['MonthlyIncome'].mean() if total_emp > 0 else 0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Workforce", f"{total_emp:,}")
    m2.metric("Retained Employees", f"{retained_emp:,}")
    m3.metric("Exited Employees", f"{exited_emp:,}")
    m4.metric("Overall Attrition Rate", f"{attrition_rate:.2f}%")
    
    st.markdown("---")
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.markdown("##### Workforce Retention Proportions")
        pie_fig = px.pie(
            filtered_df, 
            names='Attrition_Label', 
            color='Attrition_Label',
            color_discrete_map={'Retained': '#0EA5E9', 'Exited': '#EF4444'},
            hole=0.45
        )
        pie_fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(pie_fig, use_container_width=True)
        
    with col_b:
        st.markdown("##### Attrition Distribution by Department")
        dept_chart_df = filtered_df.groupby(['Department', 'Attrition_Label']).size().reset_index(name='Count')
        bar_dept = px.bar(
            dept_chart_df, 
            x='Department', 
            y='Count', 
            color='Attrition_Label',
            barmode='group',
            color_discrete_map={'Retained': '#0EA5E9', 'Exited': '#EF4444'}
        )
        st.plotly_chart(bar_dept, use_container_width=True)

# tab2
with tab2:
    st.subheader("Department & Job Role Risk Hotspot Analysis")
    
    st.markdown("The heatmap below illustrates attrition intensity across functional departments and designations.")
    
    pivot_df = df_raw.pivot_table(
        index='Department', 
        columns='JobRole', 
        values='Attrition_Numeric', 
        aggfunc='mean'
    ) * 100
    
    heatmap_fig = px.imshow(
        pivot_df.fillna(0),
        labels=dict(x="Job Designation", y="Department", color="Attrition Rate (%)"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale="Reds",
        aspect="auto"
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # high risk call out 
    role_risk = df_raw.groupby('JobRole')['Attrition_Numeric'].agg(['count', 'sum', 'mean']).reset_index()
    role_risk['mean'] = role_risk['mean'] * 100
    role_risk.columns = ['Job Role', 'Headcount', 'Exits', 'Attrition Rate (%)']
    role_risk = role_risk.sort_values(by='Attrition Rate (%)', ascending=False)
    
    col_r1, col_r2 = st.columns([2, 1])
    with col_r1:
        st.markdown("##### Job Roles Ranked by Exit Frequency")
        st.dataframe(role_risk.style.format({'Attrition Rate (%)': '{:.2f}%'}), hide_index=True)
    with col_r2:
        st.markdown("<div class='risk-high'>", unsafe_allow_html=True)
        st.markdown("<b> High-Risk Hotspot Warning</b>")
        st.write("• **Sales Representatives** display the highest exit rate at **39.76%**.")
        st.write("• **Laboratory Technicians** display a **23.94%** exit rate.")
        st.write("• **Human Resources Specialists** display a **23.08%** exit rate.")
        st.markdown("</div>", unsafe_allow_html=True)

# tb3
with tab3:
    st.subheader("Demographic Attrition Explorer")
    
    d1, d2 = st.columns(2)
    
    with d1:
        st.markdown("##### Attrition Rate by Age Bracket")
        age_attr = filtered_df.groupby('Age Group', observed=False)['Attrition_Numeric'].mean().reset_index()
        age_attr['Attrition Rate (%)'] = age_attr['Attrition_Numeric'] * 100
        fig_age = px.bar(
            age_attr, 
            x='Age Group', 
            y='Attrition Rate (%)', 
            color='Age Group',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
    with d2:
        st.markdown("##### Attrition by Marital Status & Gender")
        mg_df = filtered_df.groupby(['MaritalStatus', 'Gender'])['Attrition_Numeric'].mean().reset_index()
        mg_df['Attrition Rate (%)'] = mg_df['Attrition_Numeric'] * 100
        fig_mg = px.bar(
            mg_df, 
            x='MaritalStatus', 
            y='Attrition Rate (%)', 
            color='Gender', 
            barmode='group',
            color_discrete_map={'Male': '#1E40AF', 'Female': '#DB2777'}
        )
        st.plotly_chart(fig_mg, use_container_width=True)

# teneure and  workload

with tab4:
    st.subheader("Tenure & Workload Drivers Analysis")
    
    w1, w2 = st.columns(2)
    
    with w1:
        st.markdown("##### Impact of Overtime on Attrition")
        ot_df = filtered_df.groupby('OverTime')['Attrition_Numeric'].mean().reset_index()
        ot_df['Attrition Rate (%)'] = ot_df['Attrition_Numeric'] * 100
        fig_ot = px.bar(
            ot_df, 
            x='OverTime', 
            y='Attrition Rate (%)', 
            color='OverTime',
            color_discrete_map={'Yes': '#EF4444', 'No': '#10B981'}
        )
        st.plotly_chart(fig_ot, use_container_width=True)
        
    with w2:
        st.markdown("##### Business Travel Frequency vs Attrition")
        bt_df = filtered_df.groupby('BusinessTravel')['Attrition_Numeric'].mean().reset_index()
        bt_df['Attrition Rate (%)'] = bt_df['Attrition_Numeric'] * 100
        fig_bt = px.bar(
            bt_df, 
            x='BusinessTravel', 
            y='Attrition Rate (%)', 
            color='BusinessTravel',
            color_discrete_sequence=px.colors.sequential.Plotly3
        )
        st.plotly_chart(fig_bt, use_container_width=True)
        
    st.markdown("---")
    st.markdown("##### Attrition Rate by Company Tenure Bands")
    ten_df = filtered_df.groupby('Tenure Bucket', observed=False)['Attrition_Numeric'].mean().reset_index()
    ten_df['Attrition Rate (%)'] = ten_df['Attrition_Numeric'] * 100
    fig_ten = px.line(
        ten_df, 
        x='Tenure Bucket', 
        y='Attrition Rate (%)', 
        markers=True,
        line_shape='linear'
    )
    st.plotly_chart(fig_ten, use_container_width=True)