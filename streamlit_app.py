%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

st.set_page_config(
    page_title="Smart Traffic Safety",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0e1117; color: white; }
[data-testid="stSidebar"] { background-color: #1a1d27; }
.metric-box { background:#1e2130; border-radius:8px; padding:20px; margin:5px; }
.metric-label { color:#aaa; font-size:13px; }
.metric-value { color:white; font-size:28px; font-weight:bold; }
.module-done { background:#1a3a2a; color:#2ecc71; padding:12px 18px;
               border-radius:6px; margin:6px 0; font-size:15px; }
.footer { color:#555; font-size:12px; text-align:center;
          border-top:1px solid #333; padding-top:12px; margin-top:30px; }
div[data-testid="stRadio"] > div { flex-direction: column; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    rf         = pickle.load(open('saved_models/rf_model.pkl','rb'))
    xgb        = pickle.load(open('saved_models/xgb_model.pkl','rb'))
    lr         = pickle.load(open('saved_models/lr_model.pkl','rb'))
    trf_rf     = pickle.load(open('saved_models/trf_rf_model.pkl','rb'))
    scaler     = pickle.load(open('saved_models/scaler.pkl','rb'))
    scaler_trf = pickle.load(open('saved_models/scaler_trf.pkl','rb'))
    le_sev     = pickle.load(open('saved_models/le_severity.pkl','rb'))
    le_w       = pickle.load(open('saved_models/le_weather.pkl','rb'))
    le_trf     = pickle.load(open('saved_models/le_traffic.pkl','rb'))
    le_wtrf    = pickle.load(open('saved_models/le_weather_trf.pkl','rb'))
    centroids  = pd.read_csv('saved_models/centroids.csv')
    weather_time_analysis = pickle.load(open('saved_models/weather_time_analysis.pkl', 'rb'))
    # Load pre-computed trend data saved from notebook
    accident_trends = pickle.load(open('saved_models/accident_trends.pkl', 'rb'))
    return (rf, xgb, lr, trf_rf, scaler, scaler_trf,
            le_sev, le_w, le_trf, le_wtrf, centroids,
            weather_time_analysis, accident_trends)

(rf, xgb, lr, trf_rf, scaler, scaler_trf,
 le_sev, le_w, le_trf, le_wtrf, centroids,
 weather_time_analysis, accident_trends) = load_models()

# ── Risk Engine ───────────────────────────────────────────
def calculate_risk(traffic_level, accident_density, weather_condition):
    t_map = {'Low':0.2, 'Medium':0.6, 'High':1.0}
    a_map = {'Low':0.1, 'Moderate':0.5, 'High':1.0}
    w_map = {'Clear':0.1, 'Clouds':0.2, 'Mist':0.3, 'Haze':0.3,
             'Drizzle':0.4, 'Rain':0.6, 'Fog':0.7,
             'Snow':0.8, 'Thunderstorm':0.9, 'Squall':0.85}
    t = t_map.get(traffic_level, 0.5)
    a = a_map.get(accident_density, 0.3)
    w = w_map.get(weather_condition, 0.3)
    score = round(0.4*t + 0.4*a + 0.2*w, 4)
    if score < 0.25:   cat = 'Low'
    elif score < 0.50: cat = 'Moderate'
    elif score < 0.75: cat = 'High'
    else:              cat = 'Severe'
    return score, cat

# ── Helper: dark-themed axes ─────────────────────────────
def dark_ax(ax):
    ax.set_facecolor('#1e2130')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚦 Smart Traffic Safety")
    st.markdown("Najmul Hoda | 02416412822")
    st.markdown("B.Tech ECE 8th Sem")
    st.markdown("Guide: Dr. Shiv Ram Meena")
    st.markdown("GGSIPU Dwarka")
    st.markdown("---")
    st.markdown("### 📍 Navigate")
    page = st.radio("", [
        "Overview",
        "Accident Analysis",
        "Traffic Prediction",
        "Risk Engine",
        "Route Recommendation",
        "Live Maps"
    ])

FOOTER = """<div class='footer'>Smart Traffic Safety System |
Najmul Hoda (02416412822) | B.Tech ECE 8th Sem | GGSIPU |
Guide: Dr. Shiv Ram Meena</div>"""

# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Smart Traffic Safety System")
    st.markdown("**Major Project | B.Tech ECE | GGSIPU**")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Accidents Analyzed", "Traffic Records", "Hotspots Detected", "Best Accuracy"],
        ["97,833", "48,204", "6", "91.0%"]
    ):
        col.markdown(f"""<div class='metric-box'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Modules Completed")
        for m in [
            "✅ Data Collection and Preprocessing",
            "✅ Exploratory Data Analysis",
            "✅ Accident Hotspot Detection (KMeans K=6)",
            "✅ Severity Prediction (RF + XGBoost + LR)",
            "✅ Traffic Level Prediction (RF - 91%)",
            "✅ Traffic Safety Risk Engine",
            "✅ Route Recommendation System",
            "✅ Interactive Dashboard"
        ]:
            st.markdown(f"<div class='module-done'>{m}</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("Tech Stack")
        tech = pd.DataFrame({
            "Component": ["Language", "ML Models", "Clustering",
                          "Geospatial", "Dashboard", "Platform"],
            "Technology": ["Python 3.10", "Random Forest, XGBoost, LR",
                           "KMeans (K=6)", "Folium", "Streamlit", "Google Colab"]
        })
        st.dataframe(tech, use_container_width=True, hide_index=True)

    st.markdown(FOOTER, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — ACCIDENT ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "Accident Analysis":
    st.title("Accident Data Analysis")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    for col, label, val in zip(
        [c1, c2, c3],
        ["Total Records", "High Hotspots", "High Severity"],
        ["97,833", "2,000", "1,935"]
    ):
        col.markdown(f"""<div class='metric-box'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: Severity pie + Hourly line ──────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Severity Distribution")
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        sizes  = [45.4, 30.5, 24.1]
        labels = ['Medium', 'Low', 'High']
        colors = ['#f39c12', '#2ecc71', '#e74c3c']
        ax.pie(sizes, labels=labels, colors=colors,
               autopct='%1.1f%%', startangle=90,
               textprops={'color': 'white', 'fontsize': 13})
        st.pyplot(fig)

    with col2:
        st.subheader("Accidents by Hour")
        hours = list(range(24))
        vals  = [120, 80, 60, 50, 70, 150, 280, 340, 300, 260,
                 240, 250, 270, 260, 250, 280, 340, 370, 320,
                 280, 240, 200, 170, 140]
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        dark_ax(ax)
        ax.plot(hours, vals, color='#3498db', linewidth=2, marker='o', markersize=4)
        ax.fill_between(hours, vals, alpha=0.2, color='#3498db')
        ax.axvspan(7,  9,  alpha=0.2, color='red', label='Rush Hours')
        ax.axvspan(17, 19, alpha=0.2, color='red')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Accidents')
        ax.legend(facecolor='#1e2130', labelcolor='white')
        st.pyplot(fig)

    st.markdown("---")

    # ── Row 2: EDA plots (Day of Week + Top States) ────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accidents by Day of Week")
        days     = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_vals = [15200, 15800, 15500, 15300, 16100, 10200, 9700]
        fig, ax  = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        dark_ax(ax)
        ax.bar(days, day_vals, color='#9b59b6')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Number of Accidents')
        for i, v in enumerate(day_vals):
            ax.text(i, v + 100, f'{v:,}', ha='center',
                    fontsize=8, color='white')
        st.pyplot(fig)

    with col2:
        st.subheader("Top 10 States by Accidents")
        states     = ['CA', 'FL', 'TX', 'OR', 'VA', 'NY', 'PA', 'SC', 'NC', 'MN']
        state_vals = [18200, 12400, 9800, 6700, 6200, 5800, 5500, 5100, 4900, 4600]
        fig, ax    = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        dark_ax(ax)
        ax.bar(states, state_vals, color='#e67e22')
        ax.set_xlabel('State')
        ax.set_ylabel('Number of Accidents')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

    st.markdown("---")

    # ── Row 3: Weather-Time Heatmap ────────────────────────
    st.subheader("Accident Frequency by Weather Condition and Hour")
    fig_hm, ax_hm = plt.subplots(figsize=(15, 8), facecolor='#0e1117')
    ax_hm.set_facecolor('#0e1117')
    sns.heatmap(
        weather_time_analysis.head(10),
        cmap='viridis', annot=True, fmt='d',
        linewidths=.5, ax=ax_hm,
        cbar_kws={'label': 'Number of Accidents'}
    )
    ax_hm.set_title('Top 10 Weather Conditions vs. Hour of Accident', color='white')
    ax_hm.set_xlabel('Hour of Day', color='white')
    ax_hm.set_ylabel('Weather Condition', color='white')
    ax_hm.tick_params(axis='x', colors='white')
    ax_hm.tick_params(axis='y', colors='white', rotation=0)
    st.pyplot(fig_hm)

    st.markdown("---")

    # ── Row 4: Accident Trends (Week / Month / Year) ───────
    st.subheader("Accident Trends by Week, Month, and Year")

    weekly  = accident_trends['weekly']   # Series indexed by week number
    monthly = accident_trends['monthly']  # Series indexed by month (1-12)
    yearly  = accident_trends['yearly']   # Series indexed by year

    fig_tr, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#1e2130')
    fig_tr.suptitle('Accident Trends by Week, Month, and Year',
                    fontsize=15, fontweight='bold', color='white')

    for ax in axes:
        dark_ax(ax)

    # Accidents by Week
    axes[0].bar(weekly.index.astype(int), weekly.values, color='#9b59b6')
    axes[0].set_title('Accidents by Week of Year')
    axes[0].set_xlabel('Week Number')
    axes[0].set_ylabel('Number of Accidents')
    axes[0].tick_params(axis='x', rotation=45)

    # Accidents by Month
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    axes[1].bar(monthly.index, monthly.values, color='#2ecc71')
    axes[1].set_title('Accidents by Month')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Number of Accidents')
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(month_names, rotation=45, color='white')

    # Accidents by Year
    axes[2].bar(yearly.index, yearly.values, color='#3498db')
    axes[2].set_title('Accidents by Year')
    axes[2].set_xlabel('Year')
    axes[2].set_ylabel('Number of Accidents')
    axes[2].set_xticks(yearly.index)
    axes[2].set_xticklabels([str(y) for y in yearly.index],
                             rotation=45, color='white')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_tr)

    st.markdown(FOOTER, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — TRAFFIC PREDICTION
# ══════════════════════════════════════════════════════════
elif page == "Traffic Prediction":
    st.title("Traffic Prediction Module")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    for col, label, val in zip(
        [c1, c2, c3],
        ["Dataset", "Records", "Accuracy"],
        ["Metro Interstate", "48,204", "91.0%"]
    ):
        col.markdown(f"""<div class='metric-box'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Traffic EDA Section ────────────────────────────────
    st.subheader("📊 Traffic EDA")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Traffic Level Distribution**")
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        ax.pie([61.6, 38.4],
               labels=['Low', 'Medium'],
               colors=['#2ecc71', '#f39c12'],
               autopct='%1.1f%%', startangle=90,
               textprops={'color': 'white', 'fontsize': 14})
        ax.set_title('Traffic Level Distribution', color='white')
        st.pyplot(fig)

    with col2:
        st.markdown("**Avg Traffic Volume by Hour**")
        hours = list(range(24))
        vals  = [500, 300, 200, 150, 200, 600, 2800, 3050,
                 2400, 2000, 1800, 1900, 2000, 1900, 1800,
                 2100, 2800, 3050, 2600, 2200, 1800, 1400,
                 1000, 700]
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        dark_ax(ax)
        ax.plot(hours, vals, color='#e67e22', linewidth=2.5,
                marker='o', markersize=5)
        ax.fill_between(hours, vals, alpha=0.2, color='#e67e22')
        ax.axvspan(7,  9,  alpha=0.2, color='red', label='Rush Hours')
        ax.axvspan(17, 19, alpha=0.2, color='red')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Avg Traffic Volume')
        ax.legend(facecolor='#1e2130', labelcolor='white')
        st.pyplot(fig)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Avg Traffic Volume by Day of Week**")
        days     = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_vals = [2800, 2900, 2850, 2870, 3050, 1800, 1500]
        fig, ax  = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        dark_ax(ax)
        ax.bar(days, day_vals, color='#9b59b6')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Avg Traffic Volume')
        for i, v in enumerate(day_vals):
            ax.text(i, v + 30, f'{v:,}', ha='center',
                    fontsize=8, color='white')
        st.pyplot(fig)

    with col2:
        st.markdown("**Avg Traffic Volume by Month**")
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        month_vals  = [2200, 2100, 2400, 2600, 2700, 2500,
                       2300, 2400, 2600, 2700, 2500, 2100]
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        dark_ax(ax)
        ax.bar(range(1, 13), month_vals, color='#3498db')
        ax.set_xlabel('Month')
        ax.set_ylabel('Avg Traffic Volume')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names, rotation=45, color='white')
        st.pyplot(fig)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Traffic Volume by Weather Condition**")
        weather_labels = ['Clear', 'Clouds', 'Rain', 'Mist',
                          'Snow', 'Drizzle', 'Fog', 'Thunderstorm']
        weather_vals   = [3100, 2800, 2200, 2400, 1800, 2000, 1600, 1400]
        colors_w = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(weather_labels)))
        fig, ax  = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        dark_ax(ax)
        bars = ax.barh(weather_labels, weather_vals, color=colors_w)
        ax.set_xlabel('Avg Traffic Volume')
        ax.set_title('Weather vs Traffic Volume', color='white')
        st.pyplot(fig)

    with col2:
        st.markdown("**Rush Hour vs Non-Rush Hour Traffic**")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
        ax.set_facecolor('#1e2130')
        ax.pie([35, 65],
               labels=['Rush Hour', 'Non-Rush Hour'],
               colors=['#e74c3c', '#3498db'],
               autopct='%1.1f%%', startangle=90,
               textprops={'color': 'white', 'fontsize': 13},
               explode=[0.05, 0])
        ax.set_title('Rush Hour Distribution', color='white')
        st.pyplot(fig)

    st.markdown("---")

    # ── Predict Section ────────────────────────────────────
    st.subheader("🚗 Predict Traffic Level")
    col1, col2 = st.columns(2)
    with col1:
        t_weather = st.selectbox("Weather", le_wtrf.classes_)
        t_temp    = st.slider("Temperature (K)", 250.0, 310.0, 288.0)
        t_rain    = st.slider("Rain 1h (mm)", 0.0, 100.0, 0.0)
        t_snow    = st.slider("Snow 1h (mm)", 0.0, 100.0, 0.0)
        t_clouds  = st.slider("Cloud Cover (%)", 0, 100, 40)
    with col2:
        t_hour    = st.slider("Hour", 0, 23, 8)
        t_day     = st.slider("Day of Week", 0, 6, 0)
        t_month   = st.slider("Month", 1, 12, 6)
        t_weekend = st.checkbox("Is Weekend?")
        t_holiday = st.checkbox("Is Holiday?")
        t_rush    = st.checkbox("Is Rush Hour?")

    if st.button("🚗 Predict Traffic"):
        tw_enc = le_wtrf.transform([t_weather])[0]
        wr = {'Clear':0.1, 'Clouds':0.2, 'Mist':0.3, 'Haze':0.3,
              'Drizzle':0.4, 'Rain':0.6, 'Fog':0.7,
              'Snow':0.8, 'Thunderstorm':0.9, 'Squall':0.85
              }.get(t_weather, 0.3)
        feat   = np.array([[t_hour, t_day, t_month,
                            int(t_weekend), int(t_holiday),
                            int(t_rush), tw_enc, t_temp,
                            t_rain, t_snow, t_clouds, wr]])
        feat_s = scaler_trf.transform(feat)
        pred   = trf_rf.predict(feat_s)[0]
        label  = le_trf.inverse_transform([pred])[0]
        color  = {'Low':'green', 'Medium':'orange',
                  'High':'red'}.get(label, 'gray')
        st.markdown(f"## Traffic Level: :{color}[**{label}**]")

    st.markdown(FOOTER, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 4 — RISK ENGINE
# ══════════════════════════════════════════════════════════
elif page == "Risk Engine":
    st.title("Traffic Safety Risk Engine")
    st.markdown("---")
    st.latex(r"Risk = 0.4 \times Traffic + 0.4 \times Accident + 0.2 \times Weather")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Calculate Risk Score")
        r_traffic  = st.selectbox("Traffic Level", ['Low', 'Medium', 'High'])
        r_accident = st.selectbox("Accident Density", ['Low', 'Moderate', 'High'])
        r_weather  = st.selectbox("Weather",
                                  ['Clear', 'Clouds', 'Mist', 'Haze',
                                   'Drizzle', 'Rain', 'Fog', 'Snow',
                                   'Thunderstorm', 'Squall'])
        if st.button("Calculate Risk", type="primary"):
            score, cat = calculate_risk(r_traffic, r_accident, r_weather)
            color = {'Low':'green', 'Moderate':'orange',
                     'High':'red', 'Severe':'violet'}.get(cat, 'gray')
            st.markdown(f"### Risk Score: `{score}`")
            st.markdown(f"### Risk Category: :{color}[**{cat}**]")

    with col2:
        st.subheader("Full Risk Matrix")
        combos = []
        for t in ['Low', 'Medium', 'High']:
            for a in ['Low', 'Moderate', 'High']:
                for w in ['Clear', 'Rain', 'Snow', 'Thunderstorm']:
                    s, c = calculate_risk(t, a, w)
                    combos.append({'Traffic':t, 'Accident':a,
                                   'Weather':w, 'Score':s, 'Level':c})
        risk_df = pd.DataFrame(combos).sort_values('Score', ascending=False).head(10)
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

    st.markdown(FOOTER, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 5 — ROUTE RECOMMENDATION
# ══════════════════════════════════════════════════════════
elif page == "Route Recommendation":
    st.title("Route Recommendation System")
    st.markdown("---")
    st.info("Configure each route - system finds the safest path.")

    ROUTES_DEFAULT = [
        {'id':'Route A', 'desc':'Highway via Downtown',
         'distance_km':12.5, 'time_min':22,
         'traffic':'High', 'accident':'High', 'weather':'Rain'},
        {'id':'Route B', 'desc':'Expressway (longer)',
         'distance_km':17.2, 'time_min':28,
         'traffic':'Medium', 'accident':'Low', 'weather':'Clear'},
        {'id':'Route C', 'desc':'Residential Streets',
         'distance_km':10.1, 'time_min':35,
         'traffic':'Low', 'accident':'Moderate', 'weather':'Fog'},
        {'id':'Route D', 'desc':'Ring Road (safest)',
         'distance_km':20.3, 'time_min':32,
         'traffic':'Low', 'accident':'Low', 'weather':'Clear'},
    ]

    cols = st.columns(4)
    route_inputs = []
    for i, (col, r) in enumerate(zip(cols, ROUTES_DEFAULT)):
        with col:
            st.markdown(f"**{r['id']}**")
            t = st.selectbox("Traffic",
                ['Low','Medium','High'],
                index=['Low','Medium','High'].index(r['traffic']),
                key=f"rt{i}")
            a = st.selectbox("Accident",
                ['Low','Moderate','High'],
                index=['Low','Moderate','High'].index(r['accident']),
                key=f"ra{i}")
            w = st.selectbox("Weather",
                ['Clear','Clouds','Rain','Fog','Snow','Thunderstorm'],
                index=['Clear','Clouds','Rain','Fog','Snow','Thunderstorm'
                       ].index(r['weather'])
                      if r['weather'] in
                      ['Clear','Clouds','Rain','Fog','Snow','Thunderstorm']
                      else 0,
                key=f"rw{i}")
            d  = st.number_input("Distance(km)", value=r['distance_km'], key=f"rd{i}")
            tm = st.number_input("Time(min)", value=float(r['time_min']), key=f"rt2{i}")
            route_inputs.append({'id':r['id'], 'desc':r['desc'],
                                 'distance_km':d, 'time_min':tm,
                                 'traffic':t, 'accident':a, 'weather':w})

    if st.button("Find Safest Route", type="primary"):
        results = []
        for r in route_inputs:
            score, cat = calculate_risk(r['traffic'], r['accident'], r['weather'])
            results.append({'Route':r['id'], 'Description':r['desc'],
                            'Distance(km)':r['distance_km'],
                            'Est.Time(min)':r['time_min'],
                            'Risk Score':score, 'Risk Level':cat})
        res_df = pd.DataFrame(results).sort_values('Risk Score')
        best   = res_df.iloc[0]
        st.success(
            f"RECOMMENDED: {best['Route']} | "
            f"Risk: {best['Risk Score']} ({best['Risk Level']}) | "
            f"Distance: {best['Distance(km)']} km")

        st.subheader("Route Safety Comparison")
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
        colors_map = {'Low':'#2ecc71', 'Moderate':'#f39c12',
                      'High':'#e74c3c', 'Severe':'#8e44ad'}
        bar_colors = [colors_map.get(l, '#3498db') for l in res_df['Risk Level']]
        bars = ax.bar(res_df['Route'], res_df['Risk Score'],
                      color=bar_colors, edgecolor='black')
        ax.axhline(0.25, color='green',  linestyle='--', alpha=0.7)
        ax.axhline(0.50, color='orange', linestyle='--', alpha=0.7)
        ax.axhline(0.75, color='red',    linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Risk Score')
        patches = [
            mpatches.Patch(color='#8e44ad', label='Severe'),
            mpatches.Patch(color='#e74c3c', label='High'),
            mpatches.Patch(color='#f39c12', label='Moderate'),
            mpatches.Patch(color='#2ecc71', label='Low'),
        ]
        ax.legend(handles=patches, loc='upper right')
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(3)
        for bar, row in zip(bars, res_df.itertuples()):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f'{row._5}\n({row._6})',
                    ha='center', fontsize=9, fontweight='bold')
        st.pyplot(fig)

    st.markdown(FOOTER, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 6 — LIVE MAPS
# ══════════════════════════════════════════════════════════
elif page == "Live Maps":
    st.title("Geospatial Visualization")
    st.markdown("---")

    map_type  = st.radio("Map Type",
                         ["Heatmap", "Hotspot Clusters", "Risk Zones"],
                         horizontal=True)
    color_map = {'High':'red', 'Moderate':'orange', 'Low':'green'}

    if map_type == "Hotspot Clusters":
        st.subheader("Accident Hotspot Clusters")
        m = folium.Map(location=[34.05, -118.25], zoom_start=8,)
        for _, row in centroids.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=20,
                color=color_map.get(row['Density'], 'blue'),
                fill=True, fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>Hotspot</b><br>"
                    f"Density: {row['Density']}<br>"
                    f"Lat: {row['Latitude']:.2f}, "
                    f"Lng: {row['Longitude']:.2f}",
                    max_width=200)
            ).add_to(m)
        st_folium(m, width=900, height=500)

    elif map_type == "Heatmap":
        st.subheader("Accident Heatmap")
        from folium.plugins import HeatMap
        m = folium.Map(location=[37.0, -95.0], zoom_start=4,)
        heat_data = centroids[['Latitude', 'Longitude']].values.tolist()
        HeatMap(heat_data, radius=40, blur=30,
                gradient={0.2:'blue', 0.4:'lime',
                          0.65:'yellow', 1:'red'}).add_to(m)
        st_folium(m, width=900, height=500)

    else:
        st.subheader("Risk Zones")
        m = folium.Map(location=[37.0, -95.0], zoom_start=4,)
        for _, row in centroids.iterrows():
            folium.Circle(
                location=[row['Latitude'], row['Longitude']],
                radius=80000,
                color=color_map.get(row['Density'], 'blue'),
                fill=True, fill_opacity=0.3,
                popup=f"Risk Zone: {row['Density']}"
            ).add_to(m)
        st_folium(m, width=900, height=500)

    st.markdown(FOOTER, unsafe_allow_html=True)
