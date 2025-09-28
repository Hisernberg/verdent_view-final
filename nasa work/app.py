import streamlit as st
import ee
import geemap.foliumap as geemap
import folium
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import tempfile
import os
import requests
from dotenv import load_dotenv
from scipy.signal import find_peaks
import json

# Load environment variables
load_dotenv()

# =========================
# 🔑 Initialize GEE
# =========================
@st.cache_resource
def initialize_gee():
    """Initialize Google Earth Engine with proper authentication."""
    try:
        # Try to initialize with existing credentials
        ee.Initialize()
        return True
    except Exception as e:
        try:
            # If that fails, try to authenticate
            ee.Authenticate()
            ee.Initialize()
            return True
        except Exception as auth_error:
            st.error(f"Failed to initialize Google Earth Engine: {auth_error}")
            return False

# =========================
# 🌍 Streamlit Configuration
# =========================
st.set_page_config(
    page_title="verdent_view",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2D5016 0%, #4A7C59 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: #E8F5E8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4A7C59;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: #4A7C59;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: #2D5016;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 🏠 Main Application
# =========================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌸 BloomWatch AI</h1>
        <p>Advanced Ecological Intelligence Platform for Global Plant Bloom Monitoring</p>
        <p><em>Powered by NASA Satellite Data, Google Earth Engine & AI Analytics</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize GEE
    if not initialize_gee():
        st.error("Cannot proceed without Google Earth Engine access. Please check your credentials.")
        return

    # Sidebar Configuration
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # Dataset Selection
        dataset_options = {
            "MODIS Terra/Aqua (16-day)": "MODIS/006/MOD13Q1",
            "MODIS Terra/Aqua (8-day SR)": "MODIS/006/MOD09A1", 
            "Landsat 8/9 Surface Reflectance": "LANDSAT/LC08/C02/T1_L2",
            "VIIRS Vegetation Indices": "NOAA/VIIRS/001/VNP13A1",
            "Sentinel-2 Surface Reflectance": "COPERNICUS/S2_SR"
        }
        
        selected_dataset = st.selectbox(
            "📡 Satellite Dataset",
            options=list(dataset_options.keys()),
            index=0
        )
        
        # Vegetation Index
        index_options = ["NDVI", "EVI", "SAVI", "MSAVI"]
        vegetation_index = st.selectbox("🌱 Vegetation Index", index_options)
        
        # Time Range
        st.subheader("📅 Time Period")
        start_date = st.date_input(
            "Start Date",
            value=datetime.date(2023, 1, 1),
            min_value=datetime.date(2000, 1, 1),
            max_value=datetime.date.today()
        )
        end_date = st.date_input(
            "End Date", 
            value=datetime.date(2023, 12, 31),
            min_value=start_date,
            max_value=datetime.date.today()
        )
        
        # Country/Region Selection
        st.subheader("🌍 Study Area")
        countries = {
            "Bangladesh": [90.4125, 23.8103],
            "India": [77.2090, 28.6139],
            "USA": [-95.7129, 37.0902],
            "Brazil": [-47.8825, -15.7942],
            "Australia": [133.7751, -25.2744],
            "Kenya": [37.9062, -0.0236],
            "Germany": [10.4515, 51.1657],
            "Custom AOI": None
        }
        
        selected_country = st.selectbox("Select Country/Region", list(countries.keys()))
        
        if selected_country != "Custom AOI":
            center_coords = countries[selected_country]
            st.info(f"Using {selected_country} as study area")
        else:
            st.info("Draw custom area on the map below")

    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Interactive Map & Area Selection")
        
        # Create interactive map
        if selected_country != "Custom AOI":
            center_coords = countries[selected_country]
            m = geemap.Map(center=center_coords, zoom=6)
            
            # Add country boundary if available
            try:
                if selected_country == "Bangladesh":
                    country_boundary = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Bangladesh'))
                elif selected_country == "India":
                    country_boundary = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'India'))
                elif selected_country == "USA":
                    country_boundary = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'United States of America'))
                else:
                    country_boundary = None
                
                if country_boundary:
                    m.addLayer(country_boundary, {'color': 'red', 'fillColor': 'transparent'}, f'{selected_country} Boundary')
                    
            except Exception as e:
                st.warning(f"Could not load country boundary: {e}")
        else:
            m = geemap.Map(center=[20, 0], zoom=2)
        
        # Add drawing tools
        m.add_basemap("SATELLITE")
        map_data = m.to_streamlit(height=500)
    
    with col2:
        st.subheader("📊 Analysis Controls")
        
        # Analysis parameters
        bloom_threshold = st.slider("🌸 Bloom Detection Threshold", 0.1, 0.9, 0.3, 0.05)
        smoothing_window = st.slider("📈 Smoothing Window (days)", 5, 30, 15)
        
        # Run Analysis Button
        run_analysis = st.button("🚀 Run BloomWatch Analysis", type="primary")

    # Analysis Results Section
    if run_analysis:
        with st.spinner("🔄 Processing satellite data and detecting blooms..."):
            
            # Create AOI based on selection
            if selected_country != "Custom AOI":
                if selected_country == "Bangladesh":
                    aoi = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Bangladesh')).geometry()
                else:
                    # Use point-based AOI for other countries (can be expanded)
                    center_coords = countries[selected_country]
                    aoi = ee.Geometry.Point(center_coords).buffer(50000)  # 50km buffer
            else:
                # Default to Bangladesh for demo
                aoi = ee.Geometry.Point([90.4125, 23.8103]).buffer(50000)
            
            # Load and process data
            try:
                collection_id = dataset_options[selected_dataset]
                df_results = load_and_process_data(collection_id, vegetation_index, aoi, start_date, end_date)
                
                if not df_results.empty:
                    # Detect blooms
                    bloom_events = detect_bloom_events(df_results, bloom_threshold, smoothing_window)
                    
                    # Display results
                    display_results(df_results, bloom_events, selected_dataset, vegetation_index, selected_country)
                    
                    # Generate AI report
                    generate_ai_report(df_results, bloom_events, selected_dataset, vegetation_index, selected_country)
                    
                else:
                    st.error("No data available for the selected parameters. Please try different settings.")
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# =========================
# 📥 Data Loading Functions
# =========================
@st.cache_data(ttl=3600)
def load_and_process_data(collection_id, vegetation_index, aoi, start_date, end_date):
    """Load satellite data and compute vegetation indices."""
    try:
        # Load collection
        collection = ee.ImageCollection(collection_id).filterDate(str(start_date), str(end_date)).filterBounds(aoi)
        
        # Compute vegetation index based on dataset
        if "MOD13Q1" in collection_id:
            # MODIS vegetation indices
            if vegetation_index == "NDVI":
                band = "NDVI"
            elif vegetation_index == "EVI":
                band = "EVI"
            else:
                band = "NDVI"  # fallback
        elif "LANDSAT" in collection_id:
            # Landsat - compute NDVI from bands
            def compute_ndvi(image):
                ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
                return image.addBands(ndvi)
            collection = collection.map(compute_ndvi)
            band = "NDVI"
        elif "VIIRS" in collection_id:
            band = "NDVI"
        else:
            band = "NDVI"  # fallback
        
        # Extract time series
        def extract_values(image):
            stats = image.select(band).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=500,
                bestEffort=True
            )
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'value': stats.get(band),
                'system:time_start': image.get('system:time_start')
            })
        
        time_series = collection.map(extract_values)
        
        # Convert to pandas DataFrame
        data = time_series.getInfo()
        
        records = []
        for feature in data['features']:
            props = feature['properties']
            if props['value'] is not None:
                records.append({
                    'date': pd.to_datetime(props['date']),
                    'value': float(props['value']) / 10000 if 'MODIS' in collection_id else float(props['value']),  # Scale MODIS values
                    'timestamp': props['system:time_start']
                })
        
        df = pd.DataFrame(records)
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame()

# =========================
# 🌸 Bloom Detection Functions
# =========================
def detect_bloom_events(df, threshold, smoothing_window):
    """Detect bloom events using peak detection algorithms."""
    if df.empty:
        return []
    
    # Smooth the data
    df['smoothed'] = df['value'].rolling(window=min(smoothing_window, len(df)), center=True).mean()
    
    # Find peaks
    peaks, properties = find_peaks(
        df['smoothed'].fillna(df['value']), 
        height=threshold,
        distance=30,  # Minimum 30 days between peaks
        prominence=0.1
    )
    
    bloom_events = []
    for i, peak_idx in enumerate(peaks):
        bloom_event = {
            'peak_date': df.iloc[peak_idx]['date'],
            'peak_value': df.iloc[peak_idx]['value'],
            'peak_idx': peak_idx,
            'prominence': properties['prominences'][i] if 'prominences' in properties else 0,
            'height': properties['peak_heights'][i] if 'peak_heights' in properties else df.iloc[peak_idx]['value']
        }
        
        # Find bloom start and end (half-maximum points)
        half_max = bloom_event['peak_value'] * 0.7
        
        # Find start
        start_idx = peak_idx
        for j in range(peak_idx, max(0, peak_idx - 60), -1):  # Look back up to 60 days
            if df.iloc[j]['value'] < half_max:
                start_idx = j
                break
        
        # Find end
        end_idx = peak_idx
        for j in range(peak_idx, min(len(df), peak_idx + 60)):  # Look forward up to 60 days
            if df.iloc[j]['value'] < half_max:
                end_idx = j
                break
        
        bloom_event['start_date'] = df.iloc[start_idx]['date']
        bloom_event['end_date'] = df.iloc[end_idx]['date']
        bloom_event['duration'] = (bloom_event['end_date'] - bloom_event['start_date']).days
        
        bloom_events.append(bloom_event)
    
    return bloom_events

# =========================
# 📊 Results Display Functions
# =========================
def display_results(df, bloom_events, dataset, vegetation_index, country):
    """Display analysis results with visualizations."""
    
    st.subheader("📈 Time Series Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🌸 Bloom Events</h4>
            <h2>{len(bloom_events)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_intensity = np.mean([event['peak_value'] for event in bloom_events]) if bloom_events else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Avg Intensity</h4>
            <h2>{avg_intensity:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_duration = np.mean([event['duration'] for event in bloom_events]) if bloom_events else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>⏱️ Avg Duration</h4>
            <h2>{avg_duration:.0f} days</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        data_points = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <h4>📡 Data Points</h4>
            <h2>{data_points}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Time series plot
    fig = create_time_series_plot(df, bloom_events, vegetation_index)
    st.plotly_chart(fig, use_container_width=True)
    
    # Bloom calendar
    if bloom_events:
        st.subheader("🗓️ Bloom Calendar")
        bloom_df = pd.DataFrame([{
            'Peak Date': event['peak_date'].strftime('%Y-%m-%d'),
            'Start Date': event['start_date'].strftime('%Y-%m-%d'),
            'End Date': event['end_date'].strftime('%Y-%m-%d'),
            'Duration (days)': event['duration'],
            'Peak Intensity': f"{event['peak_value']:.3f}",
            'Prominence': f"{event['prominence']:.3f}"
        } for event in bloom_events])
        
        st.dataframe(bloom_df, use_container_width=True)
    
    # Export options
    st.subheader("📥 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Time Series (CSV)",
            csv_data,
            f"bloomwatch_{country}_{vegetation_index}_{datetime.date.today()}.csv",
            "text/csv"
        )
    
    with col2:
        if bloom_events:
            bloom_csv = pd.DataFrame(bloom_events).to_csv(index=False)
            st.download_button(
                "⬇️ Download Bloom Events (CSV)",
                bloom_csv,
                f"bloom_events_{country}_{vegetation_index}_{datetime.date.today()}.csv",
                "text/csv"
            )

def create_time_series_plot(df, bloom_events, vegetation_index):
    """Create interactive time series plot with bloom events."""
    fig = go.Figure()
    
    # Add main time series
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['value'],
        mode='lines+markers',
        name=f'{vegetation_index} Values',
        line=dict(color='#4A7C59', width=2),
        marker=dict(size=4)
    ))
    
    # Add smoothed line if available
    if 'smoothed' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['smoothed'],
            mode='lines',
            name='Smoothed Trend',
            line=dict(color='#2D5016', width=3, dash='dash')
        ))
    
    # Add bloom events
    for i, event in enumerate(bloom_events):
        # Peak marker
        fig.add_trace(go.Scatter(
            x=[event['peak_date']],
            y=[event['peak_value']],
            mode='markers',
            name=f'Bloom Peak {i+1}',
            marker=dict(
                size=15,
                color='red',
                symbol='star',
                line=dict(width=2, color='darkred')
            ),
            showlegend=i == 0
        ))
        
        # Bloom period shading
        fig.add_vrect(
            x0=event['start_date'],
            x1=event['end_date'],
            fillcolor="rgba(255, 192, 203, 0.3)",
            layer="below",
            line_width=0,
        )
    
    fig.update_layout(
        title=f'{vegetation_index} Time Series with Bloom Detection',
        xaxis_title='Date',
        yaxis_title=f'{vegetation_index} Value',
        hovermode='x unified',
        height=500,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

# =========================
# 🤖 AI Report Generation
# =========================
def generate_ai_report(df, bloom_events, dataset, vegetation_index, country):
    """Generate AI-powered ecological report using Groq."""
    st.subheader("🤖 AI-Generated Ecological Report")
    
    with st.spinner("🧠 Generating intelligent ecological insights..."):
        try:
            # Prepare data summary for LLM
            data_summary = prepare_data_summary(df, bloom_events, dataset, vegetation_index, country)
            
            # Generate report using Groq
            report = call_groq_api(data_summary)
            
            # Display report
            st.markdown("### 📋 Ecological Analysis Report")
            st.markdown(report)
            
            # Export report as PDF
            pdf_buffer = create_pdf_report(report, data_summary)
            if pdf_buffer:
                st.download_button(
                    "⬇️ Download Full Report (PDF)",
                    pdf_buffer,
                    f"bloomwatch_report_{country}_{datetime.date.today()}.pdf",
                    "application/pdf"
                )
                
        except Exception as e:
            st.error(f"Failed to generate AI report: {str(e)}")
            st.info("Please check your Groq API key in the .env file")

def prepare_data_summary(df, bloom_events, dataset, vegetation_index, country):
    """Prepare structured data summary for LLM input."""
    summary = {
        'dataset': dataset,
        'vegetation_index': vegetation_index,
        'country': country,
        'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        'total_observations': len(df),
        'bloom_events_count': len(bloom_events),
        'average_vegetation_index': df['value'].mean(),
        'max_vegetation_index': df['value'].max(),
        'min_vegetation_index': df['value'].min(),
        'bloom_events': []
    }
    
    for i, event in enumerate(bloom_events):
        summary['bloom_events'].append({
            'event_number': i + 1,
            'peak_date': event['peak_date'].strftime('%Y-%m-%d'),
            'start_date': event['start_date'].strftime('%Y-%m-%d'),
            'end_date': event['end_date'].strftime('%Y-%m-%d'),
            'duration_days': event['duration'],
            'peak_intensity': event['peak_value'],
            'prominence': event['prominence']
        })
    
    return summary

def call_groq_api(data_summary):
    """Call Groq API to generate ecological report."""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        return "Error: Groq API key not found. Please add GROQ_API_KEY to your .env file."
    
    # Construct detailed prompt
    prompt = f"""
You are BloomWatch AI, an expert ecological analyst specializing in satellite-based vegetation monitoring and plant phenology.

ANALYSIS DATA:
- Dataset: {data_summary['dataset']}
- Vegetation Index: {data_summary['vegetation_index']}
- Study Area: {data_summary['country']}
- Analysis Period: {data_summary['date_range']}
- Total Observations: {data_summary['total_observations']}
- Bloom Events Detected: {data_summary['bloom_events_count']}
- Average {data_summary['vegetation_index']}: {data_summary['average_vegetation_index']:.3f}
- Peak {data_summary['vegetation_index']}: {data_summary['max_vegetation_index']:.3f}

BLOOM EVENTS SUMMARY:
"""
    
    for event in data_summary['bloom_events']:
        prompt += f"""
Event {event['event_number']}:
- Peak Date: {event['peak_date']}
- Duration: {event['duration_days']} days ({event['start_date']} to {event['end_date']})
- Peak Intensity: {event['peak_intensity']:.3f}
- Prominence: {event['prominence']:.3f}
"""
    
    prompt += """

TASK: Generate a comprehensive ecological analysis report that includes:

1. **Executive Summary**: Brief overview of blooming activity and key findings

2. **Bloom Pattern Analysis**: 
   - Seasonal timing and duration patterns
   - Intensity variations and trends
   - Comparison to typical bloom cycles

3. **Ecological Significance**:
   - Impact on local ecosystems and biodiversity
   - Pollination implications for agriculture
   - Wildlife habitat and food source considerations

4. **Environmental Health Implications**:
   - Pollen production and allergy risk assessment
   - Air quality considerations
   - Public health recommendations

5. **Agricultural Insights**:
   - Crop timing and yield implications
   - Pest and disease risk factors
   - Irrigation and management recommendations

6. **Climate and Environmental Factors**:
   - Potential climate change indicators
   - Drought or stress indicators
   - Seasonal anomalies and their causes

7. **Actionable Recommendations**:
   - For farmers and agricultural managers
   - For public health agencies
   - For environmental conservation efforts
   - For policy makers

STYLE REQUIREMENTS:
- Professional, scientific tone accessible to diverse stakeholders
- Use specific data points and quantitative insights
- Balance technical accuracy with practical applicability
- Include both immediate and long-term implications
- Provide clear, actionable recommendations

Generate a detailed, insightful report that transforms satellite data into meaningful ecological intelligence.
"""
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": "You are BloomWatch AI, an expert ecological analyst specializing in satellite-based vegetation monitoring, plant phenology, and environmental intelligence. Provide detailed, scientifically accurate, and practically useful ecological insights."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Failed to generate report: {str(e)}"

def create_pdf_report(report_text, data_summary):
    """Create PDF report with the AI-generated content."""
    try:
        buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        doc = SimpleDocTemplate(buffer.name, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Build PDF content
        story = []
        
        # Title
        title = Paragraph("🌸 BloomWatch AI - Ecological Analysis Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Metadata
        metadata = f"""
        <b>Study Area:</b> {data_summary['country']}<br/>
        <b>Dataset:</b> {data_summary['dataset']}<br/>
        <b>Vegetation Index:</b> {data_summary['vegetation_index']}<br/>
        <b>Analysis Period:</b> {data_summary['date_range']}<br/>
        <b>Bloom Events Detected:</b> {data_summary['bloom_events_count']}<br/>
        <b>Report Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        meta_para = Paragraph(metadata, styles['Normal'])
        story.append(meta_para)
        story.append(Spacer(1, 20))
        
        # Report content
        report_para = Paragraph(report_text.replace('\n', '<br/>'), styles['Normal'])
        story.append(report_para)
        
        # Build PDF
        doc.build(story)
        
        # Read the PDF content
        with open(buffer.name, 'rb') as f:
            pdf_content = f.read()
        
        # Clean up
        os.unlink(buffer.name)
        
        return pdf_content
        
    except Exception as e:
        st.error(f"Failed to create PDF: {str(e)}")
        return None

# =========================
# 🚀 Run Application
# =========================
if __name__ == "__main__":
    main()