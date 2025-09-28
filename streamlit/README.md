# 🌸 Verdent_view AI

**Advanced Ecological Intelligence Platform for Global Plant Bloom Monitoring**

Powered by NASA Satellite Data, Google Earth Engine & AI Analytics

## 🌟 Features

- **Satellite Data Analysis**: Process MODIS, Landsat, Sentinel-2, and VIIRS data
- **Bloom Detection**: Advanced algorithms to identify plant blooming patterns
- **AI Reports**: Intelligent ecological insights powered by Groq AI
- **Interactive Maps**: Geospatial visualization with area selection
- **Time Series Analysis**: Comprehensive vegetation index tracking
- **Export Capabilities**: Download data and reports in multiple formats

## 🚀 Live Demo

[Visit BloomWatch AI](https://your-app-name.streamlit.app)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Google Earth Engine account
- Groq API key

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bloomwatch-ai.git
   cd bloomwatch-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Configure Google Earth Engine**
   ```bash
   earthengine authenticate
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🌍 Supported Datasets

- **MODIS Terra/Aqua**: 16-day and 8-day surface reflectance
- **Landsat 8/9**: Surface reflectance data
- **Sentinel-2**: High-resolution surface reflectance
- **VIIRS**: Vegetation indices from NOAA

## 📊 Vegetation Indices

- **NDVI**: Normalized Difference Vegetation Index
- **EVI**: Enhanced Vegetation Index
- **SAVI**: Soil Adjusted Vegetation Index
- **MSAVI**: Modified Soil Adjusted Vegetation Index

## 🔐 Environment Setup

### Required Environment Variables

Create a `.env` file with:

```bash
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
```

### Google Earth Engine Authentication

For production deployment, use a service account:

1. Create a service account in Google Cloud Console
2. Download the JSON key file
3. Set `GOOGLE_APPLICATION_CREDENTIALS` to the file path

## 📱 Usage

1. **Select Dataset**: Choose from multiple satellite data sources
2. **Configure Parameters**: Set vegetation index, time range, and study area
3. **Run Analysis**: Process satellite data and detect bloom events
4. **View Results**: Interactive visualizations and detailed metrics
5. **Generate Report**: AI-powered ecological insights
6. **Export Data**: Download results in CSV and PDF formats

## 🌍 Study Areas

Pre-configured countries and regions:
- Bangladesh
- India
- USA
- Brazil
- Australia
- Kenya
- Germany
- Custom Area of Interest (AOI)

## 🤖 AI Features

- **Bloom Pattern Analysis**: Seasonal timing and intensity patterns
- **Ecological Significance**: Impact on biodiversity and ecosystems
- **Environmental Health**: Pollen and air quality implications
- **Agricultural Insights**: Crop timing and management recommendations
- **Climate Indicators**: Environmental change detection

## 📈 Bloom Detection Algorithm

The app uses advanced signal processing techniques:
- Peak detection with configurable thresholds
- Smoothing algorithms to reduce noise
- Prominence-based filtering
- Duration and intensity analysis

## 🔄 Data Processing Pipeline

1. **Satellite Data Ingestion**: Query and filter satellite imagery
2. **Vegetation Index Calculation**: Compute NDVI, EVI, SAVI, MSAVI
3. **Time Series Extraction**: Aggregate data over study areas
4. **Bloom Detection**: Apply peak detection algorithms
5. **Visualization**: Create interactive plots and maps
6. **AI Analysis**: Generate ecological insights

## 🚀 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Configure secrets in Streamlit dashboard
4. Deploy application

### Required Secrets (Streamlit Cloud)

Add these in your Streamlit Cloud secrets:

```toml
GROQ_API_KEY = "your_groq_api_key"

[gee_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40project.iam.gserviceaccount.com"
```

## 🛠️ Development

### Project Structure

```
bloomwatch-ai/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
├── .streamlit/
│   └── config.toml           # Streamlit configuration
└── assets/                   # Static assets (optional)
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Earth Engine**: Satellite data processing platform
- **NASA**: Earth observation data
- **Groq**: AI-powered report generation
- **Streamlit**: Web application framework

## 📞 Support

For issues and questions:
- Open a GitHub issue
- Contact: your-email@example.com

## 🔗 Links

- [Google Earth Engine](https://earthengine.google.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Groq API](https://groq.com/)

---


**BloomWatch AI** - Transforming satellite data into ecological intelligence 🌸🤖
