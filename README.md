Land Use Land Cover Change Analysis in the Niger Delta Using Multi-Temporal Satellite Data and Machine Learning
📌 Project Overview

This project demonstrates a machine learning–driven approach to Land Use / Land Cover (LULC) classification and change detection in the Niger Delta using synthetic multi-temporal satellite data. It simulates real-world remote sensing workflows for environmental monitoring, urban expansion analysis, and ecosystem assessment.

The system integrates spectral indices, supervised machine learning, and temporal comparison to identify land cover transitions over time.

🎯 Objectives

Simulate multi-temporal satellite observations for the Niger Delta

Classify land cover types using machine learning

Detect and quantify land cover changes across time periods

Provide a reproducible framework for environmental and geospatial research

🛰️ Land Cover Classes

The model classifies pixels into the following categories:

Water Bodies

Vegetation

Built-up Areas

Bare Land

Wetlands

🧠 Methodology

Synthetic Data Generation
Multi-temporal satellite-like data representing spectral bands (Red, NIR, Green, SWIR) were generated to mimic real satellite observations.

Feature Engineering
Vegetation and water indices such as NDVI and NDWI were computed to improve class separability.

Machine Learning Classification
A Random Forest Classifier was trained to perform LULC classification for each time period.

Change Detection Analysis
Classified maps from different years were compared to identify land cover transitions and spatial trends.

Evaluation & Visualization
Model performance metrics and feature importance analysis were generated to validate results.

🧪 Technologies Used

Python

NumPy

Pandas

Scikit-learn

Matplotlib

📂 Project Structure
├── lulc_change_niger_delta_ml.py
├── lulc_change_niger_delta_dataset.xlsx
├── README.md

▶️ How to Run

Clone the repository:

git clone https://github.com/your-username/lulc-change-niger-delta-ml.git


Install dependencies:

pip install numpy pandas scikit-learn matplotlib


Run the script:

python lulc_change_niger_delta_ml.py

📊 Output

LULC classification results

Feature importance plots

Land cover change statistics

Model performance metrics

⚠️ Disclaimer

This project uses synthetic data for research demonstration and educational purposes. It is designed to replicate real satellite-based workflows but should not be interpreted as operational environmental intelligence.

🌍 Applications

Environmental impact assessment

Urban growth monitoring

Wetland and mangrove change analysis

Climate adaptation and land management studies

Author
AGBOZU EBINGIYE NELVIN

LinkedIn: *https://www.linkedin.com/in/agbozu-ebi/


📜 License

This project is released under the MIT License.
