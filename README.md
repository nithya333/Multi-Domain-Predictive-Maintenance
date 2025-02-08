# Multi-Domain-Predictive-Maintenance

- Event: Ingenium Hackathon 2025 hosted by NIT Trichy
- Team Name: OctoPaul MaintAIneers
- Team: Nithyashree M, Krupa P Nadgir, Lekhana A, Manaswini Simhadri Kavali

## Problem Statement Overview:
 The problem statement focuses on the limitations of traditional maintenance
 strategies in industrial automation, such as manual operations, scheduled
 inspections, and reactive fixes, which often result in costly downtime,
 inefficiency, and unreliable performance. 

## Objectives
- Build a Scalable Solution Tailoring to Multiple Industrial Domains (eg.automotive, manufacturing,
aerospace).
- Support Feature-Specific Analysis of parameters for different types of machinery, enabling users to select
features most relevant to their operational goals. 
- Reduce resource wastage by predicting optimal maintenance schedules and spare part requirements.
- Develop a customizable dashboard for users to configure parameters, view real-time predictions, and receive insights tailored to their industry.

## Proposed Solution - Adaptive all-in-one software platform:
#### Domain-Based generalised predictive model templates:
 The software is planned to have multi-domain categorised predictive
 maintenance machinery parameters. Based on the historical data of these features, time-series forecasting and regression
 models will be built to all possible parameters (metrics) for each of these
 domains providing domain-specific predictive model templates.
 Example: Separate predictive model templates for various domains such as
 Hydraulic and Pneumatic Systems, Manufacturing and Processing Equipment,
 Conveyance Systems with their respective machinery parameters mentioned
 in the table.
 #### Adaptive Dashboard with Model Template Recommendations for Multi-Industry Applications:
 To make the system adaptable to various machinery types and operating
 environments, we came up with an all-in-one customisable dashboard with a
 recommendation system. The dashboard would suggest model templates to
 the user based on user input (e.g., machinery type, operating conditions).
 Through the interactive dashboard, the user can select the required model
 template and in turn select their domain-specific parameters that are
 applicable for their industry out of the various available parameters.On the
 chosen model template, fine-tuning is done to give predictions & forecasts of
 their industry data on these selective parameters in real-time.
 Example: The user inputs the information about their industry, and the
 recommendation system of our dashboard would recommend him/her the
 model template that is best suited for their firm and the user can customise
 their template by selecting required machine parameters applicable to his
 firm. The model will then give the user predictions of the tool wear and tear
 and will analyse the RUL (remaining useful life) of the machinery.

## Domain-specific model templates
1. **Naval System**
   - **Monitored Components**: Propeller, Hull, Gas Turbines, Gear Box, Controller.  
   - **Sensors Used**: Speed Log Sensor, Torque Sensor, Thermocouple, Pressure Sensor, Actuator Feedback, Flow Meter.  
   - **Predictive Maintenance**: Decay Coefficient Monitoring, Linear Relationship Modeling, Anomaly Detection, Early Failure Alerts.

2. **Hydraulic System**
   - **Monitored Components**: Pumps, Valves, Accumulators, Coolers.  
   - **Sensors Used**: Pressure Sensors, Flow Sensors, Temperature Sensors, Power.  
   - **Predictive Maintenance**: Failure Prediction, Maintenance Scheduling, Performance Analytics, Cost Savings.

3. **Ball Bearing System**
   - **Monitored Components**: Ball Bearings, Motor Shaft, Overall Motor Health, Couplings & Gears.  
   - **Sensors Used**: Accelerometers (Drive End, Fan End, Base).  
   - **Predictive Maintenance**: Vibration Analysis, Remaining Useful Life, Feature Extraction, Fault Severity Classification.

4. **Manufacturing & Processing Equipment**
   - **Monitored Components**: Motors & Drives, Cutting Equipment, Hydraulic Systems, Conditions Around.  
   - **Sensors Used**: Temperature Sensors, Speed Sensors, Torque Sensors, Wear Sensors.  
   - **Predictive Maintenance**: Failure Prediction, Component Health, Failure Alerts, Optimized Maintenance.


##  Key features of the proposed solution
* Single Universal System: Adaptable to various machinery types and operational environments through customized choice of model parameters applicable to their specific domain.
* Data Integration Layer: Aggregates sensor data with historical data such as maintenance logs and failure reports, providing a comprehensive view of system performance.
* AI/ML-Driven Models: Pre-trained models based on various machinery parameters across all industry domains, generalized for common patterns (e.g., motor vibrations, thermal anomalies). This universal model can be fine-tuned to apply industry-specific parameters through user customization.
