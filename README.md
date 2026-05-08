# 🏥 Emergency Department BI Solution — Patient Flow & Revenue Optimization

> A Business Intelligence project designed to reduce ED overcrowding, minimize revenue leakage, and improve patient outcomes at **Hôpital Charles Nicolle**.

---

## 📌 Project Overview

The Emergency Department at Hôpital Charles Nicolle handles approximately **180,000 visits per year** but faces critical operational challenges: excessive patient wait times, high "Left Without Being Seen" (LWBS) rates, and misaligned staffing schedules.

This BI solution visualizes the **end-to-end patient journey** to help administrators identify and relieve throughput bottlenecks — with a primary goal of reducing the LWBS rate by **40%** within the first year.

---

## 🎯 Business Objectives

| Objective | Description |
|---|---|
| Bottleneck Detection | Identify the longest delays across the patient journey stages |
| Peak Demand Analysis | Determine high-volume days and hours for better resource planning |
| Staffing Alignment | Compare actual nurse-to-patient ratios against clinical standards |
| LWBS Risk Profiling | Profile patients most likely to leave without being seen |
| Triage Effectiveness | Detect potential triage misclassifications |
| Re-admission Risk | Track 72-hour return rates by diagnosis |
| Resource Saturation | Monitor bed occupancy trends throughout the day |

---

## 📊 Key Performance Indicators (KPIs)

| # | KPI | Formula | Business Relevance |
|---|---|---|---|
| 1 | **Door-to-Doc Time** | Medical assessment time − Arrival time | Most critical metric for patient safety |
| 2 | **LWBS Rate** | Patients left ÷ Total arrivals × 100 | Measures lost revenue & liability risk |
| 3 | **Average Length of Stay (LOS)** | Discharge time − Arrival time | Measures total system efficiency |
| 4 | **Boarding/Waiting Time** | Admission to ward − Decision to admit | Delay in moving patients out of ED |
| 5 | **Bed Occupancy Rate** | Occupied beds ÷ Total beds × 100 | Predicts department saturation |
| 6 | **72-Hour Return Rate** | Returns < 72h ÷ Total discharges × 100 | Indicates premature discharge risk |
| 7 | **Patients per Staff Hour** | Total patient volume ÷ Total nursing hours × 100 | Optimizes staffing and shift planning |
| 8 | **Average Triage Time** | Triage start − Arrival time | Measures speed of initial risk stratification |

---

## 🛠️ Tools & Technologies

| Layer | Tool |
|---|---|
| Dashboard & Visualization | Power BI |
| Data Processing & Analysis | Python |
| Data Source | Electronic Health Record (EHR) System |
| Data Modeling | Star Schema (Fact + Dimension tables) |

---

## 📁 Repository Structure

```
📁 ed-bi-solution/
├── 📁 data/
│   ├── 📁 raw/               # Original CSV/Excel files from EHR system
│   ├── 📁 processed/         # Cleaned and transformed datasets
│   └── data_dictionary.md    # Column definitions and data types
├── 📁 dashboard/
│   └── ed_patient_flow.pbix  # Power BI dashboard file
├── 📁 scripts/
│   └── *.py                  # Python scripts for data processing & analysis
├── 📁 assets/
│   └── 📁 screenshots/       # Dashboard preview images
├── 📁 docs/
│   └── use_case.pdf          # Full use case & business understanding document
└── README.md
```

---

## 🗂️ Data Dictionary

### `Fact_ED_Visit_Metrics` — Core fact table (1 row per patient visit)

| Column | Description | Type |
|---|---|---|
| `Encounter_ID` | Unique identifier per patient visit | Integer |
| `Door_to_Doc_Time` | Time from arrival to medical assessment (minutes) | Float |
| `Total_LOS` | Total length of stay in the ED (minutes) | Float |
| `Boarding_Time` | Time waiting after admission decision (minutes) | Float |
| `LWBS_Flag` | 1 if patient left without being seen, else 0 | Boolean |
| `Returned_72h_Flag` | 1 if patient returned within 72 hours, else 0 | Boolean |
| `Total_Staff_Time` | Total nursing/doctor time allocated (minutes) | Float |
| `Revenue_Generated` | Total billed amount (TND) | Float |
| `FK_Date` | Foreign key → Dim_Date | Integer |
| `FK_Patient` | Foreign key → Dim_Patient | Integer |
| `FK_Staff` | Foreign key → Dim_Staff | Integer |
| `FK_Diagnosis` | Foreign key → Dim_Diagnosis | Integer |

### `Dim_Patient` — Patient dimension

| Column | Description | Type |
|---|---|---|
| `Patient_ID` | Unique patient identifier | Integer |
| `Acuity_Level` | ESI triage score (1–5) | Integer |
| `Admission_Status` | Admitted / Discharged / Transferred | String |
| `Patient_Age_Group` | Age bracket of the patient | String |
| `Patient_Zip_Code` | Geographic location | String |
| `Reason_for_Visit` | Primary presenting complaint | String |

### `Dim_Staff` — Staff dimension

| Column | Description | Type |
|---|---|---|
| `Staff_ID` | Unique staff identifier | Integer |
| `Staff_Role` | MD / RN / Tech | String |
| `Contract_Type` | Full-time / Part-time | String |
| `Shift_Scheduled` | Assigned shift | String |
| `Average_Performance_Score` | Performance metric | Float |

### `Dim_Date` — Date dimension

| Column | Description | Type |
|---|---|---|
| `Date_Key` | Surrogate key | Integer |
| `Full_Date` | Calendar date | Date |
| `Day_of_Week` | Monday–Sunday | String |
| `Month_Name` | January–December | String |
| `Is_Holiday_Flag` | 1 if public holiday, else 0 | Boolean |
| `Fiscal_Quarter` | Q1–Q4 | String |

---

## 📷 Dashboard Preview

> *Screenshots coming soon — see the `/assets/screenshots/` folder.*

---

## 👥 Team

| Name |
|---|
| Mouhamed Dhia Ben Kilani |
| Mohamed Aziz Amri |
| Wassef Bellila |
| Ala Chaaleb |

---

## 📄 Documentation

For full business context, analytical objectives, and data modeling details:

👉 [View Use Case & Business Understanding](docs/use_case.pdf)

---

*Project submitted: December 2025*
