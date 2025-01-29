# CVM-Syntegra-Insights

**Description**:  
A comprehensive analysis of Syntegra’s Medicare CCLF (Claims) dataset focused on Cardio Vascular Metabolic (CVM) diseases. This repository answers key business questions involving claims trends, healthcare provider (HCP) prescribing behaviors, and patient age demographics. The insights will aid Sales & Marketing leadership in refining strategies for CVM products and improving patient outcomes.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Business Questions](#business-questions)
- [Key Insights](#key-insights)
- [Repository Structure](#repository-structure)
- [Usage & Notebooks](#usage-notebooks)
- [Data Quality Checks](#data-quality-checks)
- [Vizualizations](#vizualizations)
- [Future Work](#future-work)
- [Contact](#contact)

---

## Project Overview
In **BIA-810D: Healthcare and Advanced Data Analytics** (Fall 2024), we analyzed de-identified Medicare CCLF claims data (sourced from [Syntegra.io](https://www.syntegra.io/download-syntegra-data)) with a focus on Cardio Vascular Metabolic (CVM) diseases. By examining Part A and Part B (DME & Physicians) claims, we derived key insights on:
- CVM claims trends vs. total claims
- Healthcare provider (HCP) claim volume and segmentation
- Patient age buckets and their claim volumes
- Recommended strategies for Sales & Marketing teams

**Objectives:**
1. **Identify** the share of CVM procedures among total claims from 2016 to 2018.  
2. **Segment** HCPs based on their CVM claim counts (Disease Aware, Trialists, Rising Stars, High-Volume Prescribers).  
3. **Analyze** patient age buckets for CVM claims and year-over-year volume changes.  
4. **Recommend** data-driven strategies to optimize sales force deployment and marketing budgets.

---

## Dataset Description
The following datasets (in CSV format) are used to address the questions:

- **`beneficiary_demographics.csv`**  
  Contains patient-level demographic information (DOB, gender, etc.).
- **`parta_claims_header.csv`**  
  Header-level details for Part A claims (inpatient/outpatient).
- **`parta_claims_revenue_center_detail.csv`**  
  Detailed charges for revenue centers (linked to Part A claims).
- **`parta_diagnosis_code.csv`**  
  ICD diagnosis codes linked to each claim.
- **`parta_procedure_code.csv`**  
  ICD procedure codes linked to each claim (not extensively used in this analysis but included).
- **`partb_dme.csv`**  
  Durable Medical Equipment (DME) claims data under Part B.
- **`partb_physicians.csv`**  
  Physician/supplier Part B claims data, including HCPCS/CPT codes.

Supporting/auxiliary files include:
- **`Analysis.docx`**  
  Additional analysis notes and approach.
- **`ImportantLinks.txt`**  
  Links related to references, data sources, and documentation.
- **`Muhammad_Dastgir.ipynb`**, **`Muhammad_Dastgir.pdf`**, **`Muhammad_Dastgir_new_submission.html`**  
  Jupyter Notebook, PDF, and HTML report forms of the analysis.

---

## Business Questions
1. **Trends of CVM Claims (2016–2018)**
   - What percentage of total claims are CVM-related each year?
   - How do these trends impact sales & marketing strategies?

2. **HCP Behavior & Segmentation**
   - How many HCPs file 1 CVM claim, 2–4, 5–9, and 10+?
   - Implications for sales force allocation (in-person vs. digital promotion)?

3. **Patient Age Demographics (2016–2018)**
   - How are CVM claims distributed across various age buckets (18–59, 60–69, 70–79, 80+)?
   - Year-over-year changes and recommended marketing shifts for each segment?

---

## Key Insights
1. **Rising CVM Share**  
   - CVM claims exhibit an upward trend, indicating increased prevalence of heart disease, stroke, and diabetes-related procedures.
   - Calls for stronger engagement with cardiologists and primary care physicians.

2. **HCP Segmentation Strategy**  
   - **Disease Aware (1 CVM claim)** – Large in number and often responsible for surprisingly high total costs.
   - **Trialists (2–4 claims)** – Steady growth, potential to become consistent prescribers.
   - **Rising Stars (5–9 claims)** – Smaller cohort with high engagement potential.
   - **High-Volume Prescribers (10+ claims)** – Typically comfortable prescribing CVM therapies; focus on brand-specific messaging.

3. **Patient Age Buckets**  
   - Largest volume of CVM claims typically in older age brackets (70+).
   - Marketing budgets could shift toward senior education, while younger populations benefit from more prevention-focused campaigns.

4. **Sales & Marketing Implications**  
   - **Disease Aware** providers may represent key opportunities if educated on broader treatment options.
   - **Digital engagements** for lower-volume prescribers; **in-person sales** for high-volume.  
   - Highlight real-world evidence to bolster prescribing confidence.

---

## Repository Structure

CVM_Syntegra_Insights
- beneficiary_demographics.csv
- parta_claims_header.csv
- parta_claims_revenue_center_detail.csv
- parta_diagnosis_code.csv
- parta_procedure_code.csv
- partb_dme.csv
- partb_physicians.csv
- Analysis.docx
- ImportantLinks.txt
- Muhammad_Dastgir.ipynb
- Muhammad_Dastgir.pdf
- Muhammad_Dastgir_new_submission.html
- README.md  <– You are here!


## Usage & Notebooks

- **`Muhammad_Dastgir.ipynb`**: The primary analysis notebook.  
  - Imports CSV files, cleans data, merges Part A & Part B claims.  
  - Performs data quality checks (e.g., uniqueness, null checks).  
  - Generates visualizations (stacked bar charts, distribution plots).  
  - Answers key business questions with relevant tables & discussion points.

---

## Technical Approach

1. **Data Loading & Merging**  
   - Unified multiple datasets on `claim_id` and `patient_id` via outer/left joins as appropriate.  
   - Appended DME and Physicians data to form a comprehensive dataset of Medicare claims.

2. **Data Quality Checks**  
   - Validated uniqueness of identifiers (e.g., `claim_id`).  
   - Ensured all essential columns (e.g., `hcpcs_code`, `diagnosis_code`) are non-null when required.  
   - Confirmed date formats are uniform (`YYYY-MM-DD`).

3. **Segmentation & Aggregation**  
   - Classified HCPs based on CVM claims volume:
     - **Disease Aware** (1 CVM claim)  
     - **Trialists** (2–4 CVM claims)  
     - **Rising Stars** (5–9 CVM claims)  
     - **High-Volume Prescribers** (≥ 10 CVM claims)  
   - Calculated patient ages at time of claim; grouped claims by age buckets (18–59, 60–69, 70–79, 80+).

4. **Visualization & Reporting**  
   - Created stacked bar charts to show CVM vs. non-CVM claims across years (2016–2018).  
   - Charted HCP segmentation distributions per year.  
   - Summarized patient age bucket trends year-over-year.

---

## Data Quality Checks

Some key checks implemented:

1. **Duplicates**  
   - Verified `claim_id` uniqueness (where expected).  
   - Removed duplicate rows in case of repeated records.

2. **Missing Values**  
   - Dropped rows missing core identifying info (`hcpcs_code`, `diagnosis_code`) or replaced as needed.  
   - Ensured essential date columns (`claim_date`) are present.

3. **Valid Formats**  
   - Ensured `claim_date` is parsed in `YYYY-MM-DD`.  
   - Converted numeric fields (`claim_cost`, `patient_age`) where applicable.

---

## Visualizations

Within the notebook, you’ll find:

1. **100% Stacked Bar Chart**  
   - Depicting CVM vs. non-CVM claim share for 2016, 2017, and 2018.

2. **Stacked Bar for HCP Segmentation**  
   - Disease Aware, Trialists, Rising Stars, High-Volume Prescribers per year (2016–2018).

3. **Age Bucket Stacked Bar Chart**  
   - Number of CVM claims in each age bracket (18–59, 60–69, 70–79, 80+) for each year.

4. **Tabular Year-Over-Year Changes**  
   - Percentage change in claim volume for each patient-age bucket from one year to the next.

---

## Future Work

- **Deeper Diagnosis Categorization**  
  - Map ICD codes to more refined categories (e.g., Type 2 Diabetes vs. Hypertension) for granular insights.

- **Predictive Modeling**  
  - Develop machine learning models (e.g., regression, time series) to forecast CVM claim volumes.

- **Cost Stratification**  
  - Investigate highest-cost procedures within CVM claims, enabling more targeted budget allocations.

- **Geographic Analyses**  
  - Combine demographic info with state/county details to locate high-burden regions.

---

## Contact

**Author**: Muhammad Hamid Ahmed Dastgir  
**Course**: BIA-810D – Healthcare and Advanced Data Analytics (Fall 2024)

For questions or collaboration opportunities, please reach out at:  
**Email**: `mdastgir@stevens.edu`

> **Disclaimer**: This analysis uses de-identified data from Syntegra. All references to real patients/providers are coincidental. This project is for educational purposes; any commercial or clinical application should undergo further validation and ethical review.

