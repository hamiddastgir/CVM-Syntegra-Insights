# Mid-Term Project

#### Overview
- Use Medicare CCLF Claims from Syntegra dataset to answer key business questions
- Extra credit for building up on the questions below (additional questions + answers)
- One Jupyter notebook solution with clear Python code and all cell outputs available
- At least two data quality checks

## Step 0. Prepare raw input datasets

Here we will 1) load original datasets, 2) remove unused columns, 3) de-duplicate rows, and 4) join datasets, not necessarily in this order

Assumptions: 
- Claim ID (cur_clm_uniq_id) represents one claim, which may or may not have more than one code (code could be HCPCS/CPT, diagnosis, procedure...)
- There is a one-to-many relationship between patient IDs (bene_mbi_id) and claim IDs (cur_clm_uniq_id), i.e. each claim is unique to one patient, but one patient can have more than one claim

### 0.1 Import required packages


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
```


```python
# Turn off the automatic setting that redacts the columns/rows from the dataframe output
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 200)
```

### 0.2 Load & select columns to be used from raw (original) datasets

### 0.2.1 Load & select columns from Claims Header dataset


```python
# Load Claims Header dataset
parta_claims_header_raw_df = pd.read_csv("/Users/hamiddastgir/Library/CloudStorage/Dropbox/Semester 3/BIA 810 - Healthcare Analytics/Mid Term/Syntegra Datasets Files/parta_claims_header.csv")
parta_claims_header_raw_df.sort_values(by=['cur_clm_uniq_id'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cur_clm_uniq_id</th>
      <th>prvdr_oscar_num</th>
      <th>bene_mbi_id</th>
      <th>bene_hic_num</th>
      <th>clm_type_cd</th>
      <th>clm_from_dt</th>
      <th>clm_thru_dt</th>
      <th>clm_bill_fac_type_cd</th>
      <th>clm_bill_clsfctn_cd</th>
      <th>prncpl_dgns_cd</th>
      <th>admtg_dgns_cd</th>
      <th>clm_mdcr_npmt_rsn_cd</th>
      <th>clm_pmt_amt</th>
      <th>clm_nch_prmry_pyr_cd</th>
      <th>prvdr_fac_fips_st_cd</th>
      <th>bene_ptnt_stus_cd</th>
      <th>dgns_drg_cd</th>
      <th>clm_op_srvc_type_cd</th>
      <th>fac_prvdr_npi_num</th>
      <th>oprtg_prvdr_npi_num</th>
      <th>atndg_prvdr_npi_num</th>
      <th>othr_prvdr_npi_num</th>
      <th>clm_adjsmt_type_cd</th>
      <th>clm_efctv_dt</th>
      <th>clm_idr_ld_dt</th>
      <th>bene_eqtbl_bic_hicn_num</th>
      <th>clm_admsn_type_cd</th>
      <th>clm_admsn_src_cd</th>
      <th>clm_bill_freq_cd</th>
      <th>clm_query_cd</th>
      <th>dgns_prcdr_icd_ind</th>
      <th>clm_mdcr_instnl_tot_chrg_amt</th>
      <th>clm_mdcr_ip_pps_cptl_ime_amt</th>
      <th>clm_oprtnl_ime_amt</th>
      <th>clm_mdcr_ip_pps_dsprprtnt_amt</th>
      <th>clm_hipps_uncompd_care_amt</th>
      <th>clm_oprtnl_dsprtnt_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>510</th>
      <td>100190</td>
      <td>111821</td>
      <td>1228</td>
      <td>NaN</td>
      <td>40</td>
      <td>2018-06-10</td>
      <td>2018-06-10</td>
      <td>7</td>
      <td>7</td>
      <td>M1611</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>127.79</td>
      <td>NaN</td>
      <td>11</td>
      <td>1</td>
      <td>NaN</td>
      <td>F</td>
      <td>1780608992</td>
      <td>NaN</td>
      <td>1.972732e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>415.80</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>521</th>
      <td>100402</td>
      <td>100226</td>
      <td>1261</td>
      <td>NaN</td>
      <td>60</td>
      <td>2017-05-27</td>
      <td>2017-06-02</td>
      <td>1</td>
      <td>1</td>
      <td>K5733</td>
      <td>K5733</td>
      <td>NaN</td>
      <td>10602.46</td>
      <td>NaN</td>
      <td>10</td>
      <td>6</td>
      <td>330.0</td>
      <td>NaN</td>
      <td>1689611501</td>
      <td>NaN</td>
      <td>1.285688e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>70795.63</td>
      <td>609.13</td>
      <td>0.00</td>
      <td>13.92</td>
      <td>231.15</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>525</th>
      <td>100464</td>
      <td>360051</td>
      <td>12978</td>
      <td>NaN</td>
      <td>40</td>
      <td>2017-06-26</td>
      <td>2017-06-26</td>
      <td>1</td>
      <td>3</td>
      <td>R079</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>199.45</td>
      <td>NaN</td>
      <td>36</td>
      <td>1</td>
      <td>NaN</td>
      <td>C</td>
      <td>1073688354</td>
      <td>NaN</td>
      <td>1.982693e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>2709.80</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>536</th>
      <td>100698</td>
      <td>140276</td>
      <td>11789</td>
      <td>NaN</td>
      <td>40</td>
      <td>2017-07-28</td>
      <td>2017-07-28</td>
      <td>1</td>
      <td>3</td>
      <td>M545</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>85.25</td>
      <td>NaN</td>
      <td>14</td>
      <td>1</td>
      <td>NaN</td>
      <td>C</td>
      <td>1376521575</td>
      <td>NaN</td>
      <td>1.912991e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>115.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>540</th>
      <td>100750</td>
      <td>230216</td>
      <td>12138</td>
      <td>NaN</td>
      <td>40</td>
      <td>2018-01-13</td>
      <td>2018-01-13</td>
      <td>1</td>
      <td>3</td>
      <td>Z0289</td>
      <td>NaN</td>
      <td>N</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>23</td>
      <td>9</td>
      <td>NaN</td>
      <td>C</td>
      <td>1982685384</td>
      <td>NaN</td>
      <td>1.063442e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>226.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>230</th>
      <td>1698691</td>
      <td>390145</td>
      <td>10007</td>
      <td>NaN</td>
      <td>40</td>
      <td>2016-12-11</td>
      <td>2016-12-11</td>
      <td>1</td>
      <td>3</td>
      <td>Z01818</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>43.01</td>
      <td>NaN</td>
      <td>39</td>
      <td>1</td>
      <td>NaN</td>
      <td>C</td>
      <td>1689691214</td>
      <td>NaN</td>
      <td>1.679505e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>235.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4365</th>
      <td>1698722</td>
      <td>200021</td>
      <td>10985</td>
      <td>NaN</td>
      <td>40</td>
      <td>2018-06-16</td>
      <td>2018-06-16</td>
      <td>1</td>
      <td>3</td>
      <td>E782</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>179.09</td>
      <td>NaN</td>
      <td>20</td>
      <td>1</td>
      <td>NaN</td>
      <td>C</td>
      <td>1932164795</td>
      <td>NaN</td>
      <td>1.548289e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1939.35</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4366</th>
      <td>1698935</td>
      <td>210022</td>
      <td>1297</td>
      <td>NaN</td>
      <td>40</td>
      <td>2018-04-06</td>
      <td>2018-04-06</td>
      <td>1</td>
      <td>3</td>
      <td>I110</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>400.81</td>
      <td>NaN</td>
      <td>21</td>
      <td>9</td>
      <td>NaN</td>
      <td>C</td>
      <td>1205896446</td>
      <td>NaN</td>
      <td>1.922016e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>554.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4367</th>
      <td>1699005</td>
      <td>100057</td>
      <td>12194</td>
      <td>NaN</td>
      <td>40</td>
      <td>2016-04-27</td>
      <td>2016-04-27</td>
      <td>1</td>
      <td>3</td>
      <td>I348</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>265.19</td>
      <td>NaN</td>
      <td>10</td>
      <td>1</td>
      <td>NaN</td>
      <td>C</td>
      <td>1821019571</td>
      <td>NaN</td>
      <td>1.437130e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>8423.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4368</th>
      <td>1699102</td>
      <td>330191</td>
      <td>11842</td>
      <td>NaN</td>
      <td>60</td>
      <td>2017-01-30</td>
      <td>2017-02-03</td>
      <td>1</td>
      <td>1</td>
      <td>I441</td>
      <td>R42</td>
      <td>NaN</td>
      <td>6476.96</td>
      <td>NaN</td>
      <td>33</td>
      <td>3</td>
      <td>287.0</td>
      <td>NaN</td>
      <td>1871606764</td>
      <td>NaN</td>
      <td>1.679594e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>17897.91</td>
      <td>467.56</td>
      <td>20.53</td>
      <td>39.19</td>
      <td>669.18</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>8626 rows × 37 columns</p>
</div>



#### Data Quality Check #1: If true, the original dataset was unique on claim ID


```python
parta_claims_header_raw_df_count = parta_claims_header_raw_df.shape[0]
parta_claims_header_raw_uniq_clm_id_df = parta_claims_header_raw_df['cur_clm_uniq_id'].drop_duplicates()

parta_claims_header_raw_df_count == parta_claims_header_raw_uniq_clm_id_df.shape[0]
```




    True




```python
# Select only the desired columns (renaming if needed) and remove duplicates if any
parta_claims_header_df = parta_claims_header_raw_df[[
    'cur_clm_uniq_id', 'bene_mbi_id', 'atndg_prvdr_npi_num', 
    'clm_from_dt', 'prncpl_dgns_cd', 'clm_pmt_amt'
]].drop_duplicates().rename(
    columns={
        'cur_clm_uniq_id': 'claim_id',
        'bene_mbi_id': 'patient_id',
        'clm_from_dt': 'claim_date',
        'atndg_prvdr_npi_num': 'npi_id'
    }
)
parta_claims_header_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>prncpl_dgns_cd</th>
      <th>clm_pmt_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>M25551</td>
      <td>259.01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1004555</td>
      <td>10133</td>
      <td>1.942275e+09</td>
      <td>2018-11-02</td>
      <td>Z9861</td>
      <td>29.56</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1011605</td>
      <td>10163</td>
      <td>1.578546e+09</td>
      <td>2018-01-02</td>
      <td>C439</td>
      <td>45.88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1011758</td>
      <td>1003</td>
      <td>1.952368e+09</td>
      <td>2018-06-12</td>
      <td>R310</td>
      <td>9.40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101424</td>
      <td>10052</td>
      <td>1.336125e+09</td>
      <td>2016-04-13</td>
      <td>L821</td>
      <td>34.18</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8621</th>
      <td>999774</td>
      <td>10367</td>
      <td>NaN</td>
      <td>2017-11-06</td>
      <td>R072</td>
      <td>374.08</td>
    </tr>
    <tr>
      <th>8622</th>
      <td>999808</td>
      <td>10496</td>
      <td>1.740225e+09</td>
      <td>2017-07-19</td>
      <td>R079</td>
      <td>360.89</td>
    </tr>
    <tr>
      <th>8623</th>
      <td>999878</td>
      <td>12160</td>
      <td>1.497784e+09</td>
      <td>2018-01-18</td>
      <td>R5383</td>
      <td>90.73</td>
    </tr>
    <tr>
      <th>8624</th>
      <td>999961</td>
      <td>12090</td>
      <td>1.083691e+09</td>
      <td>2018-03-10</td>
      <td>C73</td>
      <td>329.44</td>
    </tr>
    <tr>
      <th>8625</th>
      <td>999976</td>
      <td>10768</td>
      <td>1.770564e+09</td>
      <td>2016-01-18</td>
      <td>E785</td>
      <td>25.60</td>
    </tr>
  </tbody>
</table>
<p>8626 rows × 6 columns</p>
</div>



#### Data Quality Check #2: If true, the filtered dataset did not have any duplicates


```python
parta_claims_header_df_count = parta_claims_header_df.shape[0]

parta_claims_header_raw_df_count == parta_claims_header_df_count
```




    True



#### Data Quality Check #3: If the resulting dataframe is empty, it means all the records have diagnosis code (if it's not empty it should be removed now since we want only the ones with valid codes for analysis)


```python
parta_claims_header_df.loc[~parta_claims_header_df.prncpl_dgns_cd.notnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>prncpl_dgns_cd</th>
      <th>clm_pmt_amt</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### 0.2.2 Load & select columns from Claims Revenue Center dataset


```python
# Load Claims Revenue Center dataset
# Note this dataset has more than one record for each claim ID (cur_clm_uniq_id)
# Also note there are two sets of date columns, 
#  one for claim ID (clm_from/thru_dt) and one for claim line (clm_line_from/thru_dt)
parta_claims_revenue_center_detail_raw_df = pd.read_csv(
    "/Users/hamiddastgir/Library/CloudStorage/Dropbox/Semester 3/BIA 810 - Healthcare Analytics/Mid Term/Syntegra Datasets Files/parta_claims_revenue_center_detail.csv"
)
parta_claims_revenue_center_detail_raw_df.sort_values(by=['cur_clm_uniq_id'])
```

    /var/folders/zk/ffvbnsts2dg8tf_rqkmdm36m0000gn/T/ipykernel_63105/3269972768.py:5: DtypeWarning: Columns (19) have mixed types. Specify dtype option on import or set low_memory=False.
      parta_claims_revenue_center_detail_raw_df = pd.read_csv(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cur_clm_uniq_id</th>
      <th>clm_line_num</th>
      <th>bene_mbi_id</th>
      <th>bene_hic_num</th>
      <th>clm_type_cd</th>
      <th>clm_line_from_dt</th>
      <th>clm_line_thru_dt</th>
      <th>clm_line_prod_rev_ctr_cd</th>
      <th>clm_line_instnl_rev_ctr_dt</th>
      <th>clm_line_hcpcs_cd</th>
      <th>bene_eqtbl_bic_hicn_num</th>
      <th>prvdr_oscar_num</th>
      <th>clm_from_dt</th>
      <th>clm_thru_dt</th>
      <th>clm_line_srvc_unit_qty</th>
      <th>clm_line_cvrd_pd_amt</th>
      <th>hcpcs_1_mdfr_cd</th>
      <th>hcpcs_2_mdfr_cd</th>
      <th>hcpcs_3_mdfr_cd</th>
      <th>hcpcs_4_mdfr_cd</th>
      <th>hcpcs_5_mdfr_cd</th>
      <th>clm_rev_apc_hipps_cd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>318</th>
      <td>100073</td>
      <td>1</td>
      <td>12620</td>
      <td>NaN</td>
      <td>40</td>
      <td>2018-12-02 00:00:00</td>
      <td>2018-12-02 00:00:00</td>
      <td>403</td>
      <td>2018-12-02 00:00:00</td>
      <td>77063</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-12-02 00:00:00</td>
      <td>2018-12-02 00:00:00</td>
      <td>1</td>
      <td>24.11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>00000</td>
    </tr>
    <tr>
      <th>383</th>
      <td>100184</td>
      <td>1</td>
      <td>10080</td>
      <td>NaN</td>
      <td>40</td>
      <td>2018-09-06 00:00:00</td>
      <td>2018-09-06 00:00:00</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-09-06 00:00:00</td>
      <td>2018-09-06 00:00:00</td>
      <td>0</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>00000</td>
    </tr>
    <tr>
      <th>384</th>
      <td>100190</td>
      <td>1</td>
      <td>1228</td>
      <td>NaN</td>
      <td>40</td>
      <td>2018-06-10 00:00:00</td>
      <td>2018-06-10 00:00:00</td>
      <td>521</td>
      <td>2018-06-10 00:00:00</td>
      <td>G0467</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-06-10 00:00:00</td>
      <td>2018-06-10 00:00:00</td>
      <td>1</td>
      <td>133.74</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>00000</td>
    </tr>
    <tr>
      <th>385</th>
      <td>100190</td>
      <td>2</td>
      <td>1228</td>
      <td>NaN</td>
      <td>40</td>
      <td>2018-06-10 00:00:00</td>
      <td>2018-06-10 00:00:00</td>
      <td>521</td>
      <td>2018-06-10 00:00:00</td>
      <td>98960</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-06-10 00:00:00</td>
      <td>2018-06-10 00:00:00</td>
      <td>1</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>00000</td>
    </tr>
    <tr>
      <th>386</th>
      <td>100190</td>
      <td>3</td>
      <td>1228</td>
      <td>NaN</td>
      <td>40</td>
      <td>2018-06-10 00:00:00</td>
      <td>2018-06-10 00:00:00</td>
      <td>521</td>
      <td>2018-06-10 00:00:00</td>
      <td>J1100</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-06-10 00:00:00</td>
      <td>2018-06-10 00:00:00</td>
      <td>4</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>00000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29896</th>
      <td>1699197</td>
      <td>2</td>
      <td>1177</td>
      <td>NaN</td>
      <td>40</td>
      <td>2016-05-22 00:00:00</td>
      <td>2016-05-22 00:00:00</td>
      <td>302</td>
      <td>2016-05-22 00:00:00</td>
      <td>86592</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-05-22 00:00:00</td>
      <td>2016-05-22 00:00:00</td>
      <td>1</td>
      <td>5.43</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>00000</td>
    </tr>
    <tr>
      <th>29898</th>
      <td>1699212</td>
      <td>1</td>
      <td>1262</td>
      <td>NaN</td>
      <td>60</td>
      <td>2018-12-24 00:00:00</td>
      <td>2018-12-25 00:00:00</td>
      <td>730</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-12-24 00:00:00</td>
      <td>2018-12-25 00:00:00</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>00000</td>
    </tr>
    <tr>
      <th>29901</th>
      <td>1699236</td>
      <td>3</td>
      <td>10580</td>
      <td>NaN</td>
      <td>40</td>
      <td>2017-09-20 00:00:00</td>
      <td>2017-09-20 00:00:00</td>
      <td>370</td>
      <td>2017-09-20 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-09-20 00:00:00</td>
      <td>2017-09-20 00:00:00</td>
      <td>2</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>00000</td>
    </tr>
    <tr>
      <th>29899</th>
      <td>1699236</td>
      <td>1</td>
      <td>10580</td>
      <td>NaN</td>
      <td>40</td>
      <td>2017-09-20 00:00:00</td>
      <td>2017-09-20 00:00:00</td>
      <td>258</td>
      <td>2017-09-20 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-09-20 00:00:00</td>
      <td>2017-09-20 00:00:00</td>
      <td>1</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>00000</td>
    </tr>
    <tr>
      <th>29900</th>
      <td>1699236</td>
      <td>2</td>
      <td>10580</td>
      <td>NaN</td>
      <td>40</td>
      <td>2017-09-20 00:00:00</td>
      <td>2017-09-20 00:00:00</td>
      <td>360</td>
      <td>2017-09-20 00:00:00</td>
      <td>45385</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017-09-20 00:00:00</td>
      <td>2017-09-20 00:00:00</td>
      <td>1</td>
      <td>543.04</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>05312</td>
    </tr>
  </tbody>
</table>
<p>59419 rows × 22 columns</p>
</div>



#### Data Quality Check #4: If the resulting dataframe is empty, it means there is no difference between columns 'clm_line_from_dt' and 'clm_from_dt' for all the rows


```python
parta_claims_revenue_center_detail_raw_df.loc[
    ~(parta_claims_revenue_center_detail_raw_df['clm_line_from_dt'] 
      == parta_claims_revenue_center_detail_raw_df['clm_from_dt'])
]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cur_clm_uniq_id</th>
      <th>clm_line_num</th>
      <th>bene_mbi_id</th>
      <th>bene_hic_num</th>
      <th>clm_type_cd</th>
      <th>clm_line_from_dt</th>
      <th>clm_line_thru_dt</th>
      <th>clm_line_prod_rev_ctr_cd</th>
      <th>clm_line_instnl_rev_ctr_dt</th>
      <th>clm_line_hcpcs_cd</th>
      <th>bene_eqtbl_bic_hicn_num</th>
      <th>prvdr_oscar_num</th>
      <th>clm_from_dt</th>
      <th>clm_thru_dt</th>
      <th>clm_line_srvc_unit_qty</th>
      <th>clm_line_cvrd_pd_amt</th>
      <th>hcpcs_1_mdfr_cd</th>
      <th>hcpcs_2_mdfr_cd</th>
      <th>hcpcs_3_mdfr_cd</th>
      <th>hcpcs_4_mdfr_cd</th>
      <th>hcpcs_5_mdfr_cd</th>
      <th>clm_rev_apc_hipps_cd</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# Select only the desired columns (renaming if needed) and remove duplicates if any
# Select 'clm_from_dt' as the column for claim dates since we want uniqueness on claim ID, not claim line
parta_claims_revenue_center_detail_df = parta_claims_revenue_center_detail_raw_df[[
    'cur_clm_uniq_id', 'bene_mbi_id', 'clm_from_dt',
    'clm_line_hcpcs_cd', 'clm_line_cvrd_pd_amt'
]].drop_duplicates().rename(
    columns={
        'cur_clm_uniq_id': 'claim_id',
        'bene_mbi_id': 'patient_id',
        'clm_from_dt': 'claim_date',
        'clm_line_hcpcs_cd': 'hcpcs_code'
    }
)
parta_claims_revenue_center_detail_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>clm_line_cvrd_pd_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001122</td>
      <td>10081</td>
      <td>2018-05-30 00:00:00</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28 00:00:00</td>
      <td>G0283</td>
      <td>9.67</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28 00:00:00</td>
      <td>G8978</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28 00:00:00</td>
      <td>G8979</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28 00:00:00</td>
      <td>97110</td>
      <td>24.97</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>59414</th>
      <td>999961</td>
      <td>12090</td>
      <td>2018-03-10 00:00:00</td>
      <td>A9516</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>59415</th>
      <td>999961</td>
      <td>12090</td>
      <td>2018-03-10 00:00:00</td>
      <td>G8996</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>59416</th>
      <td>999976</td>
      <td>10768</td>
      <td>2016-01-18 00:00:00</td>
      <td>80053</td>
      <td>11.37</td>
    </tr>
    <tr>
      <th>59417</th>
      <td>999976</td>
      <td>10768</td>
      <td>2016-01-18 00:00:00</td>
      <td>80061</td>
      <td>12.83</td>
    </tr>
    <tr>
      <th>59418</th>
      <td>999976</td>
      <td>10768</td>
      <td>2016-01-18 00:00:00</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>46823 rows × 5 columns</p>
</div>



#### Data Quality Check #5: If the resulting dataframe is empty, it means all the records have HCPCS code (if it's not empty it should be removed now since we want only the ones with valid codes for analysis)


```python
parta_claims_revenue_center_detail_df.loc[
    ~parta_claims_revenue_center_detail_df.hcpcs_code.notnull()
]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>clm_line_cvrd_pd_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001122</td>
      <td>10081</td>
      <td>2018-05-30 00:00:00</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28 00:00:00</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1004555</td>
      <td>10133</td>
      <td>2018-11-02 00:00:00</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1004904</td>
      <td>10106</td>
      <td>2018-02-26 00:00:00</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>100974</td>
      <td>10042</td>
      <td>2017-02-20 00:00:00</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>59369</th>
      <td>999008</td>
      <td>12473</td>
      <td>2018-08-04 00:00:00</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>59402</th>
      <td>999774</td>
      <td>10367</td>
      <td>2017-11-06 00:00:00</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>59407</th>
      <td>999808</td>
      <td>10496</td>
      <td>2017-07-19 00:00:00</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>59410</th>
      <td>999943</td>
      <td>11021</td>
      <td>2016-11-20 00:00:00</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>59418</th>
      <td>999976</td>
      <td>10768</td>
      <td>2016-01-18 00:00:00</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>10799 rows × 5 columns</p>
</div>




```python
# Data Quality Check #5 failed, so remove rows with no HCPCS codes
parta_claims_revenue_center_detail_df = parta_claims_revenue_center_detail_df.loc[
    parta_claims_revenue_center_detail_df.hcpcs_code.notnull()
]
```


```python
# Update date format for claim dates to match that of Claims Header dataset for easy join
parta_claims_revenue_center_detail_df['claim_date'] = pd.to_datetime(
    parta_claims_revenue_center_detail_df['claim_date']
).dt.strftime('%Y-%m-%d')
parta_claims_revenue_center_detail_df
```

    /var/folders/zk/ffvbnsts2dg8tf_rqkmdm36m0000gn/T/ipykernel_63105/2671745514.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      parta_claims_revenue_center_detail_df['claim_date'] = pd.to_datetime(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>clm_line_cvrd_pd_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28</td>
      <td>G0283</td>
      <td>9.67</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28</td>
      <td>G8978</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28</td>
      <td>G8979</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28</td>
      <td>97110</td>
      <td>24.97</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28</td>
      <td>97140</td>
      <td>20.33</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>59413</th>
      <td>999961</td>
      <td>12090</td>
      <td>2018-03-10</td>
      <td>78014</td>
      <td>400.05</td>
    </tr>
    <tr>
      <th>59414</th>
      <td>999961</td>
      <td>12090</td>
      <td>2018-03-10</td>
      <td>A9516</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>59415</th>
      <td>999961</td>
      <td>12090</td>
      <td>2018-03-10</td>
      <td>G8996</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>59416</th>
      <td>999976</td>
      <td>10768</td>
      <td>2016-01-18</td>
      <td>80053</td>
      <td>11.37</td>
    </tr>
    <tr>
      <th>59417</th>
      <td>999976</td>
      <td>10768</td>
      <td>2016-01-18</td>
      <td>80061</td>
      <td>12.83</td>
    </tr>
  </tbody>
</table>
<p>36024 rows × 5 columns</p>
</div>



#### Mini-Analysis #1: Find whether there are matching claims between Claim Header and Claims Revenue Center datasets

                                                ***


```python
claims_header_unique_claims_df = parta_claims_header_df[[
    'claim_id'
]].drop_duplicates()

claims_header_unique_claims_df['header'] = 1

revenue_center_unique_claims_df = parta_claims_revenue_center_detail_df[[
    'claim_id'
]].drop_duplicates()

revenue_center_unique_claims_df['revenue'] = 1

joined_df1 = pd.merge(
    claims_header_unique_claims_df,
    revenue_center_unique_claims_df,
    on='claim_id', how = 'outer'
)
joined_df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>header</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001595</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1004555</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1011605</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1011758</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101424</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15721</th>
      <td>999074</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15722</th>
      <td>999324</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15723</th>
      <td>999350</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15724</th>
      <td>999514</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15725</th>
      <td>999943</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>15726 rows × 3 columns</p>
</div>




```python
print('# of unique claims in Claims Header dataset: ' 
      + str(claims_header_unique_claims_df.shape[0])
     )
print('# of unique claims in Claims Revenue Center dataset: ' 
      + str(revenue_center_unique_claims_df.shape[0])
     )
```

    # of unique claims in Claims Header dataset: 8626
    # of unique claims in Claims Revenue Center dataset: 13406



```python
print('# of unique claims in Claims Header and Claims Revenue Center datasets combined: ' 
      + str(joined_df1.shape[0])
     )
print('From combined list of unique claims - ')
print('# of unique claims in only Claims Header dataset: ' 
      + str(joined_df1.loc[(joined_df1.header == 1) & ~(joined_df1.revenue == 1)].shape[0])
     )
print('# of unique claims in only Claims Revenue Center dataset: ' 
      + str(joined_df1.loc[~(joined_df1.header == 1) & (joined_df1.revenue == 1)].shape[0])
     )
print('# of unique claims in both Claims Header AND Claims Revenue Center datasets: ' 
      + str(joined_df1.loc[(joined_df1.header == 1) & (joined_df1.revenue == 1)].shape[0])
     )
```

    # of unique claims in Claims Header and Claims Revenue Center datasets combined: 15726
    From combined list of unique claims - 
    # of unique claims in only Claims Header dataset: 2320
    # of unique claims in only Claims Revenue Center dataset: 7100
    # of unique claims in both Claims Header AND Claims Revenue Center datasets: 6306


Conclusion: There are quite a number of claims available in both datasets, so join them on claim ID as an outer join to get all possible claims without duplicates

                                                ***

### 0.2.3 Load & select columns from Diagnosis dataset


```python
# Load the Diagnosis dataset
# Note that 'clm_from_dt' has some records with null values, but we need claim dates for all claims
parta_diagnosis_code_raw_df = pd.read_csv("/Users/hamiddastgir/Library/CloudStorage/Dropbox/Semester 3/BIA 810 - Healthcare Analytics/Mid Term/Syntegra Datasets Files/parta_diagnosis_code.csv")
parta_diagnosis_code_raw_df.sort_values(by=['cur_clm_uniq_id', 'clm_val_sqnc_num'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cur_clm_uniq_id</th>
      <th>bene_mbi_id</th>
      <th>bene_hic_num</th>
      <th>clm_type_cd</th>
      <th>clm_prod_type_cd</th>
      <th>clm_val_sqnc_num</th>
      <th>clm_dgns_cd</th>
      <th>bene_eqtbl_bic_hicn_num</th>
      <th>prvdr_oscar_num</th>
      <th>clm_from_dt</th>
      <th>clm_thru_dt</th>
      <th>clm_poa_ind</th>
      <th>dgns_prcdr_icd_ind</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>244</th>
      <td>100190</td>
      <td>1228</td>
      <td>NaN</td>
      <td>40</td>
      <td>NaN</td>
      <td>1</td>
      <td>M1611</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-06-10 00:00:00</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>243</th>
      <td>100190</td>
      <td>1228</td>
      <td>NaN</td>
      <td>40</td>
      <td>NaN</td>
      <td>2</td>
      <td>M25572</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-06-10 00:00:00</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>246</th>
      <td>100190</td>
      <td>1228</td>
      <td>NaN</td>
      <td>40</td>
      <td>NaN</td>
      <td>3</td>
      <td>M25551</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-06-10 00:00:00</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>245</th>
      <td>100190</td>
      <td>1228</td>
      <td>NaN</td>
      <td>40</td>
      <td>NaN</td>
      <td>4</td>
      <td>M5136</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-06-10 00:00:00</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>366</th>
      <td>100402</td>
      <td>1261</td>
      <td>NaN</td>
      <td>60</td>
      <td>NaN</td>
      <td>11</td>
      <td>E119</td>
      <td>NaN</td>
      <td>100256.0</td>
      <td>2017-05-28 00:00:00</td>
      <td>2017-06-02 00:00:00</td>
      <td>Y</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16232</th>
      <td>1699102</td>
      <td>11842</td>
      <td>NaN</td>
      <td>60</td>
      <td>NaN</td>
      <td>13</td>
      <td>Z8673</td>
      <td>NaN</td>
      <td>330191.0</td>
      <td>2017-01-31 00:00:00</td>
      <td>2017-02-03 00:00:00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16236</th>
      <td>1699137</td>
      <td>10873</td>
      <td>NaN</td>
      <td>40</td>
      <td>NaN</td>
      <td>1</td>
      <td>N390</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-07-12 00:00:00</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16235</th>
      <td>1699137</td>
      <td>10873</td>
      <td>NaN</td>
      <td>40</td>
      <td>NaN</td>
      <td>2</td>
      <td>N390</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-07-12 00:00:00</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16237</th>
      <td>1699155</td>
      <td>11689</td>
      <td>NaN</td>
      <td>40</td>
      <td>NaN</td>
      <td>3</td>
      <td>K219</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-12-06 00:00:00</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16238</th>
      <td>1699155</td>
      <td>11689</td>
      <td>NaN</td>
      <td>40</td>
      <td>NaN</td>
      <td>4</td>
      <td>E039</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-12-06 00:00:00</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>32052 rows × 13 columns</p>
</div>




```python
# Select only the desired columns (renaming if needed) and remove duplicates if any
# Use 'clm_thru_dt' as claim date columns since 'clm_from_dt' has some nulls
parta_diagnosis_code_df = parta_diagnosis_code_raw_df[[
    'cur_clm_uniq_id', 'bene_mbi_id', 'clm_thru_dt', 'clm_dgns_cd'
]].drop_duplicates().rename(
    columns={
        'cur_clm_uniq_id': 'claim_id',
        'bene_mbi_id': 'patient_id',
        'clm_thru_dt': 'claim_date'
    }
)
parta_diagnosis_code_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>claim_date</th>
      <th>clm_dgns_cd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001122</td>
      <td>10081</td>
      <td>2018-05-30 00:00:00</td>
      <td>K5289</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28 00:00:00</td>
      <td>M25551</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28 00:00:00</td>
      <td>M79604</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1001865</td>
      <td>10133</td>
      <td>2018-09-14 00:00:00</td>
      <td>G459</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004555</td>
      <td>10133</td>
      <td>2018-11-02 00:00:00</td>
      <td>Z9861</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32047</th>
      <td>999878</td>
      <td>12160</td>
      <td>2018-01-18 00:00:00</td>
      <td>N390</td>
    </tr>
    <tr>
      <th>32048</th>
      <td>999943</td>
      <td>11021</td>
      <td>2016-11-20 00:00:00</td>
      <td>M545</td>
    </tr>
    <tr>
      <th>32049</th>
      <td>999961</td>
      <td>12090</td>
      <td>2018-03-10 00:00:00</td>
      <td>C73</td>
    </tr>
    <tr>
      <th>32050</th>
      <td>999962</td>
      <td>11030</td>
      <td>2018-07-17 00:00:00</td>
      <td>G8194</td>
    </tr>
    <tr>
      <th>32051</th>
      <td>999976</td>
      <td>10768</td>
      <td>2016-01-18 00:00:00</td>
      <td>E785</td>
    </tr>
  </tbody>
</table>
<p>30487 rows × 4 columns</p>
</div>



#### Data Quality Check #6: If the resulting dataframe is empty, it means all the records have values for 'clm_thru_dt' (if it's not empty it should be removed now since without claim dates it'd be difficult to use)


```python
parta_diagnosis_code_df.loc[~parta_diagnosis_code_df.claim_date.notnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>claim_date</th>
      <th>clm_dgns_cd</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



#### Data Quality Check #7: If the resulting dataframe is empty, it means all the records have diagnosis code (if it's not empty it should be removed now since we want only the ones with valid codes for analysis)


```python
parta_diagnosis_code_df.loc[~parta_diagnosis_code_df.clm_dgns_cd.notnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>claim_date</th>
      <th>clm_dgns_cd</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# Update date format for claim dates to match that of Claims Header dataset for easy join
parta_diagnosis_code_df['claim_date'] = pd.to_datetime(
    parta_diagnosis_code_df['claim_date']
).dt.strftime('%Y-%m-%d')
parta_diagnosis_code_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>claim_date</th>
      <th>clm_dgns_cd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001122</td>
      <td>10081</td>
      <td>2018-05-30</td>
      <td>K5289</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28</td>
      <td>M25551</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001595</td>
      <td>10226</td>
      <td>2018-02-28</td>
      <td>M79604</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1001865</td>
      <td>10133</td>
      <td>2018-09-14</td>
      <td>G459</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004555</td>
      <td>10133</td>
      <td>2018-11-02</td>
      <td>Z9861</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32047</th>
      <td>999878</td>
      <td>12160</td>
      <td>2018-01-18</td>
      <td>N390</td>
    </tr>
    <tr>
      <th>32048</th>
      <td>999943</td>
      <td>11021</td>
      <td>2016-11-20</td>
      <td>M545</td>
    </tr>
    <tr>
      <th>32049</th>
      <td>999961</td>
      <td>12090</td>
      <td>2018-03-10</td>
      <td>C73</td>
    </tr>
    <tr>
      <th>32050</th>
      <td>999962</td>
      <td>11030</td>
      <td>2018-07-17</td>
      <td>G8194</td>
    </tr>
    <tr>
      <th>32051</th>
      <td>999976</td>
      <td>10768</td>
      <td>2016-01-18</td>
      <td>E785</td>
    </tr>
  </tbody>
</table>
<p>30487 rows × 4 columns</p>
</div>



#### Mini-Analysis #2: Find whether there are matching claims between above two datasets and the Diagnosis dataset

                                                ***


```python
diagnosis_unique_claims_df = parta_diagnosis_code_df[[
    'claim_id'
]].drop_duplicates()

diagnosis_unique_claims_df['diagnosis'] = 1

joined_df2 = pd.merge(
    joined_df1,
    diagnosis_unique_claims_df,
    on='claim_id', how = 'outer'
)
joined_df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>header</th>
      <th>revenue</th>
      <th>diagnosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001595</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1004555</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1011605</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1011758</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101424</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19452</th>
      <td>998726</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19453</th>
      <td>999064</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19454</th>
      <td>999766</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19455</th>
      <td>999799</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19456</th>
      <td>999962</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>19457 rows × 4 columns</p>
</div>




```python
print('# of unique claims in Claims Header+Claims Revenue Center datasets: ' 
      + str(joined_df1.shape[0])
     )
print('# of unique claims in Diagnosis dataset: ' 
      + str(diagnosis_unique_claims_df.shape[0])
     )
```

    # of unique claims in Claims Header+Claims Revenue Center datasets: 15726
    # of unique claims in Diagnosis dataset: 13432



```python
print('# of unique claims in Claims Header+Claims Revenue Center and Diagnosis datasets combined: ' 
      + str(joined_df2.shape[0])
     )
print('From combined list of unique claims - ')
print('# of unique claims only in either Claims Header or Claims Revenue Center datasets: ' 
      + str(joined_df2.loc[
          ((joined_df2.header == 1) | (joined_df2.revenue == 1))
          & ~(joined_df2.diagnosis == 1)
      ].shape[0])
     )
print('# of unique claims in only Diagnosis dataset: ' 
      + str(joined_df2.loc[
          (~(joined_df2.header == 1) & ~(joined_df2.revenue == 1))
          & (joined_df2.diagnosis == 1)
      ].shape[0])
     )
print('# of unique claims in all three datasets: ' 
      + str(joined_df2.loc[
          (joined_df2.header == 1) & (joined_df2.revenue == 1) & (joined_df2.diagnosis == 1)
      ].shape[0])
     )
```

    # of unique claims in Claims Header+Claims Revenue Center and Diagnosis datasets combined: 19457
    From combined list of unique claims - 
    # of unique claims only in either Claims Header or Claims Revenue Center datasets: 6025
    # of unique claims in only Diagnosis dataset: 3731
    # of unique claims in all three datasets: 5266


Conclusion: There are quite a number of claims available in all three datasets, so join diagnosis to the first two datasets on claim ID as an outer join to get all possible claims without duplicates

                                                ***

### 0.2.4 Load & select columns from Procedure dataset


```python
# Load the Procedure dataset
parta_procedure_code_df = pd.read_csv("/Users/hamiddastgir/Library/CloudStorage/Dropbox/Semester 3/BIA 810 - Healthcare Analytics/Mid Term/Syntegra Datasets Files/parta_procedure_code.csv")
parta_procedure_code_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cur_clm_uniq_id</th>
      <th>bene_mbi_id</th>
      <th>bene_hic_num</th>
      <th>clm_type_cd</th>
      <th>clm_val_sqnc_num</th>
      <th>clm_prcdr_cd</th>
      <th>clm_prcdr_prfrm_dt</th>
      <th>bene_eqtbl_bic_hicn_num</th>
      <th>prvdr_oscar_num</th>
      <th>clm_from_dt</th>
      <th>clm_thru_dt</th>
      <th>dgns_prcdr_icd_ind</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100402</td>
      <td>1261</td>
      <td>NaN</td>
      <td>60</td>
      <td>1</td>
      <td>0DJD8ZZ</td>
      <td>2017-05-31 00:00:00</td>
      <td>NaN</td>
      <td>100256</td>
      <td>2017-05-28 00:00:00</td>
      <td>2017-06-02 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100402</td>
      <td>1261</td>
      <td>NaN</td>
      <td>60</td>
      <td>2</td>
      <td>0D9670Z</td>
      <td>2017-05-29 00:00:00</td>
      <td>NaN</td>
      <td>100256</td>
      <td>2017-05-28 00:00:00</td>
      <td>2017-06-02 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100402</td>
      <td>1261</td>
      <td>NaN</td>
      <td>60</td>
      <td>3</td>
      <td>0DJD8ZZ</td>
      <td>2017-06-01 00:00:00</td>
      <td>NaN</td>
      <td>100256</td>
      <td>2017-05-28 00:00:00</td>
      <td>2017-06-02 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100402</td>
      <td>1261</td>
      <td>NaN</td>
      <td>60</td>
      <td>4</td>
      <td>0DB78ZX</td>
      <td>2017-05-30 00:00:00</td>
      <td>NaN</td>
      <td>100256</td>
      <td>2017-05-28 00:00:00</td>
      <td>2017-06-02 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1008371</td>
      <td>1074</td>
      <td>NaN</td>
      <td>60</td>
      <td>1</td>
      <td>0T9B7ZZ</td>
      <td>2016-12-03 00:00:00</td>
      <td>NaN</td>
      <td>140007</td>
      <td>2016-12-02 00:00:00</td>
      <td>2016-12-08 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>457</th>
      <td>357821</td>
      <td>10200</td>
      <td>NaN</td>
      <td>60</td>
      <td>2</td>
      <td>4A023N7</td>
      <td>2018-06-18 00:00:00</td>
      <td>NaN</td>
      <td>100258</td>
      <td>2018-06-16 00:00:00</td>
      <td>2018-06-19 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>458</th>
      <td>357821</td>
      <td>10200</td>
      <td>NaN</td>
      <td>60</td>
      <td>1</td>
      <td>4A023N7</td>
      <td>2018-06-18 00:00:00</td>
      <td>NaN</td>
      <td>100258</td>
      <td>2018-06-16 00:00:00</td>
      <td>2018-06-19 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>459</th>
      <td>412998</td>
      <td>10106</td>
      <td>NaN</td>
      <td>60</td>
      <td>1</td>
      <td>0SRC0J9</td>
      <td>2016-12-09 00:00:00</td>
      <td>NaN</td>
      <td>250104</td>
      <td>2016-12-09 00:00:00</td>
      <td>2016-12-10 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>460</th>
      <td>460114</td>
      <td>10133</td>
      <td>NaN</td>
      <td>60</td>
      <td>1</td>
      <td>0QSH04Z</td>
      <td>2018-05-17 00:00:00</td>
      <td>NaN</td>
      <td>150112</td>
      <td>2018-05-07 00:00:00</td>
      <td>2018-05-23 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>461</th>
      <td>766818</td>
      <td>10010</td>
      <td>NaN</td>
      <td>60</td>
      <td>1</td>
      <td>B246ZZZ</td>
      <td>2016-01-12 00:00:00</td>
      <td>NaN</td>
      <td>190263</td>
      <td>2016-01-09 00:00:00</td>
      <td>2016-01-15 00:00:00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>462 rows × 12 columns</p>
</div>



Conclusion: Don't join procedure dataset since the only useful info for sake of this analysis is the procedure codes and we won't be using them in our analysis

### 0.2.5 Load & select columns from DME dataset


```python
# Load the DME dataset
partb_dme_raw_df = pd.read_csv("/Users/hamiddastgir/Library/CloudStorage/Dropbox/Semester 3/BIA 810 - Healthcare Analytics/Mid Term/Syntegra Datasets Files/partb_dme.csv")
partb_dme_raw_df.sort_values(by='cur_clm_uniq_id')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cur_clm_uniq_id</th>
      <th>clm_line_num</th>
      <th>bene_mbi_id</th>
      <th>bene_hic_num</th>
      <th>clm_type_cd</th>
      <th>clm_from_dt</th>
      <th>clm_thru_dt</th>
      <th>clm_fed_type_srvc_cd</th>
      <th>clm_pos_cd</th>
      <th>clm_line_from_dt</th>
      <th>clm_line_thru_dt</th>
      <th>clm_line_hcpcs_cd</th>
      <th>clm_line_cvrd_pd_amt</th>
      <th>clm_prmry_pyr_cd</th>
      <th>payto_prvdr_npi_num</th>
      <th>ordrg_prvdr_npi_num</th>
      <th>clm_carr_pmt_dnl_cd</th>
      <th>clm_prcsg_ind_cd</th>
      <th>clm_adjsmt_type_cd</th>
      <th>clm_efctv_dt</th>
      <th>clm_idr_ld_dt</th>
      <th>clm_cntl_num</th>
      <th>bene_eqtbl_bic_hicn_num</th>
      <th>clm_line_alowd_chrg_amt</th>
      <th>clm_disp_cd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>267</th>
      <td>100441</td>
      <td>1</td>
      <td>12064</td>
      <td>NaN</td>
      <td>82</td>
      <td>2016-10-10</td>
      <td>2016-10-10</td>
      <td>P</td>
      <td>12</td>
      <td>2016-10-10</td>
      <td>2016-10-10</td>
      <td>A4256</td>
      <td>3.24</td>
      <td>NaN</td>
      <td>1972744431</td>
      <td>1.407814e+09</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.98</td>
      <td>1</td>
    </tr>
    <tr>
      <th>268</th>
      <td>100441</td>
      <td>2</td>
      <td>12064</td>
      <td>NaN</td>
      <td>82</td>
      <td>2016-10-10</td>
      <td>2016-10-10</td>
      <td>P</td>
      <td>12</td>
      <td>2016-10-10</td>
      <td>2016-10-10</td>
      <td>E0607</td>
      <td>59.80</td>
      <td>NaN</td>
      <td>1972744431</td>
      <td>1.407814e+09</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>76.54</td>
      <td>1</td>
    </tr>
    <tr>
      <th>269</th>
      <td>100441</td>
      <td>3</td>
      <td>12064</td>
      <td>NaN</td>
      <td>82</td>
      <td>2016-10-10</td>
      <td>2016-10-10</td>
      <td>P</td>
      <td>12</td>
      <td>2016-10-10</td>
      <td>2016-10-10</td>
      <td>A4253</td>
      <td>38.57</td>
      <td>NaN</td>
      <td>1972744431</td>
      <td>1.407814e+09</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>49.92</td>
      <td>1</td>
    </tr>
    <tr>
      <th>270</th>
      <td>100441</td>
      <td>4</td>
      <td>12064</td>
      <td>NaN</td>
      <td>82</td>
      <td>2016-10-10</td>
      <td>2016-10-10</td>
      <td>P</td>
      <td>12</td>
      <td>2016-10-10</td>
      <td>2016-10-10</td>
      <td>A4259</td>
      <td>4.20</td>
      <td>NaN</td>
      <td>1972744431</td>
      <td>1.407814e+09</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>271</th>
      <td>100441</td>
      <td>5</td>
      <td>12064</td>
      <td>NaN</td>
      <td>82</td>
      <td>2016-10-10</td>
      <td>2016-10-10</td>
      <td>P</td>
      <td>12</td>
      <td>2016-10-10</td>
      <td>2016-10-10</td>
      <td>A4258</td>
      <td>2.27</td>
      <td>NaN</td>
      <td>1972744431</td>
      <td>1.407814e+09</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.84</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1541</th>
      <td>1696080</td>
      <td>2</td>
      <td>11689</td>
      <td>NaN</td>
      <td>82</td>
      <td>2016-11-18</td>
      <td>2016-11-18</td>
      <td>P</td>
      <td>12</td>
      <td>2016-11-18</td>
      <td>2016-11-18</td>
      <td>A7038</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>1376599084</td>
      <td>1.659342e+09</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>129</th>
      <td>1696545</td>
      <td>1</td>
      <td>10046</td>
      <td>NaN</td>
      <td>82</td>
      <td>2017-07-25</td>
      <td>2017-07-25</td>
      <td>R</td>
      <td>12</td>
      <td>2017-07-25</td>
      <td>2017-07-25</td>
      <td>E0570</td>
      <td>6.30</td>
      <td>NaN</td>
      <td>1346347374</td>
      <td>1.952397e+09</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1548</th>
      <td>1696792</td>
      <td>1</td>
      <td>12086</td>
      <td>NaN</td>
      <td>82</td>
      <td>2018-12-23</td>
      <td>2018-12-23</td>
      <td>P</td>
      <td>12</td>
      <td>2018-12-23</td>
      <td>2018-12-23</td>
      <td>A4604</td>
      <td>35.58</td>
      <td>NaN</td>
      <td>1790823722</td>
      <td>1.518936e+09</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>48.17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1549</th>
      <td>1697987</td>
      <td>1</td>
      <td>11074</td>
      <td>NaN</td>
      <td>82</td>
      <td>2018-04-06</td>
      <td>2018-04-06</td>
      <td>P</td>
      <td>12</td>
      <td>2018-04-06</td>
      <td>2018-04-06</td>
      <td>A4253</td>
      <td>27.92</td>
      <td>NaN</td>
      <td>1902842065</td>
      <td>1.750382e+09</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.28</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1550</th>
      <td>1698182</td>
      <td>1</td>
      <td>12549</td>
      <td>NaN</td>
      <td>82</td>
      <td>2016-02-19</td>
      <td>2016-02-19</td>
      <td>R</td>
      <td>12</td>
      <td>2016-02-19</td>
      <td>2016-02-19</td>
      <td>E0570</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>1356586747</td>
      <td>1.336253e+09</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.49</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2775 rows × 25 columns</p>
</div>




```python
# Select only the desired columns (rename columns if needed) and remove duplicates if any
partb_dme_df = partb_dme_raw_df[[
    'cur_clm_uniq_id', 'bene_mbi_id', 'ordrg_prvdr_npi_num',
    'clm_from_dt', 'clm_line_hcpcs_cd', 'clm_line_alowd_chrg_amt'
]].drop_duplicates().rename(
    columns={
        'cur_clm_uniq_id': 'claim_id',
        'bene_mbi_id': 'patient_id',
        'ordrg_prvdr_npi_num': 'npi_id',
        'clm_from_dt': 'claim_date',
        'clm_line_hcpcs_cd': 'hcpcs_code',
        'clm_line_alowd_chrg_amt': 'claim_cost'
    }
)
partb_dme_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>claim_cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1004024</td>
      <td>10202</td>
      <td>1.841430e+09</td>
      <td>2016-07-18</td>
      <td>E0601</td>
      <td>41.91</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1034063</td>
      <td>10137</td>
      <td>1.669460e+09</td>
      <td>2016-04-22</td>
      <td>E0601</td>
      <td>62.46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1046877</td>
      <td>10202</td>
      <td>1.093713e+09</td>
      <td>2016-02-03</td>
      <td>E0601</td>
      <td>29.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1072934</td>
      <td>10202</td>
      <td>1.285602e+09</td>
      <td>2016-08-15</td>
      <td>E0601</td>
      <td>27.82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1082554</td>
      <td>10174</td>
      <td>1.003895e+09</td>
      <td>2016-08-30</td>
      <td>E0431</td>
      <td>18.75</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2770</th>
      <td>998097</td>
      <td>10396</td>
      <td>1.891706e+09</td>
      <td>2016-12-06</td>
      <td>A4253</td>
      <td>69.69</td>
    </tr>
    <tr>
      <th>2771</th>
      <td>999226</td>
      <td>1095</td>
      <td>1.518066e+09</td>
      <td>2017-12-28</td>
      <td>A4256</td>
      <td>3.68</td>
    </tr>
    <tr>
      <th>2772</th>
      <td>999226</td>
      <td>1095</td>
      <td>1.518066e+09</td>
      <td>2017-12-28</td>
      <td>A4253</td>
      <td>49.92</td>
    </tr>
    <tr>
      <th>2773</th>
      <td>999226</td>
      <td>1095</td>
      <td>1.518066e+09</td>
      <td>2017-12-28</td>
      <td>A4259</td>
      <td>4.26</td>
    </tr>
    <tr>
      <th>2774</th>
      <td>999929</td>
      <td>10261</td>
      <td>1.497738e+09</td>
      <td>2018-06-12</td>
      <td>E0570</td>
      <td>3.00</td>
    </tr>
  </tbody>
</table>
<p>2731 rows × 6 columns</p>
</div>



#### Data Quality Check #8: If the resulting dataframe is empty, it means all the records have HCPCS code (if it's not empty it should be removed now since we want only the ones with valid codes for analysis)


```python
partb_dme_df.loc[~partb_dme_df.hcpcs_code.notnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>claim_cost</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



#### Mini-Analysis #3: Find whether there are matching claims between the first three datasets above and the DME dataset

                                                ***


```python
dme_unique_claims_df = partb_dme_df[[
    'claim_id'
]].drop_duplicates()

dme_unique_claims_df['dme'] = 1

joined_df3 = pd.merge(
    joined_df2,
    dme_unique_claims_df,
    on='claim_id', how = 'outer'
)
joined_df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>header</th>
      <th>revenue</th>
      <th>diagnosis</th>
      <th>dme</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001595</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1004555</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1011605</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1011758</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101424</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20960</th>
      <td>994844</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20961</th>
      <td>994885</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20962</th>
      <td>998097</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20963</th>
      <td>999226</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20964</th>
      <td>999929</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>20965 rows × 5 columns</p>
</div>




```python
print('# of unique claims in first three datasets: ' 
      + str(joined_df2.shape[0])
     )
print('# of unique claims in DME dataset: ' 
      + str(dme_unique_claims_df.shape[0])
     )
```

    # of unique claims in first three datasets: 19457
    # of unique claims in DME dataset: 1508



```python
print('# of unique claims in the four datasets combined: ' 
      + str(joined_df3.shape[0])
     )
print('From combined list of unique claims - ')
print('# of unique claims in only the first three datasets: ' 
      + str(joined_df3.loc[
          ((joined_df3.header == 1) 
          | (joined_df3.revenue == 1)
          | (joined_df3.diagnosis == 1))
          & ~(joined_df3.dme == 1)
      ].shape[0])
     )
print('# of unique claims in only DME dataset: ' 
      + str(joined_df3.loc[
          ~(joined_df3.header == 1) 
          & ~(joined_df3.revenue == 1)
          & ~(joined_df3.diagnosis == 1)
          & (joined_df3.dme == 1)
      ].shape[0])
     )
print('# of unique claims in all four datasets: ' 
      + str(joined_df3.loc[
          (joined_df3.header == 1) 
          & (joined_df3.revenue == 1)
          & (joined_df3.diagnosis == 1)
          & (joined_df3.dme == 1)
      ].shape[0])
     )
print('# of unique claims in DME and any of the first three datasets: ' 
      + str(joined_df3.loc[
          ((joined_df3.header == 1) 
          | (joined_df3.revenue == 1)
          | (joined_df3.diagnosis == 1))
          & (joined_df3.dme == 1)
      ].shape[0])
     )
```

    # of unique claims in the four datasets combined: 20965
    From combined list of unique claims - 
    # of unique claims in only the first three datasets: 19457
    # of unique claims in only DME dataset: 1508
    # of unique claims in all four datasets: 0
    # of unique claims in DME and any of the first three datasets: 0


Conclusion: None of the claims from DME is in any of the first three datasets, so append them to the output after joining the first three datasets

                                                ***

### 0.2.6 Load & select columns from Physicians dataset


```python
# Load the Physicians dataset
partb_physicians_raw_df = pd.read_csv("/Users/hamiddastgir/Library/CloudStorage/Dropbox/Semester 3/BIA 810 - Healthcare Analytics/Mid Term/Syntegra Datasets Files/partb_physicians.csv")
partb_physicians_raw_df.sort_values(by='cur_clm_uniq_id')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cur_clm_uniq_id</th>
      <th>clm_line_num</th>
      <th>bene_mbi_id</th>
      <th>bene_hic_num</th>
      <th>clm_type_cd</th>
      <th>clm_from_dt</th>
      <th>clm_thru_dt</th>
      <th>rndrg_prvdr_type_cd</th>
      <th>rndrg_prvdr_fips_st_cd</th>
      <th>clm_prvdr_spclty_cd</th>
      <th>clm_fed_type_srvc_cd</th>
      <th>clm_pos_cd</th>
      <th>clm_line_from_dt</th>
      <th>clm_line_thru_dt</th>
      <th>clm_line_hcpcs_cd</th>
      <th>clm_line_cvrd_pd_amt</th>
      <th>clm_line_prmry_pyr_cd</th>
      <th>clm_line_dgns_cd</th>
      <th>clm_rndrg_prvdr_tax_num</th>
      <th>rndrg_prvdr_npi_num</th>
      <th>clm_carr_pmt_dnl_cd</th>
      <th>clm_prcsg_ind_cd</th>
      <th>clm_adjsmt_type_cd</th>
      <th>clm_efctv_dt</th>
      <th>clm_idr_ld_dt</th>
      <th>clm_cntl_num</th>
      <th>bene_eqtbl_bic_hicn_num</th>
      <th>clm_line_alowd_chrg_amt</th>
      <th>clm_line_srvc_unit_qty</th>
      <th>hcpcs_1_mdfr_cd</th>
      <th>hcpcs_2_mdfr_cd</th>
      <th>hcpcs_3_mdfr_cd</th>
      <th>hcpcs_4_mdfr_cd</th>
      <th>hcpcs_5_mdfr_cd</th>
      <th>clm_disp_cd</th>
      <th>clm_dgns_1_cd</th>
      <th>clm_dgns_2_cd</th>
      <th>clm_dgns_3_cd</th>
      <th>clm_dgns_4_cd</th>
      <th>clm_dgns_5_cd</th>
      <th>clm_dgns_6_cd</th>
      <th>clm_dgns_7_cd</th>
      <th>clm_dgns_8_cd</th>
      <th>dgns_prcdr_icd_ind</th>
      <th>clm_dgns_9_cd</th>
      <th>clm_dgns_10_cd</th>
      <th>clm_dgns_11_cd</th>
      <th>clm_dgns_12_cd</th>
      <th>hcpcs_betos_cd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>520</th>
      <td>100020</td>
      <td>1</td>
      <td>1070</td>
      <td>NaN</td>
      <td>71</td>
      <td>2016-10-04</td>
      <td>2016-10-04</td>
      <td>5</td>
      <td>36</td>
      <td>69</td>
      <td>5</td>
      <td>81</td>
      <td>2016-10-04</td>
      <td>2016-10-04</td>
      <td>85610</td>
      <td>5.10</td>
      <td>NaN</td>
      <td>I482</td>
      <td>NaN</td>
      <td>1.619972e+09</td>
      <td>1</td>
      <td>A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.49</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>I482</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>T1H</td>
    </tr>
    <tr>
      <th>525</th>
      <td>100024</td>
      <td>1</td>
      <td>11654</td>
      <td>NaN</td>
      <td>71</td>
      <td>2016-12-10</td>
      <td>2016-12-10</td>
      <td>1</td>
      <td>39</td>
      <td>26</td>
      <td>T</td>
      <td>11</td>
      <td>2016-12-10</td>
      <td>2016-12-10</td>
      <td>90834</td>
      <td>61.17</td>
      <td>NaN</td>
      <td>F319</td>
      <td>NaN</td>
      <td>1.811965e+09</td>
      <td>1</td>
      <td>A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>79.36</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>F319</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M5B</td>
    </tr>
    <tr>
      <th>529</th>
      <td>100030</td>
      <td>1</td>
      <td>12052</td>
      <td>NaN</td>
      <td>71</td>
      <td>2017-04-15</td>
      <td>2017-04-15</td>
      <td>1</td>
      <td>5</td>
      <td>06</td>
      <td>5</td>
      <td>21</td>
      <td>2017-04-15</td>
      <td>2017-04-15</td>
      <td>93010</td>
      <td>6.92</td>
      <td>NaN</td>
      <td>R001</td>
      <td>NaN</td>
      <td>1.336344e+09</td>
      <td>1</td>
      <td>A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.53</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>R001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>T2A</td>
    </tr>
    <tr>
      <th>555</th>
      <td>100038</td>
      <td>1</td>
      <td>12345</td>
      <td>NaN</td>
      <td>71</td>
      <td>2018-07-02</td>
      <td>2018-07-02</td>
      <td>1</td>
      <td>34</td>
      <td>30</td>
      <td>4</td>
      <td>19</td>
      <td>2018-07-02</td>
      <td>2018-07-02</td>
      <td>72158</td>
      <td>89.30</td>
      <td>NaN</td>
      <td>M47816</td>
      <td>NaN</td>
      <td>1.295730e+09</td>
      <td>1</td>
      <td>A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>112.57</td>
      <td>1.0</td>
      <td>26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>M47816</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>I2D</td>
    </tr>
    <tr>
      <th>592</th>
      <td>100061</td>
      <td>1</td>
      <td>10252</td>
      <td>NaN</td>
      <td>71</td>
      <td>2016-07-04</td>
      <td>2016-07-04</td>
      <td>1</td>
      <td>33</td>
      <td>48</td>
      <td>1</td>
      <td>11</td>
      <td>2016-07-04</td>
      <td>2016-07-04</td>
      <td>99213</td>
      <td>65.83</td>
      <td>NaN</td>
      <td>L03032</td>
      <td>NaN</td>
      <td>1.861493e+09</td>
      <td>1</td>
      <td>A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>82.36</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>L03032</td>
      <td>B351</td>
      <td>L853</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M1B</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5485</th>
      <td>1699176</td>
      <td>1</td>
      <td>1008</td>
      <td>NaN</td>
      <td>71</td>
      <td>2018-10-18</td>
      <td>2018-10-18</td>
      <td>1</td>
      <td>18</td>
      <td>29</td>
      <td>1</td>
      <td>21</td>
      <td>2018-10-18</td>
      <td>2018-10-18</td>
      <td>99232</td>
      <td>56.62</td>
      <td>NaN</td>
      <td>J9601</td>
      <td>NaN</td>
      <td>1.730182e+09</td>
      <td>1</td>
      <td>A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>73.06</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>J9601</td>
      <td>J810</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M2B</td>
    </tr>
    <tr>
      <th>66051</th>
      <td>1699182</td>
      <td>1</td>
      <td>13175</td>
      <td>NaN</td>
      <td>71</td>
      <td>2016-11-21</td>
      <td>2016-11-21</td>
      <td>5</td>
      <td>31</td>
      <td>69</td>
      <td>5</td>
      <td>81</td>
      <td>2016-11-21</td>
      <td>2016-11-21</td>
      <td>80053</td>
      <td>7.43</td>
      <td>NaN</td>
      <td>E782</td>
      <td>NaN</td>
      <td>1.063497e+09</td>
      <td>1</td>
      <td>A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.87</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>E782</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>T1B</td>
    </tr>
    <tr>
      <th>66052</th>
      <td>1699186</td>
      <td>1</td>
      <td>10710</td>
      <td>NaN</td>
      <td>71</td>
      <td>2016-01-18</td>
      <td>2016-01-18</td>
      <td>1</td>
      <td>14</td>
      <td>30</td>
      <td>4</td>
      <td>23</td>
      <td>2016-01-18</td>
      <td>2016-01-18</td>
      <td>73110</td>
      <td>7.41</td>
      <td>NaN</td>
      <td>S52502A</td>
      <td>NaN</td>
      <td>1.427027e+09</td>
      <td>1</td>
      <td>A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.97</td>
      <td>1.0</td>
      <td>26</td>
      <td>LT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>S52502A</td>
      <td>S52602A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>I1B</td>
    </tr>
    <tr>
      <th>66053</th>
      <td>1699204</td>
      <td>1</td>
      <td>11540</td>
      <td>NaN</td>
      <td>71</td>
      <td>2018-05-08</td>
      <td>2018-05-08</td>
      <td>1</td>
      <td>28</td>
      <td>13</td>
      <td>1</td>
      <td>11</td>
      <td>2018-05-08</td>
      <td>2018-05-08</td>
      <td>99214</td>
      <td>80.42</td>
      <td>NaN</td>
      <td>M5116</td>
      <td>NaN</td>
      <td>1.275519e+09</td>
      <td>1</td>
      <td>A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>101.91</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>M5116</td>
      <td>M47816</td>
      <td>M48061</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>M1B</td>
    </tr>
    <tr>
      <th>66054</th>
      <td>1699222</td>
      <td>1</td>
      <td>11556</td>
      <td>NaN</td>
      <td>71</td>
      <td>2016-03-16</td>
      <td>2016-03-16</td>
      <td>1</td>
      <td>33</td>
      <td>94</td>
      <td>1</td>
      <td>11</td>
      <td>2016-03-16</td>
      <td>2016-03-16</td>
      <td>J7060</td>
      <td>0.00</td>
      <td>G</td>
      <td>I872</td>
      <td>NaN</td>
      <td>1.932188e+09</td>
      <td>1</td>
      <td>S</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.66</td>
      <td>1.0</td>
      <td>RT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>I872</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>O1E</td>
    </tr>
  </tbody>
</table>
<p>130699 rows × 49 columns</p>
</div>




```python
#Possible expansio of analysis - keep as side note

partb_physicians_raw_df.groupby('clm_pos_cd').agg(
    uniq_clm_cnt=('cur_clm_uniq_id', 'nunique')
).sort_values(by='uniq_clm_cnt', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uniq_clm_cnt</th>
    </tr>
    <tr>
      <th>clm_pos_cd</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>34209</td>
    </tr>
    <tr>
      <th>81</th>
      <td>17512</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7218</td>
    </tr>
    <tr>
      <th>21</th>
      <td>6171</td>
    </tr>
    <tr>
      <th>23</th>
      <td>4505</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1912</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1619</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1435</td>
    </tr>
    <tr>
      <th>31</th>
      <td>891</td>
    </tr>
    <tr>
      <th>32</th>
      <td>575</td>
    </tr>
    <tr>
      <th>60</th>
      <td>530</td>
    </tr>
    <tr>
      <th>99</th>
      <td>200</td>
    </tr>
    <tr>
      <th>20</th>
      <td>163</td>
    </tr>
    <tr>
      <th>12</th>
      <td>99</td>
    </tr>
    <tr>
      <th>49</th>
      <td>89</td>
    </tr>
    <tr>
      <th>13</th>
      <td>76</td>
    </tr>
    <tr>
      <th>61</th>
      <td>38</td>
    </tr>
    <tr>
      <th>33</th>
      <td>33</td>
    </tr>
    <tr>
      <th>53</th>
      <td>9</td>
    </tr>
    <tr>
      <th>50</th>
      <td>7</td>
    </tr>
    <tr>
      <th>15</th>
      <td>7</td>
    </tr>
    <tr>
      <th>51</th>
      <td>6</td>
    </tr>
    <tr>
      <th>42</th>
      <td>4</td>
    </tr>
    <tr>
      <th>54</th>
      <td>4</td>
    </tr>
    <tr>
      <th>71</th>
      <td>4</td>
    </tr>
    <tr>
      <th>72</th>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3</td>
    </tr>
    <tr>
      <th>65</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select only the desired columns and remove duplicates if any
partb_physicians_df = partb_physicians_raw_df[[
    'cur_clm_uniq_id', 'bene_mbi_id', 'rndrg_prvdr_npi_num', 'clm_from_dt', 
    'clm_line_hcpcs_cd', 'clm_line_dgns_cd', 'clm_line_alowd_chrg_amt'
]].drop_duplicates().rename(
    columns={
        'cur_clm_uniq_id': 'claim_id',
        'bene_mbi_id': 'patient_id',
        'rndrg_prvdr_npi_num': 'npi_id',
        'clm_from_dt': 'claim_date',
        'clm_line_hcpcs_cd': 'hcpcs_code', 
        'clm_line_dgns_cd': 'diagnosis_code', 
        'clm_line_alowd_chrg_amt': 'claim_cost'
    }
)
partb_physicians_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>diagnosis_code</th>
      <th>claim_cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100117</td>
      <td>10046</td>
      <td>1.073515e+09</td>
      <td>2016-11-19</td>
      <td>83861</td>
      <td>H04123</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001777</td>
      <td>10133</td>
      <td>1.053398e+09</td>
      <td>2016-12-15</td>
      <td>99213</td>
      <td>I480</td>
      <td>69.50</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001907</td>
      <td>10113</td>
      <td>1.245238e+09</td>
      <td>2017-02-09</td>
      <td>11721</td>
      <td>B351</td>
      <td>43.37</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1002867</td>
      <td>10049</td>
      <td>1.255316e+09</td>
      <td>2017-09-23</td>
      <td>88312</td>
      <td>D0359</td>
      <td>143.39</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1002871</td>
      <td>10026</td>
      <td>1.265419e+09</td>
      <td>2016-03-11</td>
      <td>87086</td>
      <td>N390</td>
      <td>10.66</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>130694</th>
      <td>999919</td>
      <td>12345</td>
      <td>1.962494e+09</td>
      <td>2018-05-03</td>
      <td>99214</td>
      <td>E782</td>
      <td>105.49</td>
    </tr>
    <tr>
      <th>130695</th>
      <td>999919</td>
      <td>12345</td>
      <td>1.962494e+09</td>
      <td>2018-05-03</td>
      <td>90732</td>
      <td>Z23</td>
      <td>108.14</td>
    </tr>
    <tr>
      <th>130696</th>
      <td>999919</td>
      <td>12345</td>
      <td>1.962494e+09</td>
      <td>2018-05-03</td>
      <td>G0009</td>
      <td>Z23</td>
      <td>19.91</td>
    </tr>
    <tr>
      <th>130697</th>
      <td>999959</td>
      <td>11445</td>
      <td>1.548250e+09</td>
      <td>2016-09-24</td>
      <td>66984</td>
      <td>H2512</td>
      <td>838.19</td>
    </tr>
    <tr>
      <th>130698</th>
      <td>999959</td>
      <td>11445</td>
      <td>1.548250e+09</td>
      <td>2016-09-24</td>
      <td>G8918</td>
      <td>H2512</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>128904 rows × 7 columns</p>
</div>



#### Data Quality Check #9: If the resulting dataframe is empty, it means all the records have HCPCS or diagnosis code (if it's not empty it should be removed now since we want only the ones with valid codes for analysis)


```python
partb_physicians_df.loc[
    (~partb_physicians_df.hcpcs_code.notnull()) | (~partb_physicians_df.diagnosis_code.notnull())
]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>diagnosis_code</th>
      <th>claim_cost</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



#### Mini-Analysis #4: Find whether there are matching claims between above four datasets and the Physicians dataset

                                                ***


```python
physicians_unique_claims_df = partb_physicians_df[[
    'claim_id'
]].drop_duplicates()

physicians_unique_claims_df['physicians'] = 1

joined_df4 = pd.merge(
    joined_df3,
    physicians_unique_claims_df,
    on='claim_id', how = 'outer'
)
joined_df4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>header</th>
      <th>revenue</th>
      <th>diagnosis</th>
      <th>dme</th>
      <th>physicians</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001595</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1004555</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1011605</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1011758</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101424</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>97953</th>
      <td>999905</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>97954</th>
      <td>999908</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>97955</th>
      <td>999916</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>97956</th>
      <td>999919</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>97957</th>
      <td>999959</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>97958 rows × 6 columns</p>
</div>




```python
print('# of unique claims in first four datasets: ' 
      + str(joined_df3.shape[0])
     )
print('# of unique claims in Physicians dataset: ' 
      + str(physicians_unique_claims_df.shape[0])
     )
```

    # of unique claims in first four datasets: 20965
    # of unique claims in Physicians dataset: 76993



```python
print('# of unique claims in the five datasets combined: ' 
      + str(joined_df4.shape[0])
     )
print('From combined list of unique claims - ')
print('# of unique claims in only the first four datasets: ' 
      + str(joined_df4.loc[
          ((joined_df4.header == 1) 
          | (joined_df4.revenue == 1)
          | (joined_df4.diagnosis == 1)
          | (joined_df4.dme == 1))
          & ~(joined_df4.physicians == 1)
      ].shape[0])
     )
print('# of unique claims in only Physicians dataset: ' 
      + str(joined_df4.loc[
          ~(joined_df4.header == 1) 
          & ~(joined_df4.revenue == 1)
          & ~(joined_df4.diagnosis == 1)
          & ~(joined_df4.dme == 1)
          & (joined_df4.physicians == 1)
      ].shape[0])
     )
print('# of unique claims in all five datasets: ' 
      + str(joined_df4.loc[
          (joined_df4.header == 1) 
          & (joined_df4.revenue == 1)
          & (joined_df4.diagnosis == 1)
          & (joined_df4.dme == 1)
          & (joined_df4.physicians == 1)
      ].shape[0])
     )
print('# of unique claims in Physicians and any of the first four datasets: ' 
      + str(joined_df4.loc[
          ((joined_df4.header == 1) 
          | (joined_df4.revenue == 1)
          | (joined_df4.diagnosis == 1)
          | (joined_df4.dme == 1))
          & (joined_df4.physicians == 1)
      ].shape[0])
     )
```

    # of unique claims in the five datasets combined: 97958
    From combined list of unique claims - 
    # of unique claims in only the first four datasets: 20965
    # of unique claims in only Physicians dataset: 76993
    # of unique claims in all five datasets: 0
    # of unique claims in Physicians and any of the first four datasets: 0


Conclusion: None of the claims from Physicians is in any of the first four datasets, so append them to the output after combining the first four datasets

                                                ***

### 0.2.7 Load & select columns from Patients dataset


```python
# Load the Patients dataset
# For sake of simplicity in concept, beneficiary = patient
beneficiary_demographics_df = pd.read_csv("/Users/hamiddastgir/Library/CloudStorage/Dropbox/Semester 3/BIA 810 - Healthcare Analytics/Mid Term/Syntegra Datasets Files/beneficiary_demographics.csv")
beneficiary_demographics_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bene_mbi_id</th>
      <th>bene_member_month</th>
      <th>bene_hic_num</th>
      <th>bene_fips_state_cd</th>
      <th>bene_fips_cnty_cd</th>
      <th>bene_zip_cd</th>
      <th>bene_dob</th>
      <th>bene_sex_cd</th>
      <th>bene_race_cd</th>
      <th>bene_age</th>
      <th>bene_mdcr_stus_cd</th>
      <th>bene_dual_stus_cd</th>
      <th>bene_death_dt</th>
      <th>bene_rng_bgn_dt</th>
      <th>bene_rng_end_dt</th>
      <th>bene_1st_name</th>
      <th>bene_midl_name</th>
      <th>bene_last_name</th>
      <th>bene_orgnl_entlmt_rsn_cd</th>
      <th>bene_entlmt_buyin_ind</th>
      <th>bene_part_a_enrlmt_bgn_dt</th>
      <th>bene_part_b_enrlmt_bgn_dt</th>
      <th>bene_line_1_adr</th>
      <th>bene_line_2_adr</th>
      <th>bene_line_3_adr</th>
      <th>bene_line_4_adr</th>
      <th>bene_line_5_adr</th>
      <th>bene_line_6_adr</th>
      <th>geo_zip_plc_name</th>
      <th>geo_usps_state_cd</th>
      <th>geo_zip5_cd</th>
      <th>geo_zip4_cd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>1/1/16 0:00</td>
      <td>NaN</td>
      <td>55</td>
      <td>79</td>
      <td>NaN</td>
      <td>5/16/45 0:00</td>
      <td>1</td>
      <td>1</td>
      <td>71</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>52</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>2/1/16 0:00</td>
      <td>NaN</td>
      <td>55</td>
      <td>79</td>
      <td>NaN</td>
      <td>5/16/45 0:00</td>
      <td>1</td>
      <td>1</td>
      <td>71</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>52</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>3/1/16 0:00</td>
      <td>NaN</td>
      <td>55</td>
      <td>79</td>
      <td>NaN</td>
      <td>5/16/45 0:00</td>
      <td>1</td>
      <td>1</td>
      <td>71</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>52</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10</td>
      <td>4/1/16 0:00</td>
      <td>NaN</td>
      <td>55</td>
      <td>79</td>
      <td>NaN</td>
      <td>5/16/45 0:00</td>
      <td>1</td>
      <td>1</td>
      <td>71</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>52</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>5/1/16 0:00</td>
      <td>NaN</td>
      <td>55</td>
      <td>79</td>
      <td>NaN</td>
      <td>5/16/45 0:00</td>
      <td>1</td>
      <td>1</td>
      <td>71</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>52</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>31179</th>
      <td>13380</td>
      <td>2/1/18 0:00</td>
      <td>NaN</td>
      <td>44</td>
      <td>7</td>
      <td>NaN</td>
      <td>3/31/47 0:00</td>
      <td>2</td>
      <td>1</td>
      <td>71</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31180</th>
      <td>13380</td>
      <td>3/1/18 0:00</td>
      <td>NaN</td>
      <td>44</td>
      <td>7</td>
      <td>NaN</td>
      <td>3/31/47 0:00</td>
      <td>2</td>
      <td>1</td>
      <td>71</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31181</th>
      <td>13380</td>
      <td>4/1/18 0:00</td>
      <td>NaN</td>
      <td>44</td>
      <td>7</td>
      <td>NaN</td>
      <td>3/31/47 0:00</td>
      <td>2</td>
      <td>1</td>
      <td>71</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31182</th>
      <td>13380</td>
      <td>5/1/18 0:00</td>
      <td>NaN</td>
      <td>44</td>
      <td>7</td>
      <td>NaN</td>
      <td>3/31/47 0:00</td>
      <td>2</td>
      <td>1</td>
      <td>71</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31183</th>
      <td>13380</td>
      <td>6/1/18 0:00</td>
      <td>NaN</td>
      <td>44</td>
      <td>7</td>
      <td>NaN</td>
      <td>3/31/47 0:00</td>
      <td>2</td>
      <td>1</td>
      <td>71</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>31184 rows × 32 columns</p>
</div>




```python
# Select only the desired columns (rename columns if needed) and remove duplicates if any
beneficiary_demographics_df = beneficiary_demographics_df[[
    'bene_mbi_id', 'bene_dob', 'bene_sex_cd'
]].drop_duplicates().rename(
    columns={
        'bene_mbi_id': 'patient_id',
        'bene_dob': 'patient_birth_date'
    }
)
beneficiary_demographics_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>patient_birth_date</th>
      <th>bene_sex_cd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>5/16/45 0:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>10007</td>
      <td>1/4/56 0:00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>63</th>
      <td>10010</td>
      <td>12/3/32 0:00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>95</th>
      <td>10013</td>
      <td>8/23/52 0:00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>131</th>
      <td>10017</td>
      <td>11/23/84 0:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>31037</th>
      <td>13374</td>
      <td>7/11/48 0:00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>31072</th>
      <td>13376</td>
      <td>11/28/52 0:00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>31103</th>
      <td>13377</td>
      <td>1/16/56 0:00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31122</th>
      <td>13379</td>
      <td>12/10/26 0:00</td>
      <td>2</td>
    </tr>
    <tr>
      <th>31155</th>
      <td>13380</td>
      <td>3/31/47 0:00</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 3 columns</p>
</div>




```python
# Gender code as identified by the CMS CCLF resource (1 = male, 2 = female, 0 = Unknown = N/A)
# Convert gender code into readable acronym and drop original column
beneficiary_demographics_df['patient_gender'] = ''
beneficiary_demographics_df.loc[beneficiary_demographics_df.bene_sex_cd == 1, 'patient_gender'] = 'M'
beneficiary_demographics_df.loc[beneficiary_demographics_df.bene_sex_cd == 2, 'patient_gender'] = 'F'
beneficiary_demographics_df = beneficiary_demographics_df.drop('bene_sex_cd', axis=1)
beneficiary_demographics_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>patient_birth_date</th>
      <th>patient_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>5/16/45 0:00</td>
      <td>M</td>
    </tr>
    <tr>
      <th>34</th>
      <td>10007</td>
      <td>1/4/56 0:00</td>
      <td>F</td>
    </tr>
    <tr>
      <th>63</th>
      <td>10010</td>
      <td>12/3/32 0:00</td>
      <td>F</td>
    </tr>
    <tr>
      <th>95</th>
      <td>10013</td>
      <td>8/23/52 0:00</td>
      <td>F</td>
    </tr>
    <tr>
      <th>131</th>
      <td>10017</td>
      <td>11/23/84 0:00</td>
      <td>M</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>31037</th>
      <td>13374</td>
      <td>7/11/48 0:00</td>
      <td>F</td>
    </tr>
    <tr>
      <th>31072</th>
      <td>13376</td>
      <td>11/28/52 0:00</td>
      <td>F</td>
    </tr>
    <tr>
      <th>31103</th>
      <td>13377</td>
      <td>1/16/56 0:00</td>
      <td>M</td>
    </tr>
    <tr>
      <th>31122</th>
      <td>13379</td>
      <td>12/10/26 0:00</td>
      <td>F</td>
    </tr>
    <tr>
      <th>31155</th>
      <td>13380</td>
      <td>3/31/47 0:00</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 3 columns</p>
</div>




```python
beneficiary_demographics_df['patient_birth_date'].head()
```




    0       5/16/45 0:00
    34       1/4/56 0:00
    63      12/3/32 0:00
    95      8/23/52 0:00
    131    11/23/84 0:00
    Name: patient_birth_date, dtype: object




```python
print(beneficiary_demographics_df['patient_birth_date'].head(10))
```

    0       5/16/45 0:00
    34       1/4/56 0:00
    63      12/3/32 0:00
    95      8/23/52 0:00
    131    11/23/84 0:00
    160     5/18/38 0:00
    195      3/1/48 0:00
    228     9/26/45 0:00
    262      2/9/46 0:00
    289      5/1/50 0:00
    Name: patient_birth_date, dtype: object


# Changing Birth Date Formatting Since it Causes Issues Later Onwards


```python
def parse_patient_birth_date(date_str):
    if pd.isna(date_str) or date_str.strip() == '':
        return pd.NaT
    date_str = date_str.strip()
    for fmt in ('%m/%d/%y %H:%M', '%m/%d/%Y %H:%M', '%Y-%m-%d'):
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.year > datetime.now().year:
                dt = dt.replace(year=dt.year - 100)
            return dt
        except ValueError:
            continue
    print(f"Could not parse date: {date_str}")
    return pd.NaT
```


```python
beneficiary_demographics_df['patient_birth_date'] = beneficiary_demographics_df['patient_birth_date'].apply(parse_patient_birth_date)
```


```python
beneficiary_demographics_df['patient_birth_date']
```




    0       1945-05-16
    34      1956-01-04
    63      1932-12-03
    95      1952-08-23
    131     1984-11-23
               ...    
    31037   1948-07-11
    31072   1952-11-28
    31103   1956-01-16
    31122   1926-12-10
    31155   1947-03-31
    Name: patient_birth_date, Length: 1000, dtype: datetime64[ns]




```python
beneficiary_demographics_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>patient_birth_date</th>
      <th>patient_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>1945-05-16</td>
      <td>M</td>
    </tr>
    <tr>
      <th>34</th>
      <td>10007</td>
      <td>1956-01-04</td>
      <td>F</td>
    </tr>
    <tr>
      <th>63</th>
      <td>10010</td>
      <td>1932-12-03</td>
      <td>F</td>
    </tr>
    <tr>
      <th>95</th>
      <td>10013</td>
      <td>1952-08-23</td>
      <td>F</td>
    </tr>
    <tr>
      <th>131</th>
      <td>10017</td>
      <td>1984-11-23</td>
      <td>M</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>31037</th>
      <td>13374</td>
      <td>1948-07-11</td>
      <td>F</td>
    </tr>
    <tr>
      <th>31072</th>
      <td>13376</td>
      <td>1952-11-28</td>
      <td>F</td>
    </tr>
    <tr>
      <th>31103</th>
      <td>13377</td>
      <td>1956-01-16</td>
      <td>M</td>
    </tr>
    <tr>
      <th>31122</th>
      <td>13379</td>
      <td>1926-12-10</td>
      <td>F</td>
    </tr>
    <tr>
      <th>31155</th>
      <td>13380</td>
      <td>1947-03-31</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 3 columns</p>
</div>



#### Mini-Analysis #5: Find whether there are matching patients between the claims datasets and the Patients dataset

                                                ***


```python
claims_header_unique_patients_df = parta_claims_header_df[[
    'patient_id'
]].drop_duplicates()

claims_header_unique_patients_df['header'] = 1

revenue_center_unique_patients_df = parta_claims_revenue_center_detail_df[[
    'patient_id'
]].drop_duplicates()

revenue_center_unique_patients_df['revenue'] = 1

diagnosis_unique_patients_df = parta_diagnosis_code_df[[
    'patient_id'
]].drop_duplicates()

diagnosis_unique_patients_df['diagnosis'] = 1

dme_unique_patients_df = partb_dme_df[[
    'patient_id'
]].drop_duplicates()

dme_unique_patients_df['dme'] = 1

physicians_unique_patients_df = partb_physicians_df[[
    'patient_id'
]].drop_duplicates()

physicians_unique_patients_df['physicians'] = 1

beneficiary_unique_patients_df = beneficiary_demographics_df[[
    'patient_id'
]].drop_duplicates()

beneficiary_unique_patients_df['beneficiary'] = 1

joined_patients_df = pd.merge(
    pd.merge(
        pd.merge(
            pd.merge(
                pd.merge(
                    claims_header_unique_patients_df,
                    revenue_center_unique_patients_df,
                    on='patient_id', how = 'outer'
                ),
                diagnosis_unique_patients_df,
                on='patient_id', how = 'outer'
            ),
            dme_unique_patients_df,
            on='patient_id', how = 'outer'
        ),
        physicians_unique_patients_df,
        on='patient_id', how = 'outer'
    ),
    beneficiary_unique_patients_df,
    on='patient_id', how = 'outer'
)

joined_patients_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>header</th>
      <th>revenue</th>
      <th>diagnosis</th>
      <th>dme</th>
      <th>physicians</th>
      <th>beneficiary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10226</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10133</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10163</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1003</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10052</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>12868</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>996</th>
      <td>13001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>997</th>
      <td>13157</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>998</th>
      <td>13298</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>999</th>
      <td>13351</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 7 columns</p>
</div>




```python
print('# of unique patients in the five datasets combined: ' 
      + str(joined_patients_df.shape[0])
     )
print('From combined list of unique patients - ')
print('# of unique patients in only the claims datasets: ' 
      + str(joined_patients_df.loc[
          ((joined_patients_df.header == 1) 
          | (joined_patients_df.revenue == 1)
          | (joined_patients_df.diagnosis == 1)
          | (joined_patients_df.dme == 1)
          | (joined_patients_df.physicians == 1))
          & ~(joined_patients_df.beneficiary == 1)
      ].shape[0])
     )
print('# of unique patients in only Beneficiary dataset: ' 
      + str(joined_patients_df.loc[
          ~(joined_patients_df.header == 1) 
          & ~(joined_patients_df.revenue == 1)
          & ~(joined_patients_df.diagnosis == 1)
          & ~(joined_patients_df.dme == 1)
          & ~(joined_patients_df.physicians == 1)
          & (joined_patients_df.beneficiary == 1)
      ].shape[0])
     )
print('# of unique patients in all five datasets: ' 
      + str(joined_patients_df.loc[
          (joined_patients_df.header == 1) 
          & (joined_patients_df.revenue == 1)
          & (joined_patients_df.diagnosis == 1)
          & (joined_patients_df.dme == 1)
          & (joined_patients_df.physicians == 1)
          & (joined_patients_df.beneficiary == 1)
      ].shape[0])
     )
print('# of unique patients in Beneficiary and any of the claims datasets: ' 
      + str(joined_patients_df.loc[
          ((joined_patients_df.header == 1) 
          | (joined_patients_df.revenue == 1)
          | (joined_patients_df.diagnosis == 1)
          | (joined_patients_df.dme == 1)
          | (joined_patients_df.physicians == 1))
          & (joined_patients_df.beneficiary == 1)
      ].shape[0])
     )
```

    # of unique patients in the five datasets combined: 1000
    From combined list of unique patients - 
    # of unique patients in only the claims datasets: 0
    # of unique patients in only Beneficiary dataset: 38
    # of unique patients in all five datasets: 276
    # of unique patients in Beneficiary and any of the claims datasets: 962


Conclusion: Most of the patients have some claims, so we can join the beneficiary dataset to the claims to get some of the patient demographics, i.e. age and gender

                                                ***

### 0.3 Combine all datasets

### 0.3.1. Join datasets with common records


```python
# Join Claims Header and Claims Revenue Center datasets on claim ID, patient ID, and claim date
# Perform outer join to capture all possible claims
medicare_df = pd.merge(
    parta_claims_header_df,
    parta_claims_revenue_center_detail_df,
    on=['claim_id','patient_id','claim_date'], how='outer'
)
medicare_df.sort_values(by='claim_id')#.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>prncpl_dgns_cd</th>
      <th>clm_pmt_amt</th>
      <th>hcpcs_code</th>
      <th>clm_line_cvrd_pd_amt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22501</th>
      <td>100073</td>
      <td>12620</td>
      <td>NaN</td>
      <td>2018-12-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>77063</td>
      <td>24.11</td>
    </tr>
    <tr>
      <th>1281</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>M1611</td>
      <td>127.79</td>
      <td>98960</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1285</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>M1611</td>
      <td>127.79</td>
      <td>99213</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1284</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>M1611</td>
      <td>127.79</td>
      <td>J2270</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1283</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>M1611</td>
      <td>127.79</td>
      <td>J1885</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30555</th>
      <td>1699195</td>
      <td>10958</td>
      <td>NaN</td>
      <td>2017-04-19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>00810</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>30556</th>
      <td>1699195</td>
      <td>10958</td>
      <td>NaN</td>
      <td>2017-04-19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>J2250</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>30557</th>
      <td>1699197</td>
      <td>1177</td>
      <td>NaN</td>
      <td>2016-05-22</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>36415</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>30558</th>
      <td>1699197</td>
      <td>1177</td>
      <td>NaN</td>
      <td>2016-05-22</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>86592</td>
      <td>5.43</td>
    </tr>
    <tr>
      <th>30559</th>
      <td>1699236</td>
      <td>10580</td>
      <td>NaN</td>
      <td>2017-09-20</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45385</td>
      <td>543.04</td>
    </tr>
  </tbody>
</table>
<p>38394 rows × 8 columns</p>
</div>




```python
# Join Medicare and Diagnosis datasets on claim ID, patient ID, and claim date
# Perform outer join to capture all possible claims
medicare_df = pd.merge(
    medicare_df,
    parta_diagnosis_code_df,
    on=['claim_id','patient_id','claim_date'], how='outer'
)
medicare_df.sort_values(by='claim_id')#.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>prncpl_dgns_cd</th>
      <th>clm_pmt_amt</th>
      <th>hcpcs_code</th>
      <th>clm_line_cvrd_pd_amt</th>
      <th>clm_dgns_cd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62201</th>
      <td>100073</td>
      <td>12620</td>
      <td>NaN</td>
      <td>2018-12-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>77063</td>
      <td>24.11</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3396</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>M1611</td>
      <td>127.79</td>
      <td>99213</td>
      <td>0.00</td>
      <td>M5136</td>
    </tr>
    <tr>
      <th>3395</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>M1611</td>
      <td>127.79</td>
      <td>99213</td>
      <td>0.00</td>
      <td>M1611</td>
    </tr>
    <tr>
      <th>3394</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>M1611</td>
      <td>127.79</td>
      <td>99213</td>
      <td>0.00</td>
      <td>M25572</td>
    </tr>
    <tr>
      <th>3393</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>M1611</td>
      <td>127.79</td>
      <td>J2270</td>
      <td>0.00</td>
      <td>M25551</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>79460</th>
      <td>1699195</td>
      <td>10958</td>
      <td>NaN</td>
      <td>2017-04-19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>88342</td>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>79461</th>
      <td>1699195</td>
      <td>10958</td>
      <td>NaN</td>
      <td>2017-04-19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>43239</td>
      <td>1275.58</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>79464</th>
      <td>1699197</td>
      <td>1177</td>
      <td>NaN</td>
      <td>2016-05-22</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>36415</td>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>79465</th>
      <td>1699197</td>
      <td>1177</td>
      <td>NaN</td>
      <td>2016-05-22</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>86592</td>
      <td>5.43</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>79466</th>
      <td>1699236</td>
      <td>10580</td>
      <td>NaN</td>
      <td>2017-09-20</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45385</td>
      <td>543.04</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>106785 rows × 9 columns</p>
</div>




```python
# Since Claims Header dataset has some principal diagnosis codes and the Diagnosis dataset 
#  supplements them with additional codes wherever possible, 
#  coalesce them with preference to the principal code from Claim Header dataset
# Once the diagnosis codes are combined into one column, remove the older columns and any duplicates
medicare_df['diagnosis_code'] = medicare_df[['prncpl_dgns_cd', 'clm_dgns_cd']].bfill(axis=1).iloc[:, 0]
medicare_df = medicare_df.drop(['prncpl_dgns_cd', 'clm_dgns_cd'], axis=1).drop_duplicates()
medicare_df.sort_values(by='claim_id').head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>clm_pmt_amt</th>
      <th>hcpcs_code</th>
      <th>clm_line_cvrd_pd_amt</th>
      <th>diagnosis_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62201</th>
      <td>100073</td>
      <td>12620</td>
      <td>NaN</td>
      <td>2018-12-02</td>
      <td>NaN</td>
      <td>77063</td>
      <td>24.11</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3386</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>127.79</td>
      <td>J1885</td>
      <td>0.00</td>
      <td>M1611</td>
    </tr>
    <tr>
      <th>3378</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>127.79</td>
      <td>98960</td>
      <td>0.00</td>
      <td>M1611</td>
    </tr>
    <tr>
      <th>3374</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>127.79</td>
      <td>G0467</td>
      <td>133.74</td>
      <td>M1611</td>
    </tr>
    <tr>
      <th>3390</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>127.79</td>
      <td>J2270</td>
      <td>0.00</td>
      <td>M1611</td>
    </tr>
    <tr>
      <th>3394</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>127.79</td>
      <td>99213</td>
      <td>0.00</td>
      <td>M1611</td>
    </tr>
    <tr>
      <th>3382</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>127.79</td>
      <td>J1100</td>
      <td>0.00</td>
      <td>M1611</td>
    </tr>
    <tr>
      <th>62426</th>
      <td>100227</td>
      <td>12140</td>
      <td>NaN</td>
      <td>2018-10-24</td>
      <td>NaN</td>
      <td>J2785</td>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>96298</th>
      <td>100402</td>
      <td>1261</td>
      <td>NaN</td>
      <td>2017-06-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>E119</td>
    </tr>
    <tr>
      <th>96297</th>
      <td>100402</td>
      <td>1261</td>
      <td>NaN</td>
      <td>2017-06-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>R197</td>
    </tr>
    <tr>
      <th>3670</th>
      <td>100402</td>
      <td>1261</td>
      <td>1.285688e+09</td>
      <td>2017-05-27</td>
      <td>10602.46</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>K5733</td>
    </tr>
    <tr>
      <th>3682</th>
      <td>100464</td>
      <td>12978</td>
      <td>1.982693e+09</td>
      <td>2017-06-26</td>
      <td>199.45</td>
      <td>74230</td>
      <td>83.17</td>
      <td>R079</td>
    </tr>
    <tr>
      <th>96310</th>
      <td>100564</td>
      <td>11929</td>
      <td>NaN</td>
      <td>2018-04-12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>F17210</td>
    </tr>
    <tr>
      <th>3700</th>
      <td>100698</td>
      <td>11789</td>
      <td>1.912991e+09</td>
      <td>2017-07-28</td>
      <td>85.25</td>
      <td>G0463</td>
      <td>85.05</td>
      <td>M545</td>
    </tr>
    <tr>
      <th>3705</th>
      <td>100750</td>
      <td>12138</td>
      <td>1.063442e+09</td>
      <td>2018-01-13</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Z0289</td>
    </tr>
    <tr>
      <th>62051</th>
      <td>100974</td>
      <td>10042</td>
      <td>NaN</td>
      <td>2017-02-20</td>
      <td>NaN</td>
      <td>G0202</td>
      <td>99.75</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3763</th>
      <td>100982</td>
      <td>12086</td>
      <td>1.902839e+09</td>
      <td>2018-06-23</td>
      <td>608.45</td>
      <td>95811</td>
      <td>832.94</td>
      <td>G4733</td>
    </tr>
    <tr>
      <th>3776</th>
      <td>101001</td>
      <td>11663</td>
      <td>1.932133e+09</td>
      <td>2016-11-08</td>
      <td>50.20</td>
      <td>71020</td>
      <td>48.65</td>
      <td>R918</td>
    </tr>
    <tr>
      <th>96381</th>
      <td>101117</td>
      <td>11835</td>
      <td>NaN</td>
      <td>2018-04-10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>N6002</td>
    </tr>
    <tr>
      <th>62674</th>
      <td>101147</td>
      <td>13359</td>
      <td>NaN</td>
      <td>2017-09-02</td>
      <td>NaN</td>
      <td>80053</td>
      <td>11.21</td>
      <td>E785</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Since Claims Header dataset records the amount Medicare paid for the claims 
#  and Claims Revenue Center dataset records the amount Medicare reimbursed the provider,
#  assume they were separate charges and add them to get the total cost for claim (for particular code)
# Once the costs are combined into one column, remove the older columns and any duplicates
medicare_df['claim_cost'] = medicare_df['clm_pmt_amt']+medicare_df['clm_line_cvrd_pd_amt']
medicare_df = medicare_df.drop(['clm_pmt_amt', 'clm_line_cvrd_pd_amt'], axis=1).drop_duplicates()
medicare_df.sort_values(by='claim_id').head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>diagnosis_code</th>
      <th>claim_cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62201</th>
      <td>100073</td>
      <td>12620</td>
      <td>NaN</td>
      <td>2018-12-02</td>
      <td>77063</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3374</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>G0467</td>
      <td>M1611</td>
      <td>261.53</td>
    </tr>
    <tr>
      <th>3394</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>99213</td>
      <td>M1611</td>
      <td>127.79</td>
    </tr>
    <tr>
      <th>3390</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>J2270</td>
      <td>M1611</td>
      <td>127.79</td>
    </tr>
    <tr>
      <th>3378</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>98960</td>
      <td>M1611</td>
      <td>127.79</td>
    </tr>
    <tr>
      <th>3386</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>J1885</td>
      <td>M1611</td>
      <td>127.79</td>
    </tr>
    <tr>
      <th>3382</th>
      <td>100190</td>
      <td>1228</td>
      <td>1.972732e+09</td>
      <td>2018-06-10</td>
      <td>J1100</td>
      <td>M1611</td>
      <td>127.79</td>
    </tr>
    <tr>
      <th>62426</th>
      <td>100227</td>
      <td>12140</td>
      <td>NaN</td>
      <td>2018-10-24</td>
      <td>J2785</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3670</th>
      <td>100402</td>
      <td>1261</td>
      <td>1.285688e+09</td>
      <td>2017-05-27</td>
      <td>NaN</td>
      <td>K5733</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>96297</th>
      <td>100402</td>
      <td>1261</td>
      <td>NaN</td>
      <td>2017-06-02</td>
      <td>NaN</td>
      <td>R197</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>96298</th>
      <td>100402</td>
      <td>1261</td>
      <td>NaN</td>
      <td>2017-06-02</td>
      <td>NaN</td>
      <td>E119</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3682</th>
      <td>100464</td>
      <td>12978</td>
      <td>1.982693e+09</td>
      <td>2017-06-26</td>
      <td>74230</td>
      <td>R079</td>
      <td>282.62</td>
    </tr>
    <tr>
      <th>96310</th>
      <td>100564</td>
      <td>11929</td>
      <td>NaN</td>
      <td>2018-04-12</td>
      <td>NaN</td>
      <td>F17210</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3700</th>
      <td>100698</td>
      <td>11789</td>
      <td>1.912991e+09</td>
      <td>2017-07-28</td>
      <td>G0463</td>
      <td>M545</td>
      <td>170.30</td>
    </tr>
    <tr>
      <th>3705</th>
      <td>100750</td>
      <td>12138</td>
      <td>1.063442e+09</td>
      <td>2018-01-13</td>
      <td>NaN</td>
      <td>Z0289</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>62051</th>
      <td>100974</td>
      <td>10042</td>
      <td>NaN</td>
      <td>2017-02-20</td>
      <td>G0202</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3763</th>
      <td>100982</td>
      <td>12086</td>
      <td>1.902839e+09</td>
      <td>2018-06-23</td>
      <td>95811</td>
      <td>G4733</td>
      <td>1441.39</td>
    </tr>
    <tr>
      <th>3776</th>
      <td>101001</td>
      <td>11663</td>
      <td>1.932133e+09</td>
      <td>2016-11-08</td>
      <td>71020</td>
      <td>R918</td>
      <td>98.85</td>
    </tr>
    <tr>
      <th>96381</th>
      <td>101117</td>
      <td>11835</td>
      <td>NaN</td>
      <td>2018-04-10</td>
      <td>NaN</td>
      <td>N6002</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>62674</th>
      <td>101147</td>
      <td>13359</td>
      <td>NaN</td>
      <td>2017-09-02</td>
      <td>80053</td>
      <td>E785</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
claims_header_revenue_diagnosis_df_count = medicare_df.shape[0]
```

### 0.3.2 Append datasets with no common records


```python
# Create a dummy column for diagnosis code in DME dataset before appending so the list of columns match
partb_dme_df['diagnosis_code'] = np.nan
partb_dme_df = partb_dme_df[[
    'claim_id', 'patient_id', 'npi_id', 'claim_date', 'hcpcs_code', 'diagnosis_code', 'claim_cost'
]]
partb_dme_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>diagnosis_code</th>
      <th>claim_cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1004024</td>
      <td>10202</td>
      <td>1.841430e+09</td>
      <td>2016-07-18</td>
      <td>E0601</td>
      <td>NaN</td>
      <td>41.91</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1034063</td>
      <td>10137</td>
      <td>1.669460e+09</td>
      <td>2016-04-22</td>
      <td>E0601</td>
      <td>NaN</td>
      <td>62.46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1046877</td>
      <td>10202</td>
      <td>1.093713e+09</td>
      <td>2016-02-03</td>
      <td>E0601</td>
      <td>NaN</td>
      <td>29.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1072934</td>
      <td>10202</td>
      <td>1.285602e+09</td>
      <td>2016-08-15</td>
      <td>E0601</td>
      <td>NaN</td>
      <td>27.82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1082554</td>
      <td>10174</td>
      <td>1.003895e+09</td>
      <td>2016-08-30</td>
      <td>E0431</td>
      <td>NaN</td>
      <td>18.75</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2770</th>
      <td>998097</td>
      <td>10396</td>
      <td>1.891706e+09</td>
      <td>2016-12-06</td>
      <td>A4253</td>
      <td>NaN</td>
      <td>69.69</td>
    </tr>
    <tr>
      <th>2771</th>
      <td>999226</td>
      <td>1095</td>
      <td>1.518066e+09</td>
      <td>2017-12-28</td>
      <td>A4256</td>
      <td>NaN</td>
      <td>3.68</td>
    </tr>
    <tr>
      <th>2772</th>
      <td>999226</td>
      <td>1095</td>
      <td>1.518066e+09</td>
      <td>2017-12-28</td>
      <td>A4253</td>
      <td>NaN</td>
      <td>49.92</td>
    </tr>
    <tr>
      <th>2773</th>
      <td>999226</td>
      <td>1095</td>
      <td>1.518066e+09</td>
      <td>2017-12-28</td>
      <td>A4259</td>
      <td>NaN</td>
      <td>4.26</td>
    </tr>
    <tr>
      <th>2774</th>
      <td>999929</td>
      <td>10261</td>
      <td>1.497738e+09</td>
      <td>2018-06-12</td>
      <td>E0570</td>
      <td>NaN</td>
      <td>3.00</td>
    </tr>
  </tbody>
</table>
<p>2731 rows × 7 columns</p>
</div>




```python
# Append DME dataset to the first three claims datasets
medicare_df = pd.concat([medicare_df, partb_dme_df])
medicare_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>diagnosis_code</th>
      <th>claim_cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>G0283</td>
      <td>M25551</td>
      <td>268.68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>G8978</td>
      <td>M25551</td>
      <td>259.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>G8979</td>
      <td>M25551</td>
      <td>259.01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>97110</td>
      <td>M25551</td>
      <td>283.98</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>97140</td>
      <td>M25551</td>
      <td>279.34</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2770</th>
      <td>998097</td>
      <td>10396</td>
      <td>1.891706e+09</td>
      <td>2016-12-06</td>
      <td>A4253</td>
      <td>NaN</td>
      <td>69.69</td>
    </tr>
    <tr>
      <th>2771</th>
      <td>999226</td>
      <td>1095</td>
      <td>1.518066e+09</td>
      <td>2017-12-28</td>
      <td>A4256</td>
      <td>NaN</td>
      <td>3.68</td>
    </tr>
    <tr>
      <th>2772</th>
      <td>999226</td>
      <td>1095</td>
      <td>1.518066e+09</td>
      <td>2017-12-28</td>
      <td>A4253</td>
      <td>NaN</td>
      <td>49.92</td>
    </tr>
    <tr>
      <th>2773</th>
      <td>999226</td>
      <td>1095</td>
      <td>1.518066e+09</td>
      <td>2017-12-28</td>
      <td>A4259</td>
      <td>NaN</td>
      <td>4.26</td>
    </tr>
    <tr>
      <th>2774</th>
      <td>999929</td>
      <td>10261</td>
      <td>1.497738e+09</td>
      <td>2018-06-12</td>
      <td>E0570</td>
      <td>NaN</td>
      <td>3.00</td>
    </tr>
  </tbody>
</table>
<p>69648 rows × 7 columns</p>
</div>




```python
claims_header_revenue_diagnosis_dme_df_count = medicare_df.shape[0]
```

#### Data Quality Check #10: If True, we appended DME dataset to the first three claims datasets without any unexpected rows accruing


```python
partb_dme_df_count = partb_dme_df.shape[0]
claims_w_dme_df_count = medicare_df.shape[0]

print('Claim count for Claims Header + Revenue Center + Diagnosis: ' 
      + str(claims_header_revenue_diagnosis_df_count))
print('Claim count for DME: ' + str(partb_dme_df_count))
print('Expected claim count after appending DME dataset: ' 
     + str(claims_header_revenue_diagnosis_df_count+partb_dme_df_count))
print('Actual claim count after appending DME dataset: '
     + str(claims_w_dme_df_count))
print('Expected and actual claim count matches: '
     + str(claims_header_revenue_diagnosis_df_count+partb_dme_df_count == claims_w_dme_df_count))
```

    Claim count for Claims Header + Revenue Center + Diagnosis: 69648
    Claim count for DME: 2731
    Expected claim count after appending DME dataset: 72379
    Actual claim count after appending DME dataset: 69648
    Expected and actual claim count matches: False



```python
# Append Physicians dataset to the first four claims datasets
medicare_df = pd.concat([medicare_df, partb_physicians_df])
medicare_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>diagnosis_code</th>
      <th>claim_cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>G0283</td>
      <td>M25551</td>
      <td>268.68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>G8978</td>
      <td>M25551</td>
      <td>259.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>G8979</td>
      <td>M25551</td>
      <td>259.01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>97110</td>
      <td>M25551</td>
      <td>283.98</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>97140</td>
      <td>M25551</td>
      <td>279.34</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>130694</th>
      <td>999919</td>
      <td>12345</td>
      <td>1.962494e+09</td>
      <td>2018-05-03</td>
      <td>99214</td>
      <td>E782</td>
      <td>105.49</td>
    </tr>
    <tr>
      <th>130695</th>
      <td>999919</td>
      <td>12345</td>
      <td>1.962494e+09</td>
      <td>2018-05-03</td>
      <td>90732</td>
      <td>Z23</td>
      <td>108.14</td>
    </tr>
    <tr>
      <th>130696</th>
      <td>999919</td>
      <td>12345</td>
      <td>1.962494e+09</td>
      <td>2018-05-03</td>
      <td>G0009</td>
      <td>Z23</td>
      <td>19.91</td>
    </tr>
    <tr>
      <th>130697</th>
      <td>999959</td>
      <td>11445</td>
      <td>1.548250e+09</td>
      <td>2016-09-24</td>
      <td>66984</td>
      <td>H2512</td>
      <td>838.19</td>
    </tr>
    <tr>
      <th>130698</th>
      <td>999959</td>
      <td>11445</td>
      <td>1.548250e+09</td>
      <td>2016-09-24</td>
      <td>G8918</td>
      <td>H2512</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>198552 rows × 7 columns</p>
</div>



#### Data Quality Check #11: If True, we appended Physicians dataset to the first four claims datasets without any unexpected rows accruing


```python
partb_physicians_df_count = partb_physicians_df.shape[0]
claims_w_physicians_df_count = medicare_df.shape[0]

print('Claim count for Claims Header + Revenue Center + Diagnosis + DME: ' 
      + str(claims_header_revenue_diagnosis_dme_df_count))
print('Claim count for Physicians: ' + str(partb_physicians_df_count))
print('Expected claim count after appending Physicians dataset: ' 
     + str(claims_header_revenue_diagnosis_dme_df_count+partb_physicians_df_count))
print('Actual claim count after appending Physicians dataset: '
     + str(claims_w_physicians_df_count))
print('Expected and actual claim count matches: '
     + str(claims_header_revenue_diagnosis_dme_df_count+partb_physicians_df_count 
           == claims_w_physicians_df_count))
```

    Claim count for Claims Header + Revenue Center + Diagnosis + DME: 69648
    Claim count for Physicians: 128904
    Expected claim count after appending Physicians dataset: 198552
    Actual claim count after appending Physicians dataset: 198552
    Expected and actual claim count matches: True



```python
# Capture # records now to compare after joining patient details
medicare_claims_df_count = medicare_df.shape[0]
```

### 0.3.3 Join patient information


```python
# Join claims data with patient details on patient ID
# Perform left join to only provide patient details for existing claims
medicare_df = pd.merge(
    medicare_df,
    beneficiary_demographics_df,
    on=['patient_id'], how='left'
)
medicare_df.sort_values(by='claim_id')#.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>diagnosis_code</th>
      <th>claim_cost</th>
      <th>patient_birth_date</th>
      <th>patient_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>70160</th>
      <td>100020</td>
      <td>1070</td>
      <td>1.619972e+09</td>
      <td>2016-10-04</td>
      <td>85610</td>
      <td>I482</td>
      <td>5.49</td>
      <td>1947-06-06</td>
      <td>M</td>
    </tr>
    <tr>
      <th>70165</th>
      <td>100024</td>
      <td>11654</td>
      <td>1.811965e+09</td>
      <td>2016-12-10</td>
      <td>90834</td>
      <td>F319</td>
      <td>79.36</td>
      <td>1977-11-08</td>
      <td>M</td>
    </tr>
    <tr>
      <th>70169</th>
      <td>100030</td>
      <td>12052</td>
      <td>1.336344e+09</td>
      <td>2017-04-15</td>
      <td>93010</td>
      <td>R001</td>
      <td>8.53</td>
      <td>1949-06-25</td>
      <td>F</td>
    </tr>
    <tr>
      <th>70195</th>
      <td>100038</td>
      <td>12345</td>
      <td>1.295730e+09</td>
      <td>2018-07-02</td>
      <td>72158</td>
      <td>M47816</td>
      <td>112.57</td>
      <td>1947-01-28</td>
      <td>F</td>
    </tr>
    <tr>
      <th>70230</th>
      <td>100061</td>
      <td>10252</td>
      <td>1.861493e+09</td>
      <td>2016-07-04</td>
      <td>99213</td>
      <td>L03032</td>
      <td>82.36</td>
      <td>1933-02-09</td>
      <td>F</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>39703</th>
      <td>1699197</td>
      <td>1177</td>
      <td>NaN</td>
      <td>2016-05-22</td>
      <td>36415</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1948-09-07</td>
      <td>M</td>
    </tr>
    <tr>
      <th>39704</th>
      <td>1699197</td>
      <td>1177</td>
      <td>NaN</td>
      <td>2016-05-22</td>
      <td>86592</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1948-09-07</td>
      <td>M</td>
    </tr>
    <tr>
      <th>134759</th>
      <td>1699204</td>
      <td>11540</td>
      <td>1.275519e+09</td>
      <td>2018-05-08</td>
      <td>99214</td>
      <td>M5116</td>
      <td>101.91</td>
      <td>1952-03-08</td>
      <td>F</td>
    </tr>
    <tr>
      <th>134760</th>
      <td>1699222</td>
      <td>11556</td>
      <td>1.932188e+09</td>
      <td>2016-03-16</td>
      <td>J7060</td>
      <td>I872</td>
      <td>10.66</td>
      <td>1955-02-04</td>
      <td>F</td>
    </tr>
    <tr>
      <th>39705</th>
      <td>1699236</td>
      <td>10580</td>
      <td>NaN</td>
      <td>2017-09-20</td>
      <td>45385</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1947-01-13</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
<p>198552 rows × 9 columns</p>
</div>



#### Data Quality Check #12: If True, we joined Patient dataset to the claims datasets without any unexpected rows accruing


```python
# Check # records to ensure no extra records happened accidentally
medicare_claims_n_patient_info_df_count = medicare_df.shape[0]

print('Claims count before adding patient details: ' + str(medicare_claims_df_count))
print('Claims count after adding patient details: ' + str(medicare_claims_n_patient_info_df_count))
print('If True, no extra records were added accidentally from joining patient details into the claims: '
      + str(medicare_claims_df_count == medicare_claims_n_patient_info_df_count))
```

    Claims count before adding patient details: 198552
    Claims count after adding patient details: 198552
    If True, no extra records were added accidentally from joining patient details into the claims: True



```python
# Get claim year
medicare_df['claim_year'] = pd.to_datetime(
    medicare_df['claim_date']
).dt.strftime('%Y')
medicare_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>diagnosis_code</th>
      <th>claim_cost</th>
      <th>patient_birth_date</th>
      <th>patient_gender</th>
      <th>claim_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>G0283</td>
      <td>M25551</td>
      <td>268.68</td>
      <td>1951-02-27</td>
      <td>M</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>G8978</td>
      <td>M25551</td>
      <td>259.01</td>
      <td>1951-02-27</td>
      <td>M</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>G8979</td>
      <td>M25551</td>
      <td>259.01</td>
      <td>1951-02-27</td>
      <td>M</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>97110</td>
      <td>M25551</td>
      <td>283.98</td>
      <td>1951-02-27</td>
      <td>M</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>97140</td>
      <td>M25551</td>
      <td>279.34</td>
      <td>1951-02-27</td>
      <td>M</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>198547</th>
      <td>999919</td>
      <td>12345</td>
      <td>1.962494e+09</td>
      <td>2018-05-03</td>
      <td>99214</td>
      <td>E782</td>
      <td>105.49</td>
      <td>1947-01-28</td>
      <td>F</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>198548</th>
      <td>999919</td>
      <td>12345</td>
      <td>1.962494e+09</td>
      <td>2018-05-03</td>
      <td>90732</td>
      <td>Z23</td>
      <td>108.14</td>
      <td>1947-01-28</td>
      <td>F</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>198549</th>
      <td>999919</td>
      <td>12345</td>
      <td>1.962494e+09</td>
      <td>2018-05-03</td>
      <td>G0009</td>
      <td>Z23</td>
      <td>19.91</td>
      <td>1947-01-28</td>
      <td>F</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>198550</th>
      <td>999959</td>
      <td>11445</td>
      <td>1.548250e+09</td>
      <td>2016-09-24</td>
      <td>66984</td>
      <td>H2512</td>
      <td>838.19</td>
      <td>1945-04-03</td>
      <td>F</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>198551</th>
      <td>999959</td>
      <td>11445</td>
      <td>1.548250e+09</td>
      <td>2016-09-24</td>
      <td>G8918</td>
      <td>H2512</td>
      <td>0.00</td>
      <td>1945-04-03</td>
      <td>F</td>
      <td>2016</td>
    </tr>
  </tbody>
</table>
<p>198552 rows × 10 columns</p>
</div>




```python
# Get patient age - subtract birthdate from claim date year to get the patient age at the time of claims
medicare_df['patient_birth_year'] = pd.to_datetime(
    medicare_df['patient_birth_date']
).dt.strftime('%Y')

medicare_df['patient_age'] = (
    medicare_df['claim_year'].astype('int') - medicare_df['patient_birth_year'].astype('int')
)

medicare_df = medicare_df.drop('patient_birth_year', axis=1)

medicare_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>diagnosis_code</th>
      <th>claim_cost</th>
      <th>patient_birth_date</th>
      <th>patient_gender</th>
      <th>claim_year</th>
      <th>patient_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>G0283</td>
      <td>M25551</td>
      <td>268.68</td>
      <td>1951-02-27</td>
      <td>M</td>
      <td>2018</td>
      <td>67</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>G8978</td>
      <td>M25551</td>
      <td>259.01</td>
      <td>1951-02-27</td>
      <td>M</td>
      <td>2018</td>
      <td>67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>G8979</td>
      <td>M25551</td>
      <td>259.01</td>
      <td>1951-02-27</td>
      <td>M</td>
      <td>2018</td>
      <td>67</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>97110</td>
      <td>M25551</td>
      <td>283.98</td>
      <td>1951-02-27</td>
      <td>M</td>
      <td>2018</td>
      <td>67</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1001595</td>
      <td>10226</td>
      <td>1.366492e+09</td>
      <td>2018-02-28</td>
      <td>97140</td>
      <td>M25551</td>
      <td>279.34</td>
      <td>1951-02-27</td>
      <td>M</td>
      <td>2018</td>
      <td>67</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>198547</th>
      <td>999919</td>
      <td>12345</td>
      <td>1.962494e+09</td>
      <td>2018-05-03</td>
      <td>99214</td>
      <td>E782</td>
      <td>105.49</td>
      <td>1947-01-28</td>
      <td>F</td>
      <td>2018</td>
      <td>71</td>
    </tr>
    <tr>
      <th>198548</th>
      <td>999919</td>
      <td>12345</td>
      <td>1.962494e+09</td>
      <td>2018-05-03</td>
      <td>90732</td>
      <td>Z23</td>
      <td>108.14</td>
      <td>1947-01-28</td>
      <td>F</td>
      <td>2018</td>
      <td>71</td>
    </tr>
    <tr>
      <th>198549</th>
      <td>999919</td>
      <td>12345</td>
      <td>1.962494e+09</td>
      <td>2018-05-03</td>
      <td>G0009</td>
      <td>Z23</td>
      <td>19.91</td>
      <td>1947-01-28</td>
      <td>F</td>
      <td>2018</td>
      <td>71</td>
    </tr>
    <tr>
      <th>198550</th>
      <td>999959</td>
      <td>11445</td>
      <td>1.548250e+09</td>
      <td>2016-09-24</td>
      <td>66984</td>
      <td>H2512</td>
      <td>838.19</td>
      <td>1945-04-03</td>
      <td>F</td>
      <td>2016</td>
      <td>71</td>
    </tr>
    <tr>
      <th>198551</th>
      <td>999959</td>
      <td>11445</td>
      <td>1.548250e+09</td>
      <td>2016-09-24</td>
      <td>G8918</td>
      <td>H2512</td>
      <td>0.00</td>
      <td>1945-04-03</td>
      <td>F</td>
      <td>2016</td>
      <td>71</td>
    </tr>
  </tbody>
</table>
<p>198552 rows × 11 columns</p>
</div>



### 0.4 Analyze the top 100 HCPCS/CPT codes

### 0.4.1 Group by HCPCS/CPT codes and count the number of unique claims IDs in descending order, and take first 100 codes with the most number of claims


```python
claim_count_per_hcpcs_df = medicare_df.groupby('hcpcs_code').agg(
    uniq_clm_cnt=('claim_id', 'nunique')
).sort_values('uniq_clm_cnt', ascending=False)

claim_count_per_hcpcs_top100_df = claim_count_per_hcpcs_df.head(100)
claim_count_per_hcpcs_top100_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uniq_clm_cnt</th>
    </tr>
    <tr>
      <th>hcpcs_code</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36415</th>
      <td>8188</td>
    </tr>
    <tr>
      <th>99214</th>
      <td>7796</td>
    </tr>
    <tr>
      <th>99213</th>
      <td>6600</td>
    </tr>
    <tr>
      <th>80053</th>
      <td>5087</td>
    </tr>
    <tr>
      <th>85025</th>
      <td>4911</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>74176</th>
      <td>336</td>
    </tr>
    <tr>
      <th>99222</th>
      <td>332</td>
    </tr>
    <tr>
      <th>G0283</th>
      <td>324</td>
    </tr>
    <tr>
      <th>11721</th>
      <td>315</td>
    </tr>
    <tr>
      <th>84481</th>
      <td>314</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 1 columns</p>
</div>



### 0.4.2 Look up descriptions of the codes online and categorize them into broader medical fields/activities

Please note the categories might not be medically/officially accurate and are only for educational purposes.


```python
hcpcs_code_100_df = pd.DataFrame({
    'hcpcs_code': [
        '36415', '99214', '99213', '80053', '85025', 
        '80061', '84443', '83036', '80048', '93010', 
        '81001', 'G8427', '93000', '88305', '82306', 
        'G0463', '87086', '99232', 'G0008', '85610', 
        '93306', '1036F', '84439', 'A0425', '99285', 
        '81003', '92014', '93005', '71020', '85027', 
        '70450', '82607', '71045', '99204', '97110', 
        '99212', 'G0202', '77063', '83735', 'G9637', 
        '99203', '71010', '90662', '99284', '71046', 
        'G0439', '82570', '84484', '84153', '17000', 
        '97140', '78452', '99215', '87186', '82043', 
        '77067', '87088', '99233', '4040F', '96372', 
        '82550', '83540', 'G8907', 'A0427', '99223', 
        'Q9967', '92012', '84550', '81002', 'G8918', 
        '92134', '82565', '82728', '17003', '83550', 
        '77080', '87077', 'A9270', '7025F', 'G0471', 
        '74177', '92015', '85652', '98941', '80076', 
        'G8420', '82746', '93880', '77052', '11100', 
        'G9551', '86140', 'G0009', '83880', '66984', 
        '74176', '99222', 'G0283', '11721', '84481'
    ],
    'description': [
        'Venous Procedures','Established Patient Office or Other Outpatient Services','Established Patient Office or Other Outpatient Services','Organ or Disease Oriented Panels','Blood count', 
        'Organ or Disease Oriented Panels','Chemistry Procedures','Hemoglobin','Organ or Disease Oriented Panels','Electrocardiogram, routine ECG with at least 12 leads', 
        'Urinalysis, by dip stick or tablet reagent for bilirubin, glucose, hemoglobin, ketones, leukocytes, nitrite, pH, protein, specific gravity, urobilinogen, any number of these constituents','Eligible clinician attests to documenting in the medical record they obtained, updated, or reviewed the patient\'s current medications','Electrocardiogram, routine ECG with at least 12 leads','Surgical pathology, gross and microscopic examination','Vitamin D', 
        'Hospital outpatient clinic visit for assessment and management of a patient','Culture, bacterial','Subsequent Hospital Inpatient or Observation Care','Administration of influenza virus vaccine','Prothrombin time', 
        'Echocardiography, transthoracic, real-time with image documentation (2D), includes M-mode recording, when performed','Patient History','Thyroxine','Ground mileage, per statute mile','Emergency department visit for the evaluation and management of a patient, which requires a medically appropriate history', 
        'Urinalysis, by dip stick or tablet reagent for bilirubin, glucose, hemoglobin, ketones, leukocytes, nitrite, pH, protein, specific gravity, urobilinogen, any number of these constituents','Ophthalmological services: medical examination and evaluation, with initiation or continuation of diagnostic and treatment program','Electrocardiogram, routine ECG with at least 12 leads','DELETED','Blood count',
        'Computed tomography, head or brain','Cyanocobalamin (Vitamin B-12)','Radiologic examination, chest','New Patient Office or Other Outpatient Services','Therapeutic procedure, 1 or more areas, each 15 minutes',
        'Established Patient Office or Other Outpatient Services','Screening mammography, bilateral (2-view study of each breast), including computer-aided detection (cad) when performed','Breast, Mammography','Chemistry Procedures','Final reports with documentation of one or more dose reduction techniques (e.g., automated exposure control, adjustment of the ma and/or kv according to patient size, use of iterative reconstruction technique)', 
        'New Patient Office or Other Outpatient Services','DELETED','Influenza virus vaccine','Emergency department visit for the evaluation and management of a patient, which requires a medically appropriate history','Radiologic examination, chest',
        'Annual wellness visit, includes a personalized prevention plan of service (pps), subsequent visit','Creatinine','Chemistry Procedures','Prostate specific antigen (PSA)','Destruction (eg, laser surgery, electrosurgery, cryosurgery, chemosurgery, surgical curettement), premalignant lesions (eg, actinic keratoses)',
        'Therapeutic Procedures','Myocardial perfusion imaging, tomographic (SPECT) (including attenuation correction, qualitative or quantitative wall motion, ejection fraction by first pass or gated technique, additional quantification, when performed)','Established Patient Office or Other Outpatient Services','Susceptibility studies, antimicrobial agent','Albumin',
        'Breast, Mammography','Culture, bacterial','Subsequent Hospital Inpatient or Observation Care','Therapeutic, Preventive or Other Interventions','Therapeutic, prophylactic, or diagnostic injection (specify substance or drug)',
        'Creatine kinase (CK), (CPK)','Chemistry Procedures','Patient documented not to have experienced any of the following events: a burn prior to discharge; a fall within the facility; wrong site/side/patient/procedure/implant event; or a hospital transfer or hospital admission upon discharge from the facility','Ambulance service, advanced life support, emergency transport, level 1 (als 1 - emergency)','New or Established Patient',
        'Low osmolar contrast material, 300-399 mg/ml iodine concentration, per ml','Ophthalmological services: medical examination and evaluation, with initiation or continuation of diagnostic and treatment program','Uric acid','Urinalysis, by dip stick or tablet reagent for bilirubin, glucose, hemoglobin, ketones, leukocytes, nitrite, pH, protein, specific gravity, urobilinogen, any number of these constituents','Patient without preoperative order for iv antibiotic surgical site infection (ssi) prophylaxis',
        'Scanning computerized ophthalmic diagnostic imaging, anterior segment, with interpretation and report','Creatinine','Chemistry Procedures','Destruction (eg, laser surgery, electrosurgery, cryosurgery, chemosurgery, surgical curettement), premalignant lesions (eg, actinic keratoses)','Chemistry Procedures',
        'Dual-energy X-ray absorptiometry (DXA), bone density study, 1 or more sites','Culture, bacterial','Non-covered item or service','Structural Measures','Collection of venous blood by venipuncture or urine sample by catheterization from an individual in a skilled nursing facility (snf) or by a laboratory on behalf of a home health agency (hha)',
        'Computed tomography, abdomen and pelvis','Special Ophthalmological Services and Procedures','Sedimentation rate, erythrocyte','Chiropractic manipulative treatment (CMT)','Organ or Disease Oriented Panels',
        'Bmi is documented within normal parameters and no follow-up plan is required','Folic acid','Duplex scan of extracranial arteries','DELETED','Biopsy of skin, subcutaneous tissue and/or mucous membrane (including simple closure), unless otherwise listed',
        'Final reports for imaging studies without an incidentally found lesion noted','C-reactive protein','Administration of pneumococcal vaccine','Chemistry Procedures','Intraocular Lens Procedures',
        'Computed tomography, abdomen and pelvis','New or Established Patient','Electrical stimulation (unattended), to one or more areas for indication(s) other than wound care, as part of a therapy plan of care','Debridement of nail(s) by any method(s)','Triiodothyronine T3'
    ],
    'category': [
        'Cardiac','Administrative','Administrative','Panels','Blood test',
        'Panels','Chemistry','Blood test','Panels','Cardiac',
        'Urinalysis','Administrative','Cardiac','Pathology','Blood test',
        'Administrative','Pathology','Administrative','Vaccine','Liver',
        'Cardiac','Administrative','Blood test','Others','Administrative',
        'Urinalysis','Ophthalmology','Cardiac','Others','Blood test',
        'Tomography','Blood test','Radiology','Administrative','Therapy',
        'Administrative','Mammography','Mammography','Chemistry','Administrative',
        'Administrative','Others','Vaccine','Administrative','Radiology',
        'Administrative','Blood test','Chemistry','Blood test','Destructive surgical procedures',
        'Therapy','Cardiac','Administrative','Pathology','Blood test',
        'Mammography','Pathology','Administrative','Therapy','Therapy',
        'Blood test','Chemistry','Administrative','Administrative','Administrative',
        'Radiology','Ophthalmology','Urinalysis','Urinalysis','Administrative',
        'Ophthalmology','Blood test','Chemistry','Destructive surgical procedures','Chemistry',
        'Radiology','Pathology','Others','Administrative','Pathology',
        'Tomography','Ophthalmology','Blood test','Chiropractic','Panels',
        'Administrative','Blood test','Radiology','Others','Pathology',
        'Administrative','Blood test','Vaccine','Chemistry','Ophthalmology',
        'Tomography','Administrative','Therapy','Destructive surgical procedures','Blood test'
    ]
})
hcpcs_code_100_df.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hcpcs_code</th>
      <th>description</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36415</td>
      <td>Venous Procedures</td>
      <td>Cardiac</td>
    </tr>
    <tr>
      <th>1</th>
      <td>99214</td>
      <td>Established Patient Office or Other Outpatient...</td>
      <td>Administrative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>99213</td>
      <td>Established Patient Office or Other Outpatient...</td>
      <td>Administrative</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80053</td>
      <td>Organ or Disease Oriented Panels</td>
      <td>Panels</td>
    </tr>
    <tr>
      <th>4</th>
      <td>85025</td>
      <td>Blood count</td>
      <td>Blood test</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>74176</td>
      <td>Computed tomography, abdomen and pelvis</td>
      <td>Tomography</td>
    </tr>
    <tr>
      <th>96</th>
      <td>99222</td>
      <td>New or Established Patient</td>
      <td>Administrative</td>
    </tr>
    <tr>
      <th>97</th>
      <td>G0283</td>
      <td>Electrical stimulation (unattended), to one or...</td>
      <td>Therapy</td>
    </tr>
    <tr>
      <th>98</th>
      <td>11721</td>
      <td>Debridement of nail(s) by any method(s)</td>
      <td>Destructive surgical procedures</td>
    </tr>
    <tr>
      <th>99</th>
      <td>84481</td>
      <td>Triiodothyronine T3</td>
      <td>Blood test</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>



# Start of Mid Term Exam


```python
## Keeping the original medicare_df and hcpcs_code_100_df for posterity
```


```python
original_medicare_df = medicare_df
```


```python
unfiltered_hcpcs_code_100_df = hcpcs_code_100_df
```

# Removing All NULL Values

Instead of having to drop null categories one by one, I dropped all null values


```python
medicare_df.isnull().sum()
```




    claim_id                  0
    patient_id                0
    npi_id                44806
    claim_date                0
    hcpcs_code            12993
    diagnosis_code         9783
    claim_cost            46886
    patient_birth_date        0
    patient_gender            0
    claim_year                0
    patient_age               0
    dtype: int64




```python
len(medicare_df)
```




    198552




```python
medicare_df = medicare_df.dropna()
```


```python
medicare_df.isnull().sum()
```




    claim_id              0
    patient_id            0
    npi_id                0
    claim_date            0
    hcpcs_code            0
    diagnosis_code        0
    claim_cost            0
    patient_birth_date    0
    patient_gender        0
    claim_year            0
    patient_age           0
    dtype: int64




```python
len(medicare_df)
```




    148688



# Removing ClaimID duplicates


### Ensuring no two FULL rows have duplicates 


```python
medicare_df.duplicated().sum()
```




    0



## checking and dropping duplicates of ClaimID


```python
medicare_df['claim_id'].duplicated().sum()
```




    65507




```python
medicare_df = medicare_df.drop_duplicates(subset=['claim_id'])
```


```python
medicare_df['claim_id'].duplicated().sum()
```




    0




```python
len(medicare_df)
```




    83181



## Excluding non-procedural categories (Administrative & Others)


```python
hcpcs_code_100_df = hcpcs_code_100_df[(hcpcs_code_100_df['category'] != 'Administrative') & (hcpcs_code_100_df['category'] != 'Others')]
```


```python
hcpcs_code_100_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hcpcs_code</th>
      <th>description</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36415</td>
      <td>Venous Procedures</td>
      <td>Cardiac</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80053</td>
      <td>Organ or Disease Oriented Panels</td>
      <td>Panels</td>
    </tr>
    <tr>
      <th>4</th>
      <td>85025</td>
      <td>Blood count</td>
      <td>Blood test</td>
    </tr>
    <tr>
      <th>5</th>
      <td>80061</td>
      <td>Organ or Disease Oriented Panels</td>
      <td>Panels</td>
    </tr>
    <tr>
      <th>6</th>
      <td>84443</td>
      <td>Chemistry Procedures</td>
      <td>Chemistry</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>94</th>
      <td>66984</td>
      <td>Intraocular Lens Procedures</td>
      <td>Ophthalmology</td>
    </tr>
    <tr>
      <th>95</th>
      <td>74176</td>
      <td>Computed tomography, abdomen and pelvis</td>
      <td>Tomography</td>
    </tr>
    <tr>
      <th>97</th>
      <td>G0283</td>
      <td>Electrical stimulation (unattended), to one or...</td>
      <td>Therapy</td>
    </tr>
    <tr>
      <th>98</th>
      <td>11721</td>
      <td>Debridement of nail(s) by any method(s)</td>
      <td>Destructive surgical procedures</td>
    </tr>
    <tr>
      <th>99</th>
      <td>84481</td>
      <td>Triiodothyronine T3</td>
      <td>Blood test</td>
    </tr>
  </tbody>
</table>
<p>72 rows × 3 columns</p>
</div>




```python
len(medicare_df[medicare_df['hcpcs_code'] == '78452'])
```




    413



### Making NPI_ID into String for easier viewing of NPI_ID


```python
medicare_df['npi_id'] = medicare_df['npi_id'].astype(str)
```

    /var/folders/zk/ffvbnsts2dg8tf_rqkmdm36m0000gn/T/ipykernel_63105/3764644550.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      medicare_df['npi_id'] = medicare_df['npi_id'].astype(str)


## Q1. Based on the trends for the share of CVM claims as a percentage of total claims over the years 2016 through 2018, what are some business insights you can gather? What are some additional analyses you could do based on these trends? 


```python
len(medicare_df['claim_id'])
```




    83181



### Removing years other than 2016, 2017, 2018


```python
medicare_df = medicare_df[medicare_df['claim_year'].isin(['2016','2017', '2018'])]
```

###### Checking


```python
medicare_df[medicare_df['claim_year'] == '2014']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>diagnosis_code</th>
      <th>claim_cost</th>
      <th>patient_birth_date</th>
      <th>patient_gender</th>
      <th>claim_year</th>
      <th>patient_age</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
cardiac_hcpcs_codes = hcpcs_code_100_df[hcpcs_code_100_df['category'] == 'Cardiac']['hcpcs_code']
```


```python
cardiac_hcpcs_codes
```




    0     36415
    9     93010
    12    93000
    20    93306
    27    93005
    51    78452
    Name: hcpcs_code, dtype: object




```python
cardiac_medicare = medicare_df[medicare_df['hcpcs_code'].isin(cardiac_hcpcs_codes)]
```


```python
cardiac_medicare
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>claim_id</th>
      <th>patient_id</th>
      <th>npi_id</th>
      <th>claim_date</th>
      <th>hcpcs_code</th>
      <th>diagnosis_code</th>
      <th>claim_cost</th>
      <th>patient_birth_date</th>
      <th>patient_gender</th>
      <th>claim_year</th>
      <th>patient_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>1011605</td>
      <td>10163</td>
      <td>1578545943.0</td>
      <td>2018-01-02</td>
      <td>36415</td>
      <td>C439</td>
      <td>49.51</td>
      <td>1944-12-25</td>
      <td>M</td>
      <td>2018</td>
      <td>74</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1016102</td>
      <td>10017</td>
      <td>1306804737.0</td>
      <td>2018-01-25</td>
      <td>36415</td>
      <td>Z79899</td>
      <td>78.46</td>
      <td>1984-11-23</td>
      <td>M</td>
      <td>2018</td>
      <td>34</td>
    </tr>
    <tr>
      <th>34</th>
      <td>104681</td>
      <td>10174</td>
      <td>1114949963.0</td>
      <td>2016-11-19</td>
      <td>36415</td>
      <td>J441</td>
      <td>71.49</td>
      <td>1954-04-13</td>
      <td>M</td>
      <td>2016</td>
      <td>62</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1049025</td>
      <td>10200</td>
      <td>1538126966.0</td>
      <td>2017-02-21</td>
      <td>36415</td>
      <td>Z7901</td>
      <td>12.12</td>
      <td>1931-04-20</td>
      <td>M</td>
      <td>2017</td>
      <td>86</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1054205</td>
      <td>10200</td>
      <td>1649262882.0</td>
      <td>2016-02-03</td>
      <td>36415</td>
      <td>I4891</td>
      <td>11.19</td>
      <td>1931-04-20</td>
      <td>M</td>
      <td>2016</td>
      <td>85</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>198436</th>
      <td>998390</td>
      <td>12634</td>
      <td>1104010131.0</td>
      <td>2017-12-15</td>
      <td>93306</td>
      <td>R011</td>
      <td>222.72</td>
      <td>1925-01-09</td>
      <td>F</td>
      <td>2017</td>
      <td>92</td>
    </tr>
    <tr>
      <th>198451</th>
      <td>998607</td>
      <td>10000</td>
      <td>1881646099.0</td>
      <td>2016-08-17</td>
      <td>36415</td>
      <td>Z13220</td>
      <td>3.00</td>
      <td>1952-06-20</td>
      <td>F</td>
      <td>2016</td>
      <td>64</td>
    </tr>
    <tr>
      <th>198471</th>
      <td>998886</td>
      <td>11923</td>
      <td>1326018839.0</td>
      <td>2016-02-25</td>
      <td>93306</td>
      <td>R079</td>
      <td>170.16</td>
      <td>1947-01-05</td>
      <td>F</td>
      <td>2016</td>
      <td>69</td>
    </tr>
    <tr>
      <th>198478</th>
      <td>998994</td>
      <td>1262</td>
      <td>1932166386.0</td>
      <td>2018-06-14</td>
      <td>36415</td>
      <td>I10</td>
      <td>3.00</td>
      <td>1948-07-25</td>
      <td>M</td>
      <td>2018</td>
      <td>70</td>
    </tr>
    <tr>
      <th>198526</th>
      <td>999618</td>
      <td>11357</td>
      <td>1962518530.0</td>
      <td>2017-11-12</td>
      <td>93010</td>
      <td>I10</td>
      <td>8.81</td>
      <td>1951-10-10</td>
      <td>F</td>
      <td>2017</td>
      <td>66</td>
    </tr>
  </tbody>
</table>
<p>8049 rows × 11 columns</p>
</div>



### Percentage of CVM claims


```python
len(cardiac_medicare)
```




    8049




```python
len(medicare_df)
```




    83181




```python
print((len(cardiac_medicare)/len(medicare_df)) * 100, '%')
```

    9.676488621199553 %



```python
total_counts = medicare_df['claim_year'].value_counts().sort_index()
cardiac_counts = cardiac_medicare['claim_year'].value_counts().sort_index()
non_cardiac_counts = total_counts - cardiac_counts.reindex(total_counts.index).fillna(0)
percentage_increase = cardiac_counts.pct_change() * 100
percentage_increase = percentage_increase.round(2)

print("Percentage Increase of Cardiac Claims per Year:")
print(percentage_increase)
```

    Percentage Increase of Cardiac Claims per Year:
    claim_year
    2016      NaN
    2017    12.25
    2018    20.25
    Name: count, dtype: float64



```python
plt.bar(total_counts.index, cardiac_counts, label = 'Cardiac')
plt.bar(total_counts.index, non_cardiac_counts, bottom= cardiac_counts, label = 'non-Cardiac')

plt.xlabel('Claim Year')
plt.ylabel('Claims Count')
plt.title("Comparison of Cardiac vs. Non-Cardiac Claims by Year")

plt.show()
```


    
![png](output_163_0.png)
    



```python

```

## Q2. Evaluate the HCP behavior in context of claim volume from 2016-2018. How many HCPs are submitting 1 CVM claim; how many HCPs are associated with more than 10 claims, etc.? Once you perform this analysis, explain how this trend can influence the sales force deployment. That is, how would you segment the HCPs and how would you allocate In-Person (sales force) vs Non-Personal Promotions (NPP, i.e. Emails, Social Media, Digital etc.) efforts?


```python
medicare_df['npi_id'] = medicare_df['npi_id'].astype(str)
```


```python
medicare_df['npi_id'].nunique()
```




    46162




```python
HCP_claim_count = medicare_df.sort_index()
```


```python
HCP_claim_count['npi_id'].value_counts()
```




    npi_id
    1538144910.0    988
    1619913449.0    661
    1063497451.0    598
    1659352276.0    594
    1518903350.0    513
                   ... 
    1508855529.0      1
    1003987967.0      1
    1124061361.0      1
    1164645016.0      1
    1881665362.0      1
    Name: count, Length: 46162, dtype: int64




```python
cardiac_medicare['npi_id'].value_counts()
```




    npi_id
    1619913449.0    94
    1346233277.0    64
    1326104613.0    57
    1245307818.0    54
    1932145778.0    51
                    ..
    1356352025.0     1
    1992719421.0     1
    1023086147.0     1
    1518963362.0     1
    1962518530.0     1
    Name: count, Length: 5599, dtype: int64




```python
disease_aware = (cardiac_medicare['npi_id'].value_counts() == 1).sum()
trialist = ((cardiac_medicare['npi_id'].value_counts() >= 2 ) & (cardiac_medicare['npi_id'].value_counts() <= 4)).sum()
rising_stars = ((cardiac_medicare['npi_id'].value_counts() >= 5) & (cardiac_medicare['npi_id'].value_counts() <= 9)).sum()
high_volume_stars = (cardiac_medicare['npi_id'].value_counts() >= 10).sum()
print('disease_aware:', disease_aware)
print('trialist:', trialist)
print('rising_stars:', rising_stars)
print('high_volume_stars', high_volume_stars)
```

    disease_aware: 4911
    trialist: 579
    rising_stars: 50
    high_volume_stars 59



```python
cardiac_medicare['claim_date'] = pd.to_datetime(cardiac_medicare['claim_date'])

cardiac_medicare['claim_year'] = cardiac_medicare['claim_date'].dt.year
cardiac_medicare['claim_quarter'] = cardiac_medicare['claim_date'].dt.to_period('Q')
```

    /var/folders/zk/ffvbnsts2dg8tf_rqkmdm36m0000gn/T/ipykernel_63105/2964846158.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      cardiac_medicare['claim_date'] = pd.to_datetime(cardiac_medicare['claim_date'])
    /var/folders/zk/ffvbnsts2dg8tf_rqkmdm36m0000gn/T/ipykernel_63105/2964846158.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      cardiac_medicare['claim_year'] = cardiac_medicare['claim_date'].dt.year
    /var/folders/zk/ffvbnsts2dg8tf_rqkmdm36m0000gn/T/ipykernel_63105/2964846158.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      cardiac_medicare['claim_quarter'] = cardiac_medicare['claim_date'].dt.to_period('Q')



```python
segments = ['Disease Aware', 'Trialists', 'Rising Stars', 'High-Volume Prescribers']
segment_counts_per_year = pd.DataFrame(columns=segments, index=[2016, 2017, 2018])

for year in [2016, 2017, 2018]:
    cardiac_medicare_year = cardiac_medicare[cardiac_medicare['claim_year'] == year]
    
    hcp_claim_counts = cardiac_medicare_year.groupby('npi_id')['claim_id'].nunique()
    
    disease_aware = (hcp_claim_counts == 1).sum()
    trialists = ((hcp_claim_counts >= 2) & (hcp_claim_counts <= 4)).sum()
    rising_stars = ((hcp_claim_counts >= 5) & (hcp_claim_counts <= 9)).sum()
    high_volume_prescribers = (hcp_claim_counts >= 10).sum()
    
    segment_counts_per_year.loc[year, 'Disease Aware'] = disease_aware
    segment_counts_per_year.loc[year, 'Trialists'] = trialists
    segment_counts_per_year.loc[year, 'Rising Stars'] = rising_stars
    segment_counts_per_year.loc[year, 'High-Volume Prescribers'] = high_volume_prescribers
```


```python
print("Counts per Segment per Year:")
print(segment_counts_per_year)
```

    Counts per Segment per Year:
         Disease Aware Trialists Rising Stars High-Volume Prescribers
    2016          1654       109           25                      19
    2017          1841       149           31                      14
    2018          2106       174           20                      27



```python
segment_counts_per_year = segment_counts_per_year.astype(int)

segment_counts_per_year.plot(kind='bar', stacked=True, figsize=(10,7), color=['#f4a261', '#e76f51', '#2a9d8f', '#264653'])

plt.xlabel('Year')
plt.ylabel('Number of Providers')
plt.title('HCP Segmentation by CVM Claims per Year')
plt.legend(title='Segments', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```


    
![png](output_175_0.png)
    


# Extra Experimentation: Trying to figure out how much does each category contributes in claim cost


```python
cardiac_medicare['claim_cost'] = pd.to_numeric(cardiac_medicare['claim_cost'], errors='coerce')

hcp_claim_costs = cardiac_medicare.groupby(['claim_year', 'npi_id'])['claim_cost'].sum().reset_index()
```

    /var/folders/zk/ffvbnsts2dg8tf_rqkmdm36m0000gn/T/ipykernel_63105/3064055160.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      cardiac_medicare['claim_cost'] = pd.to_numeric(cardiac_medicare['claim_cost'], errors='coerce')



```python
segment_claim_costs_per_year = pd.DataFrame(columns=segments, index=[2016, 2017, 2018])

for year in [2016, 2017, 2018]:
    cardiac_medicare_year = cardiac_medicare[cardiac_medicare['claim_year'] == year]
    
    hcp_claim_counts = cardiac_medicare_year.groupby('npi_id')['claim_id'].nunique()
    
    hcp_data = hcp_claim_counts.to_frame('claim_count').reset_index()
    hcp_costs = hcp_claim_costs[hcp_claim_costs['claim_year'] == year]
    
    hcp_data = pd.merge(hcp_data, hcp_costs[['npi_id', 'claim_cost']], on='npi_id', how='left')
    
    def assign_segment(count):
        if count == 1:
            return 'Disease Aware'
        elif 2 <= count <= 4:
            return 'Trialists'
        elif 5 <= count <= 9:
            return 'Rising Stars'
        else:
            return 'High-Volume Prescribers'
    
    hcp_data['Segment'] = hcp_data['claim_count'].apply(assign_segment)
    
    segment_costs = hcp_data.groupby('Segment')['claim_cost'].sum()
    
    for segment in segments:
        segment_claim_costs_per_year.loc[year, segment] = segment_costs.get(segment, 0)
```


```python
segment_claim_costs_per_year = segment_claim_costs_per_year.astype(float)
```


```python
print("Claim Costs per Segment per Year:")
print(segment_claim_costs_per_year)
```

    Claim Costs per Segment per Year:
          Disease Aware  Trialists  Rising Stars  High-Volume Prescribers
    2016      161116.41   10785.10        953.43                    708.0
    2017      183489.71   20883.69       5941.42                    651.0
    2018      208810.39   33208.92       1359.16                   1362.0



```python
segment_claim_costs_per_year.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 7),
    color=['#f4a261', '#e76f51', '#2a9d8f', '#264653']
)

plt.xlabel('Year')
plt.ylabel('Total Claim Cost')
plt.title('Contribution of Each Segment to Total Claim Cost per Year')
plt.legend(title='Segments', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```


    
![png](output_181_0.png)
    


## Q3. Evaluate the Patient Age demographics in the context of claim volume from 2016-2018. Bucket the patients into groups based on their age and explain the trends. How would you position the Marketing Budgets and the Promotions with respect to the changing landscape of the CVM claims and the respective patient segments? 


```python
medicare_df['patient_age'] = pd.to_numeric(medicare_df['patient_age'])
```


```python
segment_one = ((medicare_df['patient_age'] >= 18) & (medicare_df['patient_age'] <= 59)).sum()
segment_two = ((medicare_df['patient_age'] >= 60) & (medicare_df['patient_age'] <= 69)).sum()
segment_three = ((medicare_df['patient_age'] >= 70) & (medicare_df['patient_age'] <= 79)).sum()
segment_four = (medicare_df['patient_age'] >= 80).sum()
segment_five = (medicare_df['patient_age'] <= 18).sum()
```


```python
print('segment_one: ',segment_one)
print('segment_two: ',segment_two)
print('segment_three: ',segment_three)
print('segment_four: ',segment_four)
print('segment_five: ',segment_five)
```

    segment_one:  8835
    segment_two:  26379
    segment_three:  27774
    segment_four:  18654
    segment_five:  1539



```python
def assign_age_segment(age):
    if 18 <= age <= 59:
        return '18-59'
    elif 60 <= age <= 69:
        return '60-69'
    elif 70 <= age <= 79:
        return '70-79'
    elif age >= 80:
        return '80+'
    else:
        return 'Under 18'

medicare_df['age_segment'] = medicare_df['patient_age'].apply(assign_age_segment)
```


```python
claims_by_year_segment = medicare_df.groupby(['claim_year', 'age_segment']).size().reset_index(name='claim_count')
claims_pivot = claims_by_year_segment.pivot(index='claim_year', columns='age_segment', values='claim_count').fillna(0)
print(claims_pivot)
```

    age_segment  18-59  60-69  70-79   80+  Under 18
    claim_year                                      
    2016          2999   8191   7195  5365       583
    2017          2908   8958   9087  5756       526
    2018          2928   9230  11492  7533       430



```python
claims_pct_change = claims_pivot.pct_change().fillna(0) * 100

claims_pct_change = claims_pct_change.round(2)

print(claims_pct_change)
```

    age_segment  18-59  60-69  70-79    80+  Under 18
    claim_year                                       
    2016          0.00   0.00   0.00   0.00      0.00
    2017         -3.03   9.36  26.30   7.29     -9.78
    2018          0.69   3.04  26.47  30.87    -18.25



```python
claims_pivot.plot(kind='bar', stacked=True, figsize=(10,7))

plt.xlabel('Year')
plt.ylabel('Number of Claims')
plt.title('Number of Claims by Age Segment (2016-2018)')
plt.legend(title='Age Segment', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
```


    
![png](output_189_0.png)
    



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
