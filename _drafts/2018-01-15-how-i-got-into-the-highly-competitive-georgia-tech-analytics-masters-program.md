---
layout: post
category : Machine Learning
tagline: "my first end-to-end machine learning project"
tags : [public-health,machine-learning,python]
---
{% include JB/setup %}



#### *Predicting Global Supply Chain Outcomes for Essential HIV Medicines using Machine Learning Techniques.*

This is an overview of my final original capstore project for Udacity Machine Learning Engineer Nanodegree program. It was my first end-to-end machine learning project. You can find the full project report, data, code and notebooks on <a href="https://github.com/tichmangono/capstone_project">this github repo</a>

**Key Question:** Can we use procurement transaction data to predict whether a delivery is delayed and estimating the length of the delay
Main Data Source: From The Website: https://data.pepfar.net/additionalData. Procurement transaction data from the Supply Chain Management System (SCMS), administered by the United States Agency for International Development (USAID), provides information on health commodities, prices, and delivery destinations.

### Synopsis of Findings
A combined “classification-then-regression” machine learning model can avoid the public health and economic costs associated with delayed deliveries of HIV medicines.
 An ensemble classification algorithm, Extra Trees, is able to detect 1 in 2 delayed item deliveries. This is a significant improvement from a null hypothesis model which would 
 detect only 1 in 9 delayed items and a considerable improvement from benchmarked Random Forest classification algorithm which catches 1 in 3 delayed items. Once delayed items are 
 identified, an Extra Trees regression algorithm can predict the length of delay to within 12 days (RMSE) with an R-Squared of 0.86, which is similar to the benchmarked Random Forest
 regression performance. So, while there was no significant improvement in the regression part, the combined classification-then-regression model for Extra Trees does significantly 
 better than the benchmark.

- **Background:** only 19.5M people out of the ~37M people living with HIV are getting the essential medicines they need. Supply of these essential medicines is critical. Recent evidence https://www.devex.com/news/exclusive-documents-reveal-largest-usaid-health-project-in-trouble-90933 suggests that supply chain for major global programs has worsened after recent changes in supply chain managing organizations. See also chart displayed above.  


- **Problem Statement:** Such significant supply chain delays in delivery of medicines disrupt treatment and can lead to loss of life and ultimately increases supply chain costs. The goal is to machine learning to determine when and which products are likely to be delayed, as well quantify the extent of the delay. 



- **Datasets & Inputs:** Publicly available ***supply chain data*** from US, The President;s Emergency Plan for AIDS Relief (PEPFAR) https://data.pepfar.net/additionalData; ***Logistics Performance Index*** data from the World Bank https://lpi.worldbank.org/international/global?sort=asc&order=Infrastructure;   ***Fragile State Index*** data from Fund for Peace data http://fundforpeace.org/fsi/excel/; and finally Factory location and continent from the googlemaps API: http://maps.googleapis.com/maps/api/geocode/json?



- **Solution Statement:** A combined "classification-then-regression" machine learning algorithm where the classification algorithm predicts whether a particular product will be delayed or not and then the regression algorithm will predict the length of delay on the subset of the data which the classification predicts will be delayed. This mimicks a streamlined, prioritized decision process of a supply chain manager. 



- **Benchmark Model:** Default SciKit-Learn RandomForestClassifier and RandomForestRegressor will be used as benchmarks/baseline. Several models will then be explored to improve over the benchmark including other ensemble and tree-based models, Support-Vector Machines (SVM), XGBoost.  



- **Evaluation Metrics:** Recall and F1-score will be used for classification while R-squared and RMSE will be used for the regression part of the combined model  



### Requirements: software and libraries used
- A python file/module "my_helper_functions.py" is included in this folder with a set of my own helper functions
- The rest of the libraries can be installed using either anaconda or pip distributions
- I was running python 3.6.1 on a 64-bit windows system

#### Install/Download the following libraries and apis 
1. python 3.6.1 
2. my_helper_functions - provided in this folder. This is my own module of helper functions. It will be required to run most of the code
3. pandas
4. numpy
5. matplotlib
6. seaborn
7. time
8. datetime
9. pandas_profiling
10. pivottablejs
11. missingno
12. os
13. sklearn
14. yellowbrick
15. imblearn
16. googlemaps