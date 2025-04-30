# Financial Inclusion in Africa - Machine Learning Project

This project explores the **Financial Inclusion in Africa** dataset, which contains demographic information and financial service usage details for approximately 33,600 individuals across East Africa. The goal is to build a machine learning model capable of predicting whether an individual is likely to have or use a bank account.

![Dataset Overview](https://i.imgur.com/UNUZ4zR.jpg)

## Context

**Financial inclusion** refers to the access that individuals and businesses have to useful and affordable financial products and services that meet their needs—such as transactions, payments, savings, credit, and insurance—provided in a responsible and sustainable manner.

## Dataset Overview

The dataset was provided as part of the Zindi "Financial Inclusion in Africa" challenge. It includes demographic, socio-economic, and geographic attributes.

## Variable Definitions

- **country**: Country where the interviewee resides.
- **year**: Year in which the survey was conducted.
- **uniqueid**: Unique identifier for each interviewee.
- **location_type**: Type of location: *Rural* or *Urban*.
- **cellphone_access**: Interviewee's cellphone access: *Yes* or *No*.
- **household_size**: Number of individuals in the household.
- **age_of_respondent**: Age of the interviewee (in years).
- **gender_of_respondent**: Gender of the interviewee: *Male* or *Female*.
- **relationship_with_head**: Relationship to head of household: *Head of Household*, *Spouse*, *Child*, *Parent*, *Other relative*, *Other non-relative*, *Don't know*.
- **marital_status**: Marital status: *Married/Living together*, *Divorced/Separated*, *Widowed*, *Single/Never Married*, *Don't know*.
- **education_level**: Highest education level: *No formal education*, *Primary*, *Secondary*, *Vocational/Specialized*, *Tertiary*, *Other/Don't know/Refused*.
- **job_type**: Job type or income source: *Farming and Fishing*, *Self-employed*, *Formally employed (Government/Private)*, *Informally employed*, *Remittance/Government dependent*, *Other/No income*, *Don't know/Refused*.

## Objective

To train a classification model that predicts the likelihood of individuals having a bank account, based on their demographic and socio-economic characteristics.

