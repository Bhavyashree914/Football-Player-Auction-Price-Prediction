# 🌍 COVID-19 Impact on Global Economy – Power BI Dashboard

### 📊 Interactive Data Analytics | Power BI | Data Visualization | COVID-19 Insights

---

## 🧭 Overview
This project presents a comprehensive analysis of the **global economic and demographic impact of COVID-19** using Power BI.  
It explores relationships between **population demographics, GDP per capita, median age, and life expectancy**, highlighting how the pandemic influenced global economies and population health.

The dashboard provides **interactive visuals**, **key performance indicators (KPIs)**, and **geographical insights** to help policymakers and analysts understand COVID-19’s worldwide effects.

---

## 🎯 Objectives
- Analyze the **correlation between life expectancy** and **elderly population (65+)**.
- Compare **GDP per capita**, **median age**, and **population density** across countries.
- Visualize **continent-wise disparities** in healthcare and economic indicators.
- Provide an **interactive and intuitive dashboard** for global data exploration.

---

## 🗂️ Dataset
**Source:** Cleaned and processed dataset – `covid_final_cleaned.xlsx`  
**Data Fields Include:**
- `continent`
- `location`
- `population`
- `median_age`
- `aged_65_older`
- `gdp_per_capita`
- `life_expectancy`
- `population_density`
- `hospital_beds_per_thousand`
- `total_cases`, `total_deaths`

---

## 📈 Key Metrics (KPIs)
| Metric | Formula | Description |
|---------|----------|-------------|
| **Total Population** | `SUM(population)` | Total population across all countries |
| **Average GDP per Capita** | `AVERAGE(gdp_per_capita)` | Measures global income levels |
| **Average Life Expectancy** | `AVERAGE(life_expectancy)` | Indicates population health and longevity |
| **Average Median Age** | `AVERAGE(median_age)` | Represents the age distribution globally |

---

## 💡 Visualizations
### **Page 1 – Global Overview**
- KPI Cards: Total Population, Avg. Median Age, Avg. GDP per Capita, Avg. Life Expectancy  
- Scatter Chart: Median Age vs Aged 65+ Ratio  
- Bar Chart: GDP per Capita by Location  
- Slicer: Continent/Country Filter  
- Key Insights Panel: Observations based on data trends  

### **Page 2 – Population & Health Analysis**
- Donut Chart: Aged 65+ Ratio by Continent  
- Scatter Chart: Population Density vs Life Expectancy  
- Table: Country-wise healthcare data (beds, life expectancy)

### **Page 3 – Economic Insights**
- Filled Map: GDP per Capita by Continent  
- Bar Chart: GDP vs Life Expectancy  
- Highlight Cards: Regional comparison of key metrics  

---

## 🎨 Design Highlights
- **Theme:** Professional Dark Blue & White  
- **Color Palette:**  
  - Primary Blue – `#007acc`  
  - Accent Green – `#26a69a`  
  - Neutral Gray – `#f4f4f4`  
- **Typography:** Segoe UI  
- Clean layout with balanced spacing and focused visuals  
- Clear slicers for filtering by **continent** and **location**

---

## 🧠 Insights & Findings
- Countries with **higher median age** tend to have **better healthcare systems** and **longer life expectancy**.  
- **GDP per capita** shows a strong positive correlation with **life expectancy**.  
- **Europe** and **North America** show higher aged populations (65+) but maintain high life expectancy.  
- Developing regions in **Africa** and **Asia** show lower GDP per capita and shorter life expectancy.

---

## ⚙️ Tools & Technologies
| Tool | Purpose |
|------|----------|
| **Power BI Desktop** | Data Visualization & Dashboard Creation |
| **Microsoft Excel** | Data Cleaning & Preprocessing |
| **DAX (Data Analysis Expressions)** | Creating Calculated Measures |
| **GitHub** | Version Control & Portfolio Hosting |

---

## 📂 Project Structure
