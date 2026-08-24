# 🛒 Supermarket Customer Segmentation — Client Summary Report

**Prepared for:** Stakeholder Review  
**Date:** August 2026  
**Method:** K-Means Clustering (Unsupervised Learning)  
**Data:** 1,000 supermarket transactions

---

## 📋 Quick Overview

| Metric | Value |
|--------|-------|
| **Total Transactions** | 1,000 |
| **Customers Segmented** | 2 groups |
| **Silhouette Score** | 0.2519 (moderate separation) |
| **Project Duration** | Jan–Mar 2019 data analysis |

---

## 🎯 Business Problem

**Goal:** Discover natural customer groups to enable targeted marketing and improved retention.

**Question:** "Can we automatically group customers by purchasing behavior without prior labels?"

---

## 🧠 Results — Two Customer Segments Discovered

| Segment | Size | % of Customers | Key Trait |
|---------|------|---------------|-----------|
| **Segment 0** | 343 customers | **34%** | **High-Value Shoppers** — large baskets, high spend |
| **Segment 1** | 657 customers | **66%** | **Budget Shoppers** — smaller baskets, frequent purchases |

---

## 💰 Key Financial Metrics

| Metric | Segment 0 (High-Value) | Segment 1 (Budget) |
|--------|------------------------|--------------------|
| **Avg Transaction Total** | **$611.50** | $172.33 |
| **Avg Items per Transaction** | 7.71 | 4.36 |
| **Avg Unit Price** | $77.08 | $44.49 |
| **Avg Customer Rating** | 6.90/10 | 7.01/10 |
| **Gross Income/Transaction** | $29.12 | $8.21 |

---

## 🎯 Business Recommendations

### **For High-Value Shoppers (34%)**
- ✅ Premium product offers (Home & lifestyle, Sports & travel)
- ✅ Bulk-purchase discounts
- ✅ Loyalty program enhancements
- ✅ Personalized product recommendations

### **For Budget Shoppers (66%)**
- ✅ Entry-level product promotions
- ✅ Bundle deals ("Buy 3, Get 10% off")
- ✅ Frequency incentives ("Shop 5 times, Get $10 off")
- ✅ Cross-selling accessories & complementary products

### **Overall Insight**
- **34% of customers generate disproportionate revenue**
- **Budget shoppers rate the store slightly higher** (7.01 vs 6.90)
- **Product preferences differ** significantly between segments

---

## 📊 Technical Notes

- **Method:** K-Means Clustering (k=2 selected via Silhouette Score)
- **Features:** 30 engineered features from 17 original columns
- **Validation:** Silhouette score 0.2519 — moderate cluster separation
- **Data:** 1,000 transactions, Jan–Mar 2019
- **Frontend:** Streamlit dashboard with Dark/Light mode toggle

---

## 🎨 Frontend Dashboard

**Access:** http://localhost:8501

**Features:**
- 🌓 **Dark/Light mode toggle** in sidebar
- 📊 Segment distribution pie chart
- 📈 Average spend by segment bar chart
- 📊 Cluster vs Member/Normal heatmap
- 📦 Detailed segment profile cards
- 📊 Product line preferences
- 📊 Scatter and histogram plots
- 📊 Downloadable results

---

## 📞 Next Steps (Optional)

1. **RFM Analysis** — Add Recency from transaction dates
2. **Propensity modeling** — Predict customers likely to upgrade from Budget to High-Value
3. **A/B testing** — Validate marketing campaigns per segment
4. **Real-time segmentation** — Integrate into point-of-sale systems
5. **Seasonal analysis** — How segments behave differently by season

---

**Report Type:** Client Summary  
**Data Period:** January–March 2019  
**Method:** K-Means Clustering, k=2  
**Status:** ✅ Complete and Ready for Review

---
*This summary provides a high-level overview of the clustering analysis. Detailed cluster profiles, technical validation, and business recommendations are available in the full project documentation.*