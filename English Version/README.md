# Technical Core![算法架构图](fortext/算法架构图.drawio.png)
## 1. Feature Engineering
1)Multi-source Spatiotemporal Data Fusion: Integrated 36-dimensional spatial features, including shopping malls, transportation hubs, education, healthcare, and offices, to construct a high-granularity urban functional grid.
![成果预览](fortext/整体缩略图现有一级预测.png)
![细节图](fortext/细节图.png)
2）Proxy Estimation: Addressed the lack of real-time population density in static map data by innovatively introducing the "Proxy Variable Method." Based on public facility allocation logic, the distribution frequency of hospitals and schools was utilized as proxy indicators for population density (Proxy Variables), effectively overcoming the spatial and temporal limitations of statistical data.
## 2. Model Selection & Generalization Ability
 (L1 Regularization) Regression was selected as the final model after a cross-comparative analysis of multiple regression algorithms.
![多种模型对比](fortext/最终效果图.png)
1） Superior Robustness: The consistent performance between TrainingR² (0.5754) and Testing R²(0.5595)—with a gap < 0.02—demonstrates the model's exceptional generalization ability and effective suppression of overfitting.
![lasso模型验证](fortext/lasso验证图.png)
2）Dimensionality Reduction & De-noising: Leveraged Lasso’s shrinkage property to successfully extract 10 core contributing factors from 36 redundant features, streamlining the model structure while significantly enhancing the signal-to-noise ratio.
![多特征权重分析](fortext/权重分析.png)
# Key Insights
## 3. Top Coefficients
1)Positive Drivers: Shopping mall density (mall: 0.4135) and bus station proximity (bus_prox: 0.3182) represent the strongest spatial driving forces, validating the core site-selection logic of "traffic gateways" and "commercial clustering."

2)Negative Correction Mechanism: The model accurately captured the inhibitory effect of "Overlapping Commercial/Office Zones" (mall_off: -0.1595). This insight reveals potential spatial displacement effects in specific areas, providing a quantitative basis for avoiding high-rent, high-competition "Red Ocean" markets.
## 4. Model Reliability & Validation
1)Spatial Hit Rate: The top 20% of high-potential grids predicted by the model precisely cover 97.74% of existing store locations, demonstrating high practical utility.

2)Multi-dimensional Spatial Profiling: Conducted comprehensive "spatial health checks" for specific grids (e.g., Site #2760) using Radar Charts (Spider Charts), achieving a quantitative balance between macro-weights and micro-level site attractiveness vs. saturation risk.
![黄金点位分析图](fortext/雷达图.png) 
## 5. Strategic Recommendations
1)The model identified 5 "High Potential, Zero Competition" strategic sites across the region. These sites possess high infrastructure scores but currently lack competing brands, representing golden opportunities for market penetration and expansion into emerging business districts.
![最终选址点](fortext/预测效果最好的一张图.png) 
## 6. Limitations & Future Work
1)Enhancing Spatiotemporal Dynamics: While the current model focuses on static POI attributes, future plans include integrating mobile signaling data or real-time sales from delivery platforms to capture dynamic foot traffic fluctuations.

2)Non-linear Relationship Modeling: As retail site selection is influenced by complex factors such as rent volatility and brand preference, future iterations could introduce ensemble learning or deep neural networks to better deconstruct complex non-linear relationships.

3)Brand Influence Consideration: The current model does not account for the market influence and user preferences of different milk tea brands. Future work could incorporate brand popularity, social media reviews, or sales data into the feature set to further optimize prediction accuracy.
## 7. Project Structure


```text
├├── data/                     # Project Data (Sample Only)
││   └── sample_poi_data.csv   # Representative sample for code demonstration
│├── scripts/                  # Model Construction & Analysis Scripts
││   ├── lasso_analyse.py      # Lasso Regression core algorithm & feature selection
││   ├── Versin ridge.py       # Ridge Regression comparative model
││   ├── Version Random Forest Regression.py  # Random Forest comparative model
├│   ├── Final data compare.py # Multi-model Benchmarking & performance testing
││   └── radar chart.py        # Single-site multi-dimensional feature profiling
│├── figures/                  # Key Visualizations & Outputs
││   ├── Lasso_Drivers_Fixed.png # Lasso feature weight rankings
││   └── model_benchmark.png     # Model performance comparison chart
│└── Version_Lasso.html         # Interactive Global Potential Heatmap
   ```
Sample-Based Demonstration: For data privacy and security reasons, the full production database has been removed. This repository now contains only a single sample dataset in the data/ folder.
  
Code for Reference:The provided scripts are for methodological demonstration and peer review only. To generate the complete analytical results as shown in the figures, a full-scale real-world dataset must be imported.
