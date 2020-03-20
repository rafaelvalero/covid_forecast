# covid_forecast
Data sources and forecast techniques applied to COVID / Coronavirus

# Special Thanks
Dear Coronavirus (Covid or any other fancy name), thanks for all the time I am spending at home and not going anywhere, thanks for 
my concerns about to get toilet tissue and for be damaging my hands skins with so much soaps, thanks
for the continues paranoia to give virus to the lovely elders. I thought for me to give you something back
some of this time in my house I could use it for this repo.
Also serious people and Boris Johnson say you want to kill me. That is not polite.

# Data sources

1. Time series by country:
    1.  [European Centre for Disease Prevention and Control](https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide)

        * Used by: [Our World in Data](https://ourworldindata.org/coronavirus-source-data)

2. [John Hopkins Univeristy data and git](https://github.com/CSSEGISandData/COVID-19)

3. For the UK, I am exploring: https://www.gov.uk/government/publications/covid-19-track-coronavirus-cases
  

# Literature Review for Epidemics studies Summaries

1. Dugas, Andrea Freyer et al. “Influenza forecasting with Google Flu Trends.” PloS one vol. 8,2 (2013): e56176. doi:10.1371/journal.pone.0056176
    **Background**
    We developed a practical influenza forecast model based on real-time, geographically focused, and easy to access data, designed to provide individual medical centers with advanced warning of the expected number of influenza cases, thus allowing for sufficient time to implement interventions. Secondly, we evaluated the effects of incorporating a real-time influenza surveillance system, Google Flu Trends, and meteorological and temporal information on forecast accuracy.
    
    **Methods** 
    Forecast models designed to predict one week in advance were developed from weekly counts of confirmed influenza cases over seven seasons (2004–2011) divided into seven training and out-of-sample verification sets. Forecasting procedures using classical Box-Jenkins, generalized linear models (GLM), and generalized linear autoregressive moving average (GARMA) methods were employed to develop the final model and assess the relative contribution of external variables such as, Google Flu Trends, meteorological data, and temporal information.
    
    **Results**
    A GARMA(3,0) forecast model with Negative Binomial distribution integrating Google Flu Trends information provided the most accurate influenza case predictions. The model, on the average, predicts weekly influenza cases during 7 out-of-sample outbreaks within 7 cases for 83% of estimates. Google Flu Trend data was the only source of external information to provide statistically significant forecast improvements over the base model in four of the seven out-of-sample verification sets. Overall, the p-value of adding this external information to the model is 0.0005. The other exogenous variables did not yield a statistically significant improvement in any of the verification sets.
    
    **Conclusions** 
    Integer-valued autoregression of influenza cases provides a strong base forecast model, which is enhanced by the addition of Google Flu Trends confirming the predictive capabilities of search query based syndromic surveillance. This accessible and flexible forecast model can be used by individual medical centers to provide advanced warning of future influenza cases.

2. Escobar, L.E., Qiao, H. & Peterson, A.T. Forecasting Chikungunya spread in the Americas via data-driven empirical approaches. Parasites Vectors 9, 112 (2016). https://doi.org/10.1186/s13071-016-1403-y
3. Venna, A. Tavanaei, R. N. Gottumukkala, V. V. Raghavan, A. S. Maida and S. Nichols, "A Novel Data-Driven Model for Real-Time Influenza Forecasting," in IEEE Access, vol. 7, pp. 7691-7701, 2019.


# Interesting libraries and repositories  

1. Blog - Python Example
    * https://towardsdatascience.com/using-kalman-filter-to-predict-corona-virus-spread-72d91b74cc8
    * https://github.com/Rank23/COVID19
    
# covid_forecast.yml
covid_forecast.yml is the enviroment with the lybraries install it do:

```python
conda env create -f covid_forecast.yml
```
To use conda enviroment management in general click [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)