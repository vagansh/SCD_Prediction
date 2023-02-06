This folder contains the datasets used for the analysis. These were created from the BRFSS CDC Annual Survey data. 

Data were obtained from the Behavioural Risk Factor Surveillance System (BRFSS) survey which is conducted annually by the Centers for Disease Control 
and Prevention (CDC), USA. BRFSS, established in 1984, is the largest random digit-dialled telephone health survey of non-institutionalised U.S adults 
18 years or older. Addition of the Cognitive Decline module in the BRFSS questionnaire is a recent development. The module was developed in 2007 
following recommendations from a national panel of experts that cognitive impairment must be addressed as a public health issue. After cognitive testing 
and field testing the module was finalized by 2018. 
	
The modules contains six questions linked to Cognitive Decline. The first question of this six-question module enquires about subjective 
memory loss/confusion and this information is indicated by the variable named CIMEMLOS. Because of the way in which CIMEMLOS is defined and its inclusion
in the Cognitive Decline module, it reflects SCD due to neurological causes and not due to psychiatric conditions. Therefore, CIMEMOS can be used as in 
indicator of dementia risk. To represent the socio-economic position (SEP) of the respondent, values of two variables: INCOME2 and  EDUCA, 
were additively combined. Scientific evidence establishes that income levels and education together form a more comprehensive estimate of 
socio-economic position than either alone [1]. A higher value of SEP indicates better socio-economic background in this study. 
	
Variables/feature selected for the study were based on prior research which link these with the risk of Dementia. 

Diabetes[2] and Stroke [3] variables were included because they are linked to dementia onset.	BRFSS data for the year 2018 and 2020 were used and 
combined because these contained the variables (Table I) required for the study. Missing/refused values for each variable were removed. I
n the BRFSS data, generally these values are 7 or 77=Don’t know/Not sure, 9 or 99=Refused or BLANK. Records with BMI in the range of 18.50 to 50 were 
selected because they represented 95% of the BMI distribution in the data. 



References

[1] M. H. Lindberg, G. Chen, J. A. Olsen et al., “Combining education and income into a socioeconomic position score for use in studies of health 
inequalities,” BMC Public Health, vol. 22, no. 1, p. 969, May 2022. [Online]. Available: https://doi.org/10.1186/s12889-022-13366-8

[2] L. A. Zilliox, K. Chadrasekaran, J. Y. Kwan, and J. W. Russell, “Diabetes and Cognitive Impairment,” Current diabetes reports, vol. 16, no. 9, 
p. 87, Sep. 2016. [Online]. Available: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5528145/ 

[3] N. K. Al-Qazzaz, S. H. Ali, S. A. Ahmad, S. Islam, and K. Mohamad, “Cognitive impairment and memory dysfunction after a stroke diagnosis: a 
post-stroke memory assessment,” Neuropsychiatric Disease and Treatment, vol. 10, pp. 1677–1691, Sep. 2014. [Online].
