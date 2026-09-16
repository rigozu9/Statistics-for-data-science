# The data set is adapted from the UCI ML Repository
# Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.

import pandas as pd
import math

# import dataset
data = pd.read_csv("week2_heart_disease_data.csv", index_col=0)

chol = data['chol']
heart_disease = 1 * (data['heart-disease'] != 0)

# 2.1
print(chol.corr(heart_disease)) #0.08516361139953128

#2.2
print(len(data)) #303

b_sample_list = []
bootstrap_mean = 0

for _ in range(10000):
    bootstrap_sample = data.sample(
        n=len(data),
        replace=True
    )
    sample_chol = bootstrap_sample['chol']
    sample_heart_disease = 1 * (bootstrap_sample['heart-disease'] != 0)
    b_sample_list.append((sample_chol, sample_heart_disease))
    bootstrap_mean += sample_chol.corr(sample_heart_disease)

print(bootstrap_mean / 10000)

bootstrap_var = 0

for sample in b_sample_list:
    sample_chol = sample[0]
    sample_heart_disease = sample[1]
    bootstrap_var += (sample_chol.corr(sample_heart_disease) - bootstrap_mean / 10000) ** 2

bootstrap_var /= 10000

print("se hat boot: ",math.sqrt(bootstrap_var))



