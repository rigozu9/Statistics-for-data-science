# The data set is adapted from the UCI ML Repository
# Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.

import pandas as pd
import scipy.stats as stats
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

bootstrap_mean /= 10000
print(bootstrap_mean)

bootstrap_var = 0

for sample in b_sample_list:
    sample_chol = sample[0]
    sample_heart_disease = sample[1]
    bootstrap_var += (sample_chol.corr(sample_heart_disease) - bootstrap_mean) ** 2

bootstrap_var /= 10000

se_hat_boot = math.sqrt(bootstrap_var)
print("se hat boot: ", se_hat_boot) #0.058699802815481816

# 2.3
alpha = 0.05
z_alpha_2 = stats.norm.ppf(1 - alpha/2)
# print("Critical value",z)

# Normal interval
t_hat_n = chol.corr(heart_disease)

n_left = t_hat_n - z_alpha_2 * se_hat_boot
print("Normal interval left: ", n_left)

n_right = t_hat_n + z_alpha_2 * se_hat_boot
print("Normal interval right: ", n_right)

# Percentile interval
percentile_interval = []
for i in range(10000):
    sample_chol = b_sample_list[i][0]
    sample_heart_disease = b_sample_list[i][1]
    percentile_interval.append(sample_chol.corr(sample_heart_disease))

percentile_interval.sort()
p_left = percentile_interval[int(10000 * alpha / 2)]
p_right = percentile_interval[int(10000 * (1 - alpha / 2))]
print("Percentile interval left: ", p_left)
print("Percentile interval right: ", p_right)

# Pivotal interval
pivotal_left = 2*t_hat_n - p_right
pivotal_right = 2*t_hat_n - p_left
print("Pivotal interval left: ", pivotal_left)
print("Pivotal interval right: ", pivotal_right)