import pandas as pd
from scipy.stats import binom

# Ex 5.1 

data = pd.Series(range(11))
# print(data)

probabilities = binom.pmf(data, n=10, p=0.5)

hundred_samples = data.sample(
    n=100,
    replace=True,
    weights=probabilities
)

#print(hundred_samples)

count_5 = (hundred_samples == 5).sum()
p_hat = count_5 / 100

#print(p_hat) #0.29

# Ex 5.2
fract_of_repeats = 0
for _ in range(1000):
    hundred_samples = data.sample(
        n=100,
        replace=True,
        weights=probabilities
    )

    count_5 = (hundred_samples == 5).sum()
    p_hat = count_5 / 100
    if p_hat >= 0.16 and p_hat <= 0.33:
        fract_of_repeats += 1

# print(fract_of_repeats/1000) #0.957

# Ex 5.3
p = binom.pmf(5, n=10, p=0.5)

pmf = []

for k in range(16, 34):
    probability = binom.pmf(k, n=100, p=p)
    pmf.append(probability)

print(sum(pmf)) #0.963987921993128