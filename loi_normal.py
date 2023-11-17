import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Donnee
data = [7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 22, 22, 22, 23, 23, 23, 24, 26, 28, 28, 32, 34, 34, 35, 37, 37, 38, 42, 57, 64]

mu, std = norm.fit(data)

median = np.median(data)
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
if lower_bound <0:
    lower_bound = 0
upper_bound = q3 + 1.5 * iqr

# Calcul Pourcentage
pct_within_iqr = norm.cdf(q3, mu, std) - norm.cdf(q1, mu, std)
pct_below_lower = norm.cdf(lower_bound, mu, std)
pct_above_upper = 1 - norm.cdf(upper_bound, mu, std)

# Graphique
xmin, xmax = min(data), max(data)
x = np.linspace(xmin, xmax, 1000)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

plt.axvline(x=median, color='red', linestyle='dashed', linewidth=2, label=f'Médiane: {median:.2f}')
plt.axvline(x=q1, color='blue', linestyle='dashed', linewidth=2, label=f'Q1: {q1:.2f}')
plt.axvline(x=q3, color='green', linestyle='dashed', linewidth=2, label=f'Q3: {q3:.2f}')
plt.fill_between(x, p, where=(x >= q1) & (x <= q3), color='gray', alpha=0.5, label=f'IQR: {pct_within_iqr:.2%}')

plt.axvline(x=lower_bound, color='orange', linestyle='dotted', linewidth=2, label=f'Borne inférieure: {lower_bound:.2f} ({pct_below_lower:.2%} inférieure)')
plt.axvline(x=upper_bound, color='purple', linestyle='dotted', linewidth=2, label=f'Borne supérieure: {upper_bound:.2f} ({pct_above_upper:.2%} supérieure)')

plt.title("Loi Normal avec statistiques")
plt.xlabel('Données')
plt.ylabel('Densité de probabilité')
plt.legend()

plt.show()
