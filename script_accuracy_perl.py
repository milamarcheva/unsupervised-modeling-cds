import os
import random
import numpy as np
from variation_of_info import list_subdirectories, path_dir_anchor1
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dirpath = ""
base = f"perl eval-many-to-one.pl {dirpath}/test.current.test.pred.test_all /Users/milamarcheva/Downloads/all_tags.txt"

subdirs = list_subdirectories(path_dir_anchor1)
correct = []
df_rows = []
for sub in subdirs:
    dirpath = path_dir_anchor1+sub
    # line = os.system( f"perl eval-many-to-one.pl {dirpath}/test.current.test.pred.test_all /Users/milamarcheva/Downloads/all_tags.txt")
    line =os.popen(f"perl eval-many-to-one.pl {dirpath}/test.current.test.pred.test_all /Users/milamarcheva/Downloads/all_tags.txt").read()
    # print(line)
    i_br = line.index("(")
    i_sl = line.index("/")
    c = int(line[i_br+1:i_sl])
    correct.append(c)
    # if c/10903<0.57:
    ind = dirpath.split("_")[-7:]
    ind[6] = ind[6][0]
    ind = [int(i) for i in ind]
    ind.append(c/10903)
    df_rows.append(ind)

    # print(indices)

df = pd.DataFrame(df_rows, columns = ["UNCCOP", "UNCAUX", "CAUX", "CCOP","ART", "IRPAST","IR3", "Acc"])
print(df.head())

correct_percentage = [c/10903 for c in correct]

print('mean: ', np.average(correct_percentage), ' std: ', np.std(correct_percentage))


# plt.hist(correct_percentage, bins=100)
# plt.show()

# Categorical columns
categorical_vars = ["UNCCOP", "UNCAUX", "CAUX", "CCOP", "ART", "IRPAST", "IR3"]

# Set Seaborn style
sns.set(style="whitegrid")

# Plot configuration
n_cols = 3
n_rows = -(-len(categorical_vars) // n_cols)  # ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), constrained_layout=True)

# Flatten axes for easy iteration
axes = axes.flatten()

# Create bar plots
for i, var in enumerate(categorical_vars):
    ax = axes[i]
    sns.barplot(data=df[df.Acc<0.6][df.Acc>=0.56], x=var, y="Acc", estimator="mean", ci="sd", palette="crest", ax=ax)
    ax.set_title(f"{var}", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Mean Acc")
    if df[var].nunique() > 5:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Remove unused subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Effect of Indices on Acc for acc<0.6 and >0.56", fontsize=16)
plt.show()