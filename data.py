
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("asl_final_dataset.csv")

plt.figure(figsize=(16, 7))

sns.countplot(data=df,x="label",order=df["label"].value_counts().index)

plt.title("Sample Distribution Across ASL Dataset", fontsize=16)
plt.xlabel("ASL Sign Labels", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
