import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("train.csv")

# Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x="Sex", hue="Survived", data=data, palette="Set2")
plt.title("Survival Count by Gender")
plt.savefig("survival_by_gender.png")   # Save chart as PNG
plt.close()

# Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(x="Pclass", hue="Survived", data=data, palette="Set1")
plt.title("Survival Count by Passenger Class")
plt.savefig("survival_by_class.png")
plt.close()

# Age Distribution
plt.figure(figsize=(6, 4))
sns.histplot(data["Age"].dropna(), bins=30, kde=True, color="skyblue")
plt.title("Age Distribution of Passengers")
plt.savefig("age_distribution.png")
plt.close()

print("✅ Charts saved as 'survival_by_gender.png', 'survival_by_class.png', and 'age_distribution.png'")
