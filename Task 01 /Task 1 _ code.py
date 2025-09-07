import matplotlib.pyplot as plt

# Sample dataset (ages of people)
ages = [5, 10, 15, 20, 25, 30, 35, 40]
population = [50, 80, 65, 70, 60, 55, 40, 30]

# Plot bar chart
plt.bar(ages, population, color="skyblue")
plt.xlabel("Age")
plt.ylabel("Population")
plt.title("Sample Population Distribution by Age")
plt.show()
