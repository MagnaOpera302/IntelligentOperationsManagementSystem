import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv("CleanedUpData.csv")

# Ensure datetime is parsed correctly
df["YearMonth"] = pd.to_datetime(df["YearMonth"], format="%Y-%m")

# Extract Year and Month
df["Year"] = df["YearMonth"].dt.year
df["Month"] = df["YearMonth"].dt.month

# Create readable month names
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
df["MonthName"] = df["Month"].apply(lambda x: month_order[x - 1])

# Plot Settings
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)


# 1. Seasonal Sales Trend by Year
plt.figure()
sns.lineplot(
    x="MonthName", y="Sales", hue="Year", data=df,
    estimator="sum", marker="o", sort=False, errorbar=None
)
plt.title("Monthly Sales Trends by Year", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Total Sales", fontsize=14)
plt.xticks(ticks=range(12), labels=month_order)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# 2. Seasonal Sales Trend by Category (All Years Combined)
plt.figure()
sns.lineplot(
    x="MonthName", y="Sales", hue="Category", data=df,
    estimator="sum", marker="o", sort=False, errorbar=None
)
plt.title("Monthly Sales Trends by Category", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Total Sales", fontsize=14)
plt.xticks(ticks=range(12), labels=month_order)
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# 3. Boxplot: Sales Distribution by Month
plt.figure()
sns.boxplot(x="MonthName", y="Sales", data=df, order=month_order, palette="Set2")
plt.title("Sales Distribution by Month (All Years)", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Sales", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Boxplot: Sales Distribution by Month and Category
plt.figure()
sns.boxplot(x="MonthName", y="Sales", hue="Category", data=df, order=month_order, palette="Set3")
plt.title("Category-wise Sales Distribution by Month", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Sales", fontsize=14)
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
