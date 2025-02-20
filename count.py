import pandas as pd
import numpy as np

# Load dataset
file_path = "./Dataset.xlsx"
df = pd.read_excel(file_path)

# Count utterances per intent
intent_counts = df["Intent"].value_counts()

# Count total unique intents
total_intents = intent_counts.shape[0]

# Get top 10 intents (most utterances)
top_10_intents = intent_counts.head(10)

# Get bottom 10 intents (least utterances)
bottom_10_intents = intent_counts.tail(10)

# Calculate average & median
average_count = int(np.mean(intent_counts))


# Get 10 intents closest to average count
average_10_intents = intent_counts.iloc[(intent_counts - average_count).abs().argsort()[:10]]


# Display results
print(f"🔹 Total Unique Intents: {total_intents}\n")

print("🔹 Top 10 Intents (Most Utterances):")
print(top_10_intents)

print("\n🔹 Bottom 10 Intents (Least Utterances):")
print(bottom_10_intents)

print("\n🔹 Average 10 Intents (Closest to Mean Count):")
print(average_10_intents)


