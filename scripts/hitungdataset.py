import os

folder = "dataset"

files = [
    "main.csv",
    "awards.csv",
    "awards_suppliers.csv"
]

total = 0

print("UKURAN DATASET OPENTENDER")
print("=" * 40)

for filename in files:
    path = os.path.join(folder, filename)

    size = os.path.getsize(path)
    size_mb = size / (1024 * 1024)

    total += size

    print(f"{filename}: {size_mb:.2f} MB")

print("=" * 40)
print(f"TOTAL: {total / (1024 * 1024):.2f} MB")