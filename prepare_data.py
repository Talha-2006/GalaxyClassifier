import os
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ─── Config ───────────────────────────────────────────────────────────────────

RAW_CSV_PATH = 'data/training_solutions_rev1.csv'
INPUT_IMG_DIR = 'data/images/train/images_training_rev1/'
OUTPUT_IMG_DIR = 'data/images/train/images_resized/'
CONFIDENCE_THRESHOLD = 0.65
IMG_SIZE = (224, 224)


# ─── Step 1: Clean and Split the CSV ─────────────────────────────────────────

print("Cleaning CSV...")

df = pd.read_csv(RAW_CSV_PATH)
df = df[['GalaxyID', 'Class1.1', 'Class1.2']]

df['max_prob'] = df[['Class1.1', 'Class1.2']].max(axis=1)
df = df[df['max_prob'] >= CONFIDENCE_THRESHOLD].reset_index(drop=True)

label_map = {'Class1.1': 0, 'Class1.2': 1}
df['label'] = df[['Class1.1', 'Class1.2']].idxmax(axis=1).map(label_map)
df['label'] = df['label'].astype(int)
df = df[['GalaxyID', 'label']]

print(f"Total samples after filtering: {len(df)}")
print(df['label'].value_counts().sort_index())

# Shuffle and split 70/15/15
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.70 * len(df))
val_size = int(0.15 * len(df))

train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:train_size + val_size]
test_df = df.iloc[train_size + val_size:]

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print(f"\nSplit sizes:")
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")


# ─── Step 2: Resize Images ────────────────────────────────────────────────────

print("\nResizing images...")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

galaxy_ids = df['GalaxyID'].tolist()
already_done = set(os.listdir(OUTPUT_IMG_DIR))

skipped = 0
failed = 0

for galaxy_id in tqdm(galaxy_ids):
    fname = f"{galaxy_id}.jpg"

    if fname in already_done:
        skipped += 1
        continue

    input_path = os.path.join(INPUT_IMG_DIR, fname)

    try:
        img = Image.open(input_path).convert('RGB')
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        img.save(os.path.join(OUTPUT_IMG_DIR, fname))
    except Exception as e:
        print(f"Failed on {fname}: {e}")
        failed += 1

print(f"\nDone.")
print(f"Resized: {len(galaxy_ids) - skipped - failed}")
print(f"Skipped (already existed): {skipped}")
print(f"Failed: {failed}")