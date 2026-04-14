# ============================================
# Install needed libraries
# ============================================
# Run once in terminal:
# pip install kagglehub pandas datasets

import pandas as pd
import kagglehub
import glob
from datasets import load_dataset

# ============================================
# Download Dataset 1 from Kaggle
# ============================================
path1 = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
print("Dataset 1 downloaded to:", path1)

files1 = glob.glob(path1 + "/**/*.csv", recursive=True)
print("Files found:", files1)

new_df1 = pd.read_csv(files1[0])
print("Kaggle columns:", new_df1.columns)
print("\nDataset 1 preview:")
print(new_df1.head(2))
print("Columns:", new_df1.columns.tolist())

# ============================================
# Download Dataset 2 from Hugging Face
# ============================================
hf_data = load_dataset("ahmedheakl/resume-atlas", split="train")
new_df2 = pd.DataFrame(hf_data)
print("\nHF columns:", new_df2.columns)
print("\nDataset 2 (Hugging Face) preview:")
print(new_df2.head(2))
print("Columns:", new_df2.columns.tolist())
print("Categories:", new_df2['Category'].unique())

# ============================================
# Load your existing dataset
# ============================================
existing_df = pd.read_csv("Resume.csv")
print("\nExisting data shape:", existing_df.shape)
print("Existing columns:", existing_df.columns.tolist())
print("Existing categories:", existing_df['Category'].unique())

# ============================================
# Standardize all datasets to same format
# Both need columns: 'Resume_str' and 'Category'
# ============================================

# -- Dataset 1 (Kaggle: snehaanbhawal) --
# Usually already has 'Resume_str' and 'Category'
new_df1_clean = new_df1.rename(columns={
    'Resume_str': 'Resume_str',
    'Category': 'Category'
})[['Resume_str', 'Category']]

# -- Dataset 2 (Hugging Face: resume-atlas) --
# Has 'text' and 'category' columns — rename to match
# -- Dataset 2 (Hugging Face: resume-atlas) --

# Rename only Text → Resume_str
new_df2 = new_df2.rename(columns={
    'Text': 'Resume_str'
})

# Debug check 
print("After rename HF columns:", new_df2.columns)

# Now select correct columns
new_df2_clean = new_df2[['Resume_str', 'Category']]

# ============================================
# Merge all three datasets
# ============================================
combined_df = pd.concat(
    [
        existing_df[['Resume_str', 'Category']],
        new_df1_clean,
        new_df2_clean
    ],
    ignore_index=True
)

# Remove exact duplicate resumes
combined_df = combined_df.drop_duplicates(subset=['Resume_str'])

# Remove rows where resume text or category is missing
combined_df = combined_df.dropna(subset=['Resume_str', 'Category'])

# Strip extra whitespace from category names
combined_df['Category'] = combined_df['Category'].str.strip()

print("\nFinal combined shape:", combined_df.shape)
print("\nCategories and their counts:")
print(combined_df['Category'].value_counts())

# ============================================
# Save the expanded dataset
# ============================================
combined_df.to_csv("expanded_resume_dataset.csv", index=False)
print("\nSaved! Total resumes:", len(combined_df))
print("File saved as: expanded_resume_dataset.csv")