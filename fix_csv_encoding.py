import pandas as pd

# Path to your original data file
input_path = "/Users/naelamacbookair/desktop backup/self projects/EcoTrack/data/raw/data.csv"
# Path for the fixed file
output_path = "/Users/naelamacbookair/desktop backup/self projects/EcoTrack/data/raw/data_fixed.csv"

# Try different encodings until one works
encodings_to_try = ['latin1', 'ISO-8859-1', 'cp1252', 'mac_roman']

for encoding in encodings_to_try:
    try:
        print(f"Trying to read with {encoding} encoding...")
        # Read the CSV with a specific encoding
        df = pd.read_csv(input_path, encoding=encoding)
        
        # Preview the data
        print("Successfully read the file!")
        print(f"Data shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("First few rows:")
        print(df.head())
        
        # Save with UTF-8 encoding
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Fixed file saved to {output_path}")
        
        # Success - no need to try other encodings
        break
        
    except Exception as e:
        print(f"Failed with {encoding} encoding: {str(e)}")
        print("Trying next encoding...")