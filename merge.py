import pandas as pd

df_asl = pd.read_csv('asl_landmarks.csv')
df_words = pd.read_csv('word_landmarks.csv')

print(f"Alphabet dataset size: {df_asl.shape}")
print(f"Word dataset size: {df_words.shape}")

df_final = pd.concat([df_asl, df_words], ignore_index=True)

df_final.to_csv('asl_final_dataset.csv', index=False)

print("Merged successfully!")
print("Total dataset shape:", df_final.shape)
print("All labels now available:")
print(df_final['label'].unique())