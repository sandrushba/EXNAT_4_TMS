"""
Calculate word frequencies for EXNAT_4_TMS

Input: Python file with texts (same as for Psychopy experiment)

Output: df and csf file with frequencies
       
Written by Sandra Martin
"""


""" Import packages"""
import os
import re
import ast
import pandas as pd
import string

# for getting word frequencies:
from wordfreq import zipf_frequency, word_frequency
# You have to pip install this before importing it
# wordfreq citation: 
# Speer, R., Chin, J., Lin, A., Jewett, S., & Nathan, L. (2018, October 3). LuminosoInsight/wordfreq: v2.2. Zenodo. https://doi.org/10.5281/zenodo.1443582

""" Read in dataset """
# initiate dict
texts = {}

#base_dir = "/data/tu_martin_cloud/EXNAT/EXNAT_4_TMS/Results/EXNAT_4_TMS_Analysis/word_frequencies/"
base_dir = os.path.dirname(__file__)
rel_path = os.path.join(base_dir, 'EXNAT_4_TMS_texts_MC_Qs.py')

with open(rel_path, 'r', encoding='utf-8') as f:
    file_content = f.read()

# Match any variable pattern like 'text_01 = [...]'
matches = re.findall(r'(text_\d+)\s*=\s*(\[[^\]]+\])', file_content)

for name, list_str in matches:
    word_list = ast.literal_eval(list_str)  # Safely convert string list to actual Python list
    texts[name] = word_list

""" Loop words, get frequency for each """
results = []

for text_num, words in texts.items():
    for word in words:
        # Remove punctuation and lowercase for normalization
        clean_word = word.strip(string.punctuation).lower()
        freq = word_frequency(clean_word, 'de', wordlist='best')
        zipf = zipf_frequency(clean_word, 'de', wordlist='best')
        results.append({
            'text_number': text_num,
            'word': clean_word,
            'frequency': freq,
            'zipf_frequency': zipf,
        })

df_wordFreqs = pd.DataFrame(results)

df_wordFreqs.to_csv(path_or_buf="EXNAT_4_TMS_word_freqs.csv",
                 sep=';', na_rep='', header=True, index=False)


