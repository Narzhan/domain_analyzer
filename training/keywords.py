from rake_nltk import Rake
import pandas
import numpy as np

dataset = pandas.read_csv("text_test_data_splitted.csv", index_col=3, encoding='utf-8', delimiter=";", engine="python")
dataset = dataset.replace(np.nan, '', regex=True)
i = 0
for text in dataset.text:
    if i > 7:
        break
    r = Rake()
    r.extract_keywords_from_text(text)
    print(r.get_ranked_phrases())
    print(r.get_ranked_phrases_with_scores())
    i += 1
