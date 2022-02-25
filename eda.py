import pandas as pd
import numpy as np
from pycontractions import Contractions

# load files into dataframes
reviews_df = pd.read_json('Electronics_5.json', lines=True)
meta_df = pd.read_json('meta_Electronics.json', lines=True)

print("~~~ data loaded ~~~")

# merge datasets on ASIN
complete_df = reviews_df.merge(meta_df, on='asin', how='left')

print("~~~ dataframes merged ~~~")

# extract only products titled some form of mice
mouse_df = complete_df[~complete_df['title'].isin(['mouse', 'Mouse'])]

print("~~~ filtered for only mouse products ~~~")

# remove duplicates and account for missing values
mouse_df.dropna(inplace=True)
mouse_df.drop_duplicates(['asin', 'reviewerName', 'unixReviewTime'])

print("~~~ dropped duplicates and removed rows with n/a")

# make a new row for rating (good if > 3, bad otherwise)
def label_rating(row):
    if row['overall'] > 3:
        return 'Good'
    else:
        return 'Bad'
mouse_df['rating_class'] = mouse_df.apply(lambda row: label_rating(row), axis=1)

# concatenate review text and summary
# mouse_df['reviewText'] = mouse_df['reviewText'] + " " + mouse_df['summary']
# mouse_df.drop(['summary'], inplace=True)

print("~~~ formatted data ~~~")

# normalize text
string_cols = mouse_df.select_dtypes(include=[np.object]).columns
mouse_df[string_cols] = mouse_df[string_cols].str.replace(r'<[^<>]*>', '', regex=True)
mouse_df[string_cols] = mouse_df[string_cols].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
cont = Contractions(api_key="glove-twitter-100")
mouse_df[string_cols] = mouse_df[string_cols].apply(lambda x: cont.expand_texts([x]))


print("~~~ normalized text in data ~~~")

# process text
print(mouse_df.info())
mouse_df.to_json('mouse_reviews.json', orient="records")