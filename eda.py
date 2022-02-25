import pandas as pd
import numpy as np

# load files into dataframes
reviews_df = pd.read_json('Electronics_5.json', lines=True)
meta_df = pd.read_json('meta_Electronics.json', lines=True)

# merge datasets on ASIN
complete_df = reviews_df.merge(meta_df, on='asin', how='left')

# extract only products titled some form of mice
mouse_df = complete_df[~complete_df['title'].isin(['mouse', 'Mouse'])]

# remove duplicates and account for missing values
mouse_df.dropna(inplace=True)
mouse_df.drop_duplicates(['asin', 'reviewerName', 'unixReviewTime'])

# make a new row for rating (good if > 3, bad otherwise)
def label_rating(row):
    if row['overall'] > 3:
        return 'Good'
    else:
        return 'Bad'
mouse_df['rating_class'] = mouse_df.apply(lambda row: label_rating(row), axis=1)

# concatenate review text and summary


# process text
print(mouse_df.info())
mouse_df.to_json('mouse_reviews.json', orient="records")