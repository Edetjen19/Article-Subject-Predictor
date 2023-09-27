import json

from datetime import datetime
from collections import defaultdict

from sklearn.metrics import mean_squared_error


with open("train.json", "r") as file:
    data = json.load(file)

with open("test.json", "r") as file:
    test_data = json.load(file)




# Calculate the number of articles per category per month in the training set
articles_per_month_category = defaultdict(lambda: defaultdict(int))

for article in data:
    article_date = datetime.strptime(article['date'], '%Y-%m-%dT%H:%M:%S.%fZ')
    month_year_key = (article_date.year, article_date.month)
    articles_per_month_category[month_year_key][article['subject']] += 1

# Calculate the mean number of articles per category per month
total_months = len(set(articles_per_month_category.keys()))
mean_articles_per_category = defaultdict(float)

for month_year, category_counts in articles_per_month_category.items():
    for category, count in category_counts.items():
        mean_articles_per_category[category] += count / total_months

mean_articles_per_category


# empty for baseline
actual_counts = []
predicted_counts = []

for month_year, category_counts in articles_per_month_category.items():
    for category, mean_count in mean_articles_per_category.items():
        actual_counts.append(category_counts.get(category, 0))
        predicted_counts.append(mean_count)

# Compute the MSE
mse_baseline = mean_squared_error(actual_counts, predicted_counts)
print(mse_baseline)