from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier

from collections import defaultdict

from joblib import dump


from datetime import datetime
import json

from dateutil.relativedelta import relativedelta
import csv

# Save the model to a file

# Load the provided JSON file
with open("train.json", "r") as file:
    data = json.load(file)

with open("test.json", "r") as file:
    test_data = json.load(file)



# Extracting month and year from the date for both train and test data
def extract_date_features(data):
    months = []
    years = []
    for article in data:
        article_date = datetime.strptime(article['date'], '%Y-%m-%dT%H:%M:%S.%fZ')
        months.append(article_date.month)
        years.append(article_date.year)
    return months, years

train_months, train_years = extract_date_features(data)
test_months, test_years = extract_date_features(test_data)

# Using TF-IDF for abstract and title feature extraction
vectorizer_abstract = TfidfVectorizer(max_features=1000, stop_words='english')
vectorizer_title = TfidfVectorizer(max_features=500, stop_words='english')

train_abstract_features = vectorizer_abstract.fit_transform([article['abstract'] for article in data])
train_title_features = vectorizer_title.fit_transform([article['title'] for article in data])

test_abstract_features = vectorizer_abstract.transform([article['abstract'] for article in test_data])
test_title_features = vectorizer_title.transform([article['title'] for article in test_data])

# Combining all features
X_train = hstack([train_abstract_features, train_title_features, [[month, year] for month, year in zip(train_months, train_years)]])
X_test = hstack([test_abstract_features, test_title_features, [[month, year] for month, year in zip(test_months, test_years)]])

# Encoding the target variable 
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform([article['subject'] for article in data])

print(X_train.shape, X_test.shape, y_train.shape)

# Initializing and training the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=92, verbose=1, n_jobs=-1)
rf_classifier.fit(X_train, y_train)

#predicting the test data
y_pred = rf_classifier.predict(X_test)

#aggigate the results
predictions_by_month = defaultdict(lambda: defaultdict(int))

for i, article in enumerate(test_data):
    article_date = datetime.strptime(article['date'], '%Y-%m-%dT%H:%M:%S.%fZ')
    month_year_key = (article_date.year, article_date.month)
    subject = label_encoder.inverse_transform([y_pred[i]])[0]
    predictions_by_month[month_year_key][subject] += 1

#format the results
with open('output2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['date', 'subject', 'article_count'])
    for date, subjects in predictions_by_month.items():
        for subject, count in subjects.items():
            writer.writerow([f'{date[0]}-{date[1]:02d}-01', subject, count])


with open('output2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['date', 'subject', 'article_count'])
    for date, subjects in predictions_by_month.items():
        for subject, count in subjects.items():
            writer.writerow([f'{date[0]}-{date[1]:02d}-01', subject, count])

# Save data to CSV
