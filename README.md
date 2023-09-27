# Predicting Subjects From Articles Supervised
Written and developed by Eric Detjen  

This repository contains a machine-learning model that classifies articles into different subjects. The model uses article meta-data sourced from arXiv's condensed matter physics articles(http://www.arxiv.org/). The project leverages a variety of techniques, including TF-IDF for text feature extraction, date feature extraction, and Random Forest Classifier for model training. Experiment logging and tracking are handled using the Capital One open-source framework, Rubicon-ML.

For ease of viewing, I have included the model.ipynb file in the README here. If desired, this and all other files are available in this repo.

## Table of Contents

1. [Initialize Rubicon and Project](#initialize-rubicon-and-project)
2. [Read and Adjust Data](#read-and-adjust-data)
3. [Feature Extraction](#feature-extraction)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Logging](#logging)
7. [Saving](#saving)
8. [Model Evaluation](#model-evaluation)
9. [Results Summary](#results-summary)


```python
#imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
from joblib import dump
import pandas as pd
from IPython.display import display
from rubicon_ml import Rubicon
from datetime import datetime
import json
from colorama import Fore, Style
from dateutil.relativedelta import relativedelta
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


```

### Initialize Rubicon and Project
We initialize Rubicon for logging and create a project for Article Classification.


```python
rubicon = Rubicon(persistence="filesystem", root_dir="./rubicon-root")
project = rubicon.get_or_create_project("Article Classification")

# Log the experiment
experiment = project.log_experiment(
    model_name="Random Forest",
    tags=["text classification", "NLP"]
)

```

### Read and Adjust Data
Read the data  

we will test the accuracy by comparing using 80% of the train data to train and then for the last 20% removing the subjects and comparing the models results on that against the true subject data for the same 20%


```python
with open("train.json", "r") as file:
    data = json.load(file)

with open("test.json", "r") as file:
    final_test_data = json.load(file)

# Split the data into 80% for training and 20% for testing
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)



```

### Feature Extraction

We extract features using TF-IDF for article abstracts and titles and also extract date features.

**The TF-IDF is key here since it will yield a higher rating for words that are frequent in the specific text it is analyzing yet rare across all documents. This allows the model to focus on the words that are important and not common words like “and” or “the”.**



```python
max_features_abstract = 1000
stop_words = 'english'
max_features_title = 500
# Extracting month and year from the date for both train and test data
def extract_date_features(data):
    months = []
    years = []
    for article in data:
        article_date = datetime.strptime(article['date'], '%Y-%m-%dT%H:%M:%S.%fZ')
        months.append(article_date.month)
        years.append(article_date.year)
    return months, years


# Initialize the vectorizers and the label encoder
vectorizer_abstract = TfidfVectorizer(max_features= max_features_abstract, stop_words='english')
vectorizer_title = TfidfVectorizer(max_features= max_features_title, stop_words='english')
label_encoder = LabelEncoder()

# Extract date features for train and test data
train_months, train_years = extract_date_features(train_data)
test_months, test_years = extract_date_features(test_data)

# Using TF-IDF for abstract and title feature extraction for train and test data
train_abstract_features = vectorizer_abstract.fit_transform([article['abstract'] for article in train_data])
train_title_features = vectorizer_title.fit_transform([article['title'] for article in train_data])

test_abstract_features = vectorizer_abstract.transform([article['abstract'] for article in test_data])
test_title_features = vectorizer_title.transform([article['title'] for article in test_data])
```

### Data Preparation
I now combine all the extracted features using hstack and encode the labels using label_encoder.


```python


# Combining all features
X_train = hstack([train_abstract_features, train_title_features, [[month, year] for month, year in zip(train_months, train_years)]])
X_test = hstack([test_abstract_features, test_title_features, [[month, year] for month, year in zip(test_months, test_years)]])

# Encoding the target variable 
y_train = label_encoder.fit_transform([article['subject'] for article in train_data])
y_test = label_encoder.transform([article['subject'] for article in test_data])

```

### Model Training
I initialize and train the Random Forest Classifier.

I chose to use the random forest classifier because it is a very robust model for supervised learning that is not prone to overfitting and is very good at handling the highly dimensional categorical data that we have. It is also is very interpretable which allows us to log meaningful metrics and parameters to Rubicon.


```python
# Initializing and training the Random Forest classifier
n_estimators=200
random_state=92
verbose=1 
n_jobs=-1
rf_classifier = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, verbose = verbose, n_jobs = n_jobs)
rf_classifier.fit(X_train, y_train)

```

    [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 10 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 tasks      | elapsed:   24.4s
    [Parallel(n_jobs=-1)]: Done 180 tasks      | elapsed:  1.9min
    [Parallel(n_jobs=-1)]: Done 200 out of 200 | elapsed:  2.1min finished
    [Parallel(n_jobs=10)]: Using backend ThreadingBackend with 10 concurrent workers.
    [Parallel(n_jobs=10)]: Done  30 tasks      | elapsed:    0.3s
    [Parallel(n_jobs=10)]: Done 180 tasks      | elapsed:    1.6s
    [Parallel(n_jobs=10)]: Done 200 out of 200 | elapsed:    1.8s finished


### Logging

Using the Capital One Rubicon system to log our important parameters



```python

experiment = project.log_experiment(
    model_name="Random Forest",
    tags=["text classification", "NLP"]
)

# Log parameters dynamically from the trained RandomForestClassifier object
parameters_to_log = [
    "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
    "min_weight_fraction_leaf", "max_features", "max_leaf_nodes", 
    "min_impurity_decrease", "min_impurity_split", "bootstrap", 
    "oob_score", "n_jobs", "random_state", "verbose", "warm_start"
]

for param_name in parameters_to_log:
    param_value = getattr(rf_classifier, param_name, "Not set")
    experiment.log_parameter(name=param_name, value=param_value)

```

### Saving
I use the joblib dump to save the model so it does not have to be trained repeatedly. This file was over the GitHub size limits so it is unfortunately not in the repo. The model only takes about 6 minutes to train locally though.


```python
model_path = "random_forest_model.joblib"
dump(rf_classifier, model_path)
```

### Model Evaluation
here we evaluate the performance of the test data set.

Again, we test the accuracy by comparing using 80% of the train data to train and then for the last 20% removing the subjects and comparing the model results on that against the true subject data for the same 20%


```python
#predicting the test data
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
model_path, accuracy

experiment.log_metric(name="Accuracy_Score_5", value=accuracy)  

# Optionally, log additional text or configurations
#experiment.log_parameter(name="TF-IDF Max Features", value=f"{max_features_abstract} for abstract, {max_features_title} for title")

```

    [Parallel(n_jobs=10)]: Using backend ThreadingBackend with 10 concurrent workers.
    [Parallel(n_jobs=10)]: Done  30 tasks      | elapsed:    0.3s
    [Parallel(n_jobs=10)]: Done 180 tasks      | elapsed:    1.4s
    [Parallel(n_jobs=10)]: Done 200 out of 200 | elapsed:    1.5s finished





    <rubicon_ml.client.metric.Metric at 0x2ae3b4280>



### Results Summary
I display the results from the tests here. It shows which subjects it messes up on most and for each of those, which subjects it mistakenly labels it as.   
This is key for giving meaningful insight into what is causing the confusion.


```python
decoded_predictions = label_encoder.inverse_transform(y_pred)

# Extract the actual labels from the test data
actual_labels = [article['subject'] for article in test_data]

# Compare the predicted labels to the actual labels
results_comparison = list(zip(actual_labels, decoded_predictions))

# Initialize dictionaries to keep track of all predictions and incorrect predictions for each unique "Actual" value.
total_predictions = defaultdict(int)
incorrect_predictions = defaultdict(lambda: defaultdict(int))

df_rows = []

# Populate the dictionaries with data.
for actual, predicted in results_comparison:
    total_predictions[actual] += 1
    if actual != predicted:
        incorrect_predictions[actual][predicted] += 1

# Add rows to the DataFrame list.
for actual, predicted_dict in incorrect_predictions.items():
    total = total_predictions[actual]
    correct = total - sum(predicted_dict.values())
    correct_percentage = (correct / total) * 100

    first_row = True
    for predicted, count in predicted_dict.items():
        if first_row:
            df_rows.append({
                'Actual': actual,
                'Predicted': predicted,
                'Count': count,
                'Correct Percentage': f"{correct_percentage:.2f}%",
                'Total Articles': total
            })
            first_row = False
        else:
            df_rows.append({
                'Actual': '',
                'Predicted': predicted,
                'Count': count,
                'Correct Percentage': '',
                'Total Articles': ''
            })

    # Add a separator row
    df_rows.append({
        'Actual': '',
        'Predicted': '',
        'Count': '',
        'Correct Percentage': '',
        'Total Articles': ''
    })
pd.set_option('display.max_rows', None)

# Create a DataFrame from the list of rows.
df = pd.DataFrame(df_rows)

# Display the DataFrame.
display(df)
```

### Overall Accuracy: 61.20%

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
      <th>Count</th>
      <th>Correct Percentage</th>
      <th>Total Articles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>quantum physics</td>
      <td>mesoscale and nanoscale physics</td>
      <td>193</td>
      <td>32.22%</td>
      <td>841</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>quantum gases</td>
      <td>67</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>materials science</td>
      <td>30</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>superconductivity</td>
      <td>30</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>138</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>84</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>other condensed matter</td>
      <td>14</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>6</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>condensed matter</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>10</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td>disordered systems and neural networks</td>
      <td>statistical mechanics</td>
      <td>682</td>
      <td>23.60%</td>
      <td>1449</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td>superconductivity</td>
      <td>19</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>13</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>112</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>14</th>
      <td></td>
      <td>materials science</td>
      <td>103</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>15</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>91</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>68</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>17</th>
      <td></td>
      <td>quantum gases</td>
      <td>9</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>18</th>
      <td></td>
      <td>quantum physics</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>19</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>20</th>
      <td></td>
      <td>condensed matter</td>
      <td>9</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>21</th>
      <td></td>
      <td>physics - physics and society</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>22</th>
      <td></td>
      <td>other condensed matter</td>
      <td>7</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>23</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>24</th>
      <td>mesoscale and nanoscale physics</td>
      <td>superconductivity</td>
      <td>231</td>
      <td>74.87%</td>
      <td>5527</td>
    </tr>
    <tr>
      <th>25</th>
      <td></td>
      <td>materials science</td>
      <td>579</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>26</th>
      <td></td>
      <td>other condensed matter</td>
      <td>20</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>27</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>190</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>28</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>220</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>29</th>
      <td></td>
      <td>quantum gases</td>
      <td>19</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>30</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>29</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>31</th>
      <td></td>
      <td>quantum physics</td>
      <td>50</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>32</th>
      <td></td>
      <td>condensed matter</td>
      <td>34</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>33</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>16</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>34</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>35</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>36</th>
      <td>other condensed matter</td>
      <td>mesoscale and nanoscale physics</td>
      <td>198</td>
      <td>20.72%</td>
      <td>1144</td>
    </tr>
    <tr>
      <th>37</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>205</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>38</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>118</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>39</th>
      <td></td>
      <td>quantum gases</td>
      <td>72</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>40</th>
      <td></td>
      <td>materials science</td>
      <td>238</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>41</th>
      <td></td>
      <td>superconductivity</td>
      <td>28</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>42</th>
      <td></td>
      <td>quantum physics</td>
      <td>12</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>43</th>
      <td></td>
      <td>condensed matter</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>44</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>28</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>45</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>46</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>47</th>
      <td></td>
      <td>physics - chemical physics</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>48</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>49</th>
      <td>soft condensed matter</td>
      <td>materials science</td>
      <td>217</td>
      <td>60.15%</td>
      <td>2454</td>
    </tr>
    <tr>
      <th>50</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>549</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>51</th>
      <td></td>
      <td>other condensed matter</td>
      <td>50</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>52</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>30</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>53</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>43</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>54</th>
      <td></td>
      <td>condensed matter</td>
      <td>44</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>55</th>
      <td></td>
      <td>physics - biological physics</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>56</th>
      <td></td>
      <td>superconductivity</td>
      <td>15</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>57</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>16</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>58</th>
      <td></td>
      <td>quantum gases</td>
      <td>13</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>59</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>60</th>
      <td>superconductivity</td>
      <td>mesoscale and nanoscale physics</td>
      <td>105</td>
      <td>84.85%</td>
      <td>3643</td>
    </tr>
    <tr>
      <th>61</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>43</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>62</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>246</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>63</th>
      <td></td>
      <td>condensed matter</td>
      <td>20</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>64</th>
      <td></td>
      <td>quantum gases</td>
      <td>21</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>65</th>
      <td></td>
      <td>materials science</td>
      <td>74</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>66</th>
      <td></td>
      <td>quantum physics</td>
      <td>10</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>67</th>
      <td></td>
      <td>other condensed matter</td>
      <td>25</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>68</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>69</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>6</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>70</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>71</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>72</th>
      <td>materials science</td>
      <td>statistical mechanics</td>
      <td>244</td>
      <td>65.53%</td>
      <td>4549</td>
    </tr>
    <tr>
      <th>73</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>713</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>74</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>150</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>75</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>322</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>76</th>
      <td></td>
      <td>condensed matter</td>
      <td>31</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>77</th>
      <td></td>
      <td>physics - chemical physics</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>78</th>
      <td></td>
      <td>superconductivity</td>
      <td>76</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>79</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>15</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>80</th>
      <td></td>
      <td>quantum physics</td>
      <td>8</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>81</th>
      <td></td>
      <td>nuclear theory</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>82</th>
      <td></td>
      <td>other condensed matter</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>83</th>
      <td></td>
      <td>quantum gases</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>84</th>
      <td></td>
      <td>physics - optics</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>85</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>86</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>87</th>
      <td>strongly correlated electrons</td>
      <td>mesoscale and nanoscale physics</td>
      <td>508</td>
      <td>69.34%</td>
      <td>4883</td>
    </tr>
    <tr>
      <th>88</th>
      <td></td>
      <td>materials science</td>
      <td>370</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>89</th>
      <td></td>
      <td>superconductivity</td>
      <td>373</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>90</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>153</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>91</th>
      <td></td>
      <td>condensed matter</td>
      <td>25</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>92</th>
      <td></td>
      <td>other condensed matter</td>
      <td>10</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>93</th>
      <td></td>
      <td>quantum gases</td>
      <td>26</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>94</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>8</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>95</th>
      <td></td>
      <td>quantum physics</td>
      <td>11</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>96</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>10</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>97</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>98</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>99</th>
      <td>statistical mechanics</td>
      <td>condensed matter</td>
      <td>84</td>
      <td>74.64%</td>
      <td>4870</td>
    </tr>
    <tr>
      <th>100</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>284</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>101</th>
      <td></td>
      <td>quantum physics</td>
      <td>61</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>102</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>306</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>103</th>
      <td></td>
      <td>superconductivity</td>
      <td>55</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>104</th>
      <td></td>
      <td>materials science</td>
      <td>122</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>105</th>
      <td></td>
      <td>other condensed matter</td>
      <td>64</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>106</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>93</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>107</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>124</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>108</th>
      <td></td>
      <td>quantum gases</td>
      <td>37</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>109</th>
      <td></td>
      <td>physics - physics and society</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>110</th>
      <td></td>
      <td>mathematical physics</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>111</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>112</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>113</th>
      <td>condensed matter</td>
      <td>materials science</td>
      <td>152</td>
      <td>54.36%</td>
      <td>2360</td>
    </tr>
    <tr>
      <th>114</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>194</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>115</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>159</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>116</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>393</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>117</th>
      <td></td>
      <td>superconductivity</td>
      <td>64</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>118</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>66</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>119</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>19</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>120</th>
      <td></td>
      <td>other condensed matter</td>
      <td>18</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>121</th>
      <td></td>
      <td>quantum gases</td>
      <td>8</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>122</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>123</th>
      <td></td>
      <td>quantum physics</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>124</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>125</th>
      <td>quantum gases</td>
      <td>strongly correlated electrons</td>
      <td>113</td>
      <td>67.97%</td>
      <td>946</td>
    </tr>
    <tr>
      <th>126</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>63</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>127</th>
      <td></td>
      <td>other condensed matter</td>
      <td>8</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>128</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>56</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>129</th>
      <td></td>
      <td>materials science</td>
      <td>21</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>130</th>
      <td></td>
      <td>condensed matter</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>131</th>
      <td></td>
      <td>superconductivity</td>
      <td>18</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>132</th>
      <td></td>
      <td>quantum physics</td>
      <td>14</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>133</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>134</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>135</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>136</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>137</th>
      <td>physics - atomic physics</td>
      <td>quantum gases</td>
      <td>17</td>
      <td>1.28%</td>
      <td>78</td>
    </tr>
    <tr>
      <th>138</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>9</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>139</th>
      <td></td>
      <td>other condensed matter</td>
      <td>7</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>140</th>
      <td></td>
      <td>superconductivity</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>141</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>9</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>142</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>143</th>
      <td></td>
      <td>quantum physics</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>144</th>
      <td></td>
      <td>condensed matter</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>145</th>
      <td></td>
      <td>materials science</td>
      <td>21</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>146</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>147</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>148</th>
      <td>high energy physics - theory</td>
      <td>mesoscale and nanoscale physics</td>
      <td>53</td>
      <td>13.07%</td>
      <td>589</td>
    </tr>
    <tr>
      <th>149</th>
      <td></td>
      <td>condensed matter</td>
      <td>102</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>150</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>72</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>151</th>
      <td></td>
      <td>superconductivity</td>
      <td>36</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>152</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>215</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>153</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>8</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>154</th>
      <td></td>
      <td>materials science</td>
      <td>11</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>155</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>156</th>
      <td></td>
      <td>quantum physics</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>157</th>
      <td></td>
      <td>quantum gases</td>
      <td>7</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>158</th>
      <td></td>
      <td>other condensed matter</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>159</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>160</th>
      <td>high energy physics - phenomenology</td>
      <td>condensed matter</td>
      <td>17</td>
      <td>0.72%</td>
      <td>138</td>
    </tr>
    <tr>
      <th>161</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>49</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>162</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>15</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>163</th>
      <td></td>
      <td>superconductivity</td>
      <td>25</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>164</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>165</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>11</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>166</th>
      <td></td>
      <td>quantum physics</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>167</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>168</th>
      <td></td>
      <td>materials science</td>
      <td>7</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>169</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>170</th>
      <td></td>
      <td>quantum gases</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>171</th>
      <td></td>
      <td>other condensed matter</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>172</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>173</th>
      <td>physics - computational physics</td>
      <td>statistical mechanics</td>
      <td>30</td>
      <td>0.00%</td>
      <td>83</td>
    </tr>
    <tr>
      <th>174</th>
      <td></td>
      <td>materials science</td>
      <td>26</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>175</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>16</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>176</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>177</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>7</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>178</th>
      <td></td>
      <td>quantum physics</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>179</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>180</th>
      <td>physics - fluid dynamics</td>
      <td>soft condensed matter</td>
      <td>51</td>
      <td>0.00%</td>
      <td>91</td>
    </tr>
    <tr>
      <th>181</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>30</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>182</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>183</th>
      <td></td>
      <td>materials science</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>184</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>185</th>
      <td></td>
      <td>superconductivity</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>186</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>187</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>188</th>
      <td>mathematics - probability</td>
      <td>statistical mechanics</td>
      <td>57</td>
      <td>0.00%</td>
      <td>63</td>
    </tr>
    <tr>
      <th>189</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>190</th>
      <td></td>
      <td>quantum physics</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>191</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>192</th>
      <td>physics - chemical physics</td>
      <td>strongly correlated electrons</td>
      <td>7</td>
      <td>0.00%</td>
      <td>140</td>
    </tr>
    <tr>
      <th>193</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>29</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>194</th>
      <td></td>
      <td>materials science</td>
      <td>62</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>195</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>19</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>196</th>
      <td></td>
      <td>condensed matter</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>197</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>17</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>198</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>199</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>200</th>
      <td>physics - physics and society</td>
      <td>statistical mechanics</td>
      <td>125</td>
      <td>9.49%</td>
      <td>158</td>
    </tr>
    <tr>
      <th>201</th>
      <td></td>
      <td>materials science</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>202</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>13</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>203</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>204</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>205</th>
      <td>nonlinear sciences - chaotic dynamics</td>
      <td>statistical mechanics</td>
      <td>117</td>
      <td>1.10%</td>
      <td>182</td>
    </tr>
    <tr>
      <th>206</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>207</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>8</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>208</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>16</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>209</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>210</th>
      <td></td>
      <td>condensed matter</td>
      <td>22</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>211</th>
      <td></td>
      <td>quantum physics</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>212</th>
      <td></td>
      <td>other condensed matter</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>213</th>
      <td></td>
      <td>materials science</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>214</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>215</th>
      <td>nonlinear sciences - pattern formation and sol...</td>
      <td>materials science</td>
      <td>12</td>
      <td>0.00%</td>
      <td>74</td>
    </tr>
    <tr>
      <th>216</th>
      <td></td>
      <td>superconductivity</td>
      <td>6</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>217</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>24</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>218</th>
      <td></td>
      <td>quantum gases</td>
      <td>13</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>219</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>220</th>
      <td></td>
      <td>other condensed matter</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>221</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>7</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>222</th>
      <td></td>
      <td>condensed matter</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>223</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>224</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>225</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>226</th>
      <td>physics - biological physics</td>
      <td>soft condensed matter</td>
      <td>48</td>
      <td>0.72%</td>
      <td>139</td>
    </tr>
    <tr>
      <th>227</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>61</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>228</th>
      <td></td>
      <td>materials science</td>
      <td>18</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>229</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>7</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>230</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>231</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>232</th>
      <td>nonlinear sciences - adaptation and self-organ...</td>
      <td>disordered systems and neural networks</td>
      <td>7</td>
      <td>0.00%</td>
      <td>58</td>
    </tr>
    <tr>
      <th>233</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>39</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>234</th>
      <td></td>
      <td>condensed matter</td>
      <td>8</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>235</th>
      <td></td>
      <td>materials science</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>236</th>
      <td></td>
      <td>physics - physics and society</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>237</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>238</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>239</th>
      <td>physics - optics</td>
      <td>soft condensed matter</td>
      <td>8</td>
      <td>0.00%</td>
      <td>154</td>
    </tr>
    <tr>
      <th>240</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>49</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>241</th>
      <td></td>
      <td>materials science</td>
      <td>74</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>242</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>11</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>243</th>
      <td></td>
      <td>quantum gases</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>244</th>
      <td></td>
      <td>quantum physics</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>245</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>246</th>
      <td></td>
      <td>other condensed matter</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>247</th>
      <td></td>
      <td>superconductivity</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>248</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>249</th>
      <td></td>
      <td>condensed matter</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>250</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>251</th>
      <td>general relativity and quantum cosmology</td>
      <td>materials science</td>
      <td>2</td>
      <td>1.64%</td>
      <td>61</td>
    </tr>
    <tr>
      <th>252</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>253</th>
      <td></td>
      <td>quantum gases</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>254</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>23</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>255</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>6</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>256</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>9</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>257</th>
      <td></td>
      <td>quantum physics</td>
      <td>6</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>258</th>
      <td></td>
      <td>other condensed matter</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>259</th>
      <td></td>
      <td>condensed matter</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>260</th>
      <td></td>
      <td>superconductivity</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>261</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>262</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>263</th>
      <td>quantitative biology - populations and evolution</td>
      <td>statistical mechanics</td>
      <td>57</td>
      <td>0.00%</td>
      <td>67</td>
    </tr>
    <tr>
      <th>264</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>265</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>266</th>
      <td></td>
      <td>materials science</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>267</th>
      <td></td>
      <td>physics - physics and society</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>268</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>269</th>
      <td>quantitative biology - biomolecules</td>
      <td>statistical mechanics</td>
      <td>22</td>
      <td>0.00%</td>
      <td>49</td>
    </tr>
    <tr>
      <th>270</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>19</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>271</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>272</th>
      <td></td>
      <td>materials science</td>
      <td>7</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>273</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>274</th>
      <td>mathematical physics</td>
      <td>statistical mechanics</td>
      <td>169</td>
      <td>3.06%</td>
      <td>294</td>
    </tr>
    <tr>
      <th>275</th>
      <td></td>
      <td>quantum physics</td>
      <td>11</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>276</th>
      <td></td>
      <td>quantum gases</td>
      <td>9</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>277</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>9</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>278</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>23</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>279</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>32</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>280</th>
      <td></td>
      <td>materials science</td>
      <td>19</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>281</th>
      <td></td>
      <td>condensed matter</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>282</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>283</th>
      <td></td>
      <td>other condensed matter</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>284</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>285</th>
      <td></td>
      <td>superconductivity</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>286</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>287</th>
      <td>nuclear theory</td>
      <td>strongly correlated electrons</td>
      <td>15</td>
      <td>1.11%</td>
      <td>90</td>
    </tr>
    <tr>
      <th>288</th>
      <td></td>
      <td>statistical mechanics</td>
      <td>37</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>289</th>
      <td></td>
      <td>quantum gases</td>
      <td>7</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>290</th>
      <td></td>
      <td>materials science</td>
      <td>10</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>291</th>
      <td></td>
      <td>superconductivity</td>
      <td>9</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>292</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>293</th>
      <td></td>
      <td>other condensed matter</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>294</th>
      <td></td>
      <td>soft condensed matter</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>295</th>
      <td></td>
      <td>condensed matter</td>
      <td>2</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>296</th>
      <td></td>
      <td>quantum physics</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>297</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>298</th>
      <td>high energy physics - lattice</td>
      <td>statistical mechanics</td>
      <td>65</td>
      <td>5.93%</td>
      <td>118</td>
    </tr>
    <tr>
      <th>299</th>
      <td></td>
      <td>condensed matter</td>
      <td>23</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>300</th>
      <td></td>
      <td>superconductivity</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>301</th>
      <td></td>
      <td>quantum gases</td>
      <td>4</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>302</th>
      <td></td>
      <td>strongly correlated electrons</td>
      <td>9</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>303</th>
      <td></td>
      <td>mesoscale and nanoscale physics</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>304</th>
      <td></td>
      <td>disordered systems and neural networks</td>
      <td>1</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>305</th>
      <td></td>
      <td>high energy physics - theory</td>
      <td>3</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>306</th>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>


### Display the Logs
Finally we display the logs from the Rubicon system.


```python
# Get the project

# Loop through the experiments and print details
for experiment in project.experiments():
    print(f"Experiment ID: {experiment.id}")
    print(f"Model Name: {experiment.model_name}")
    print("Parameters:")
    for param in experiment.parameters():
        print(f"  - {param.name}: {param.value}")
    print("Metrics:")
    for metric in experiment.metrics():
        print(f"  - {metric.name}: {metric.value}")
    print("------")

```

    Experiment ID: f967a29a-0c03-4270-8a70-766802a02193
    Model Name: Random Forest
    Parameters:
    Metrics:
    ------
    Experiment ID: 08961d24-105a-415f-b78f-40bdc4e03a5e
    Model Name: Random Forest
    Parameters:
    Metrics:
    ------
    Experiment ID: 924925f4-ee26-4b89-b27e-6631c79de01d
    Model Name: Random Forest
    Parameters:
      - n_estimators: 200
      - max_depth: None
      - min_samples_split: 2
      - min_samples_leaf: 1
      - min_weight_fraction_leaf: 0.0
      - max_features: sqrt
      - max_leaf_nodes: None
      - min_impurity_decrease: 0.0
      - min_impurity_split: Not set
      - bootstrap: True
      - oob_score: False
      - n_jobs: -1
      - random_state: 92
      - verbose: 1
      - warm_start: False
    Metrics:
      - Accuracy: 0.6119800521364616
      - Accuracy_Score: 0.6119800521364616
      - New_Accuracy_Score: 0.6119800521364616
      - Accuracy_Score_1: 0.6119800521364616
      - Accuracy_Score_2: 0.6119800521364616
    ------
    Experiment ID: 6ed79741-4d0b-49b4-a1be-d2f7cd025851
    Model Name: Random Forest
    Parameters:
    Metrics:
    ------
    Experiment ID: 1a36363e-6313-4d6a-b0ab-6e818a5bdf92
    Model Name: Random Forest
    Parameters:
      - n_estimators: 200
      - max_depth: None
      - min_samples_split: 2
      - min_samples_leaf: 1
      - min_weight_fraction_leaf: 0.0
      - max_features: sqrt
      - max_leaf_nodes: None
      - min_impurity_decrease: 0.0
      - min_impurity_split: Not set
      - bootstrap: True
      - oob_score: False
      - n_jobs: -1
      - random_state: 92
      - verbose: 1
      - warm_start: False
    Metrics:
      - Accuracy_Score_2: 0.6119800521364616
    ------
    Experiment ID: 3b95e82b-c76a-45cb-b1c6-0d91a0728618
    Model Name: Random Forest
    Parameters:
      - n_estimators: 200
      - max_depth: None
      - min_samples_split: 2
      - min_samples_leaf: 1
      - min_weight_fraction_leaf: 0.0
      - max_features: sqrt
      - max_leaf_nodes: None
      - min_impurity_decrease: 0.0
      - min_impurity_split: Not set
      - bootstrap: True
      - oob_score: False
      - n_jobs: -1
      - random_state: 92
      - verbose: 1
      - warm_start: False
    Metrics:
      - Accuracy_Score_2: 0.6119800521364616
    ------
    Experiment ID: 228131f9-da73-4ffc-823f-b85ff225321c
    Model Name: Random Forest
    Parameters:
    Metrics:
    ------
    Experiment ID: 624c6e9e-59c3-4464-b86e-bc7d1677b39a
    Model Name: Random Forest
    Parameters:
    Metrics:
    ------
    Experiment ID: a6f9b5e1-f475-41b5-86fd-ed6683daa9b8
    Model Name: Random Forest
    Parameters:
      - n_estimators: 200
      - max_depth: None
      - min_samples_split: 2
      - min_samples_leaf: 1
      - min_weight_fraction_leaf: 0.0
      - max_features: sqrt
      - max_leaf_nodes: None
      - min_impurity_decrease: 0.0
      - min_impurity_split: Not set
      - bootstrap: True
      - oob_score: False
      - n_jobs: -1
      - random_state: 92
      - verbose: 1
      - warm_start: False
    Metrics:
      - Accuracy_Score_2: 0.6119800521364616
    ------



```python

```




    defaultdict(<function __main__.<lambda>()>,
                {(2013,
                  4): defaultdict(int,
                             {'mesoscale and nanoscale physics': 62,
                              'materials science': 56,
                              'superconductivity': 40,
                              'soft condensed matter': 16,
                              'strongly correlated electrons': 38,
                              'quantum physics': 5,
                              'quantum gases': 12,
                              'statistical mechanics': 41,
                              'high energy physics - theory': 1}),
                 (1998,
                  12): defaultdict(int,
                             {'mesoscale and nanoscale physics': 20,
                              'strongly correlated electrons': 16,
                              'statistical mechanics': 33,
                              'condensed matter': 8,
                              'soft condensed matter': 5,
                              'superconductivity': 10,
                              'materials science': 8,
                              'high energy physics - theory': 1,
                              'disordered systems and neural networks': 2}),
                 (2013,
                  12): defaultdict(int,
                             {'mesoscale and nanoscale physics': 56,
                              'statistical mechanics': 34,
                              'disordered systems and neural networks': 2,
                              'strongly correlated electrons': 34,
                              'soft condensed matter': 15,
                              'materials science': 53,
                              'superconductivity': 25,
                              'quantum gases': 11,
                              'quantum physics': 4,
                              'other condensed matter': 1,
                              'high energy physics - theory': 2,
                              'physics - physics and society': 1}),
                 (2011,
                  10): defaultdict(int,
                             {'strongly correlated electrons': 41,
                              'mesoscale and nanoscale physics': 48,
                              'soft condensed matter': 15,
                              'statistical mechanics': 35,
                              'superconductivity': 30,
                              'disordered systems and neural networks': 2,
                              'quantum physics': 4,
                              'materials science': 37,
                              'quantum gases': 17,
                              'high energy physics - theory': 1,
                              'physics - physics and society': 1}),
                 (2005,
                  4): defaultdict(int,
                             {'superconductivity': 20,
                              'mesoscale and nanoscale physics': 40,
                              'materials science': 30,
                              'strongly correlated electrons': 30,
                              'statistical mechanics': 42,
                              'other condensed matter': 4,
                              'disordered systems and neural networks': 7,
                              'soft condensed matter': 10,
                              'quantum physics': 1}),
                 (2012,
                  10): defaultdict(int,
                             {'statistical mechanics': 46,
                              'mesoscale and nanoscale physics': 55,
                              'materials science': 60,
                              'superconductivity': 31,
                              'disordered systems and neural networks': 5,
                              'strongly correlated electrons': 31,
                              'soft condensed matter': 19,
                              'quantum gases': 13,
                              'quantum physics': 7,
                              'high energy physics - theory': 1,
                              'other condensed matter': 1}),
                 (2010,
                  9): defaultdict(int,
                             {'superconductivity': 25,
                              'statistical mechanics': 46,
                              'soft condensed matter': 18,
                              'mesoscale and nanoscale physics': 42,
                              'quantum gases': 13,
                              'materials science': 26,
                              'quantum physics': 4,
                              'strongly correlated electrons': 26,
                              'physics - physics and society': 1,
                              'disordered systems and neural networks': 2}),
                 (2012,
                  1): defaultdict(int,
                             {'materials science': 40,
                              'strongly correlated electrons': 29,
                              'statistical mechanics': 36,
                              'mesoscale and nanoscale physics': 44,
                              'quantum gases': 18,
                              'mathematical physics': 1,
                              'superconductivity': 11,
                              'soft condensed matter': 17,
                              'quantum physics': 3,
                              'high energy physics - theory': 1,
                              'disordered systems and neural networks': 1}),
                 (2008,
                  1): defaultdict(int,
                             {'mesoscale and nanoscale physics': 41,
                              'strongly correlated electrons': 34,
                              'materials science': 22,
                              'other condensed matter': 7,
                              'superconductivity': 18,
                              'statistical mechanics': 26,
                              'soft condensed matter': 12,
                              'quantum gases': 1,
                              'quantum physics': 1,
                              'disordered systems and neural networks': 2}),
                 (2003,
                  5): defaultdict(int,
                             {'strongly correlated electrons': 26,
                              'soft condensed matter': 12,
                              'statistical mechanics': 61,
                              'superconductivity': 28,
                              'materials science': 21,
                              'mesoscale and nanoscale physics': 25,
                              'other condensed matter': 3,
                              'disordered systems and neural networks': 7,
                              'condensed matter': 1,
                              'quantum gases': 1,
                              'quantum physics': 1}),
                 (2009,
                  11): defaultdict(int,
                             {'materials science': 41,
                              'quantum physics': 5,
                              'superconductivity': 25,
                              'statistical mechanics': 27,
                              'soft condensed matter': 17,
                              'mesoscale and nanoscale physics': 50,
                              'quantum gases': 9,
                              'strongly correlated electrons': 26,
                              'mathematical physics': 1,
                              'disordered systems and neural networks': 1,
                              'other condensed matter': 1}),
                 (2007,
                  9): defaultdict(int,
                             {'materials science': 36,
                              'mesoscale and nanoscale physics': 28,
                              'statistical mechanics': 27,
                              'other condensed matter': 5,
                              'soft condensed matter': 20,
                              'superconductivity': 15,
                              'disordered systems and neural networks': 5,
                              'strongly correlated electrons': 26,
                              'quantum gases': 3,
                              'quantum physics': 2}),
                 (2008,
                  5): defaultdict(int,
                             {'materials science': 26,
                              'mesoscale and nanoscale physics': 34,
                              'soft condensed matter': 13,
                              'superconductivity': 23,
                              'statistical mechanics': 27,
                              'disordered systems and neural networks': 3,
                              'strongly correlated electrons': 23,
                              'other condensed matter': 6,
                              'quantum physics': 4,
                              'quantum gases': 2}),
                 (2005,
                  11): defaultdict(int,
                             {'mesoscale and nanoscale physics': 34,
                              'strongly correlated electrons': 33,
                              'other condensed matter': 6,
                              'soft condensed matter': 12,
                              'statistical mechanics': 34,
                              'materials science': 23,
                              'superconductivity': 16,
                              'disordered systems and neural networks': 3,
                              'high energy physics - theory': 1,
                              'quantum physics': 1}),
                 (2002,
                  7): defaultdict(int,
                             {'statistical mechanics': 50,
                              'disordered systems and neural networks': 3,
                              'mesoscale and nanoscale physics': 15,
                              'strongly correlated electrons': 28,
                              'materials science': 21,
                              'soft condensed matter': 11,
                              'superconductivity': 24,
                              'quantum gases': 1,
                              'condensed matter': 1,
                              'quantum physics': 2}),
                 (2012,
                  12): defaultdict(int,
                             {'strongly correlated electrons': 26,
                              'mesoscale and nanoscale physics': 51,
                              'materials science': 41,
                              'superconductivity': 18,
                              'soft condensed matter': 17,
                              'quantum gases': 16,
                              'statistical mechanics': 36,
                              'quantum physics': 7,
                              'high energy physics - lattice': 1,
                              'disordered systems and neural networks': 3,
                              'high energy physics - theory': 1}),
                 (2008,
                  11): defaultdict(int,
                             {'mesoscale and nanoscale physics': 36,
                              'materials science': 21,
                              'superconductivity': 26,
                              'strongly correlated electrons': 25,
                              'statistical mechanics': 39,
                              'disordered systems and neural networks': 1,
                              'soft condensed matter': 8,
                              'quantum gases': 1,
                              'quantum physics': 1,
                              'other condensed matter': 8}),
                 (2013,
                  1): defaultdict(int,
                             {'strongly correlated electrons': 37,
                              'materials science': 66,
                              'mesoscale and nanoscale physics': 43,
                              'statistical mechanics': 35,
                              'quantum physics': 5,
                              'superconductivity': 22,
                              'soft condensed matter': 16,
                              'quantum gases': 19,
                              'disordered systems and neural networks': 2}),
                 (2010,
                  8): defaultdict(int,
                             {'materials science': 41,
                              'superconductivity': 21,
                              'statistical mechanics': 34,
                              'mesoscale and nanoscale physics': 43,
                              'quantum gases': 14,
                              'strongly correlated electrons': 27,
                              'soft condensed matter': 10,
                              'quantum physics': 5,
                              'other condensed matter': 1,
                              'high energy physics - theory': 1,
                              'disordered systems and neural networks': 1}),
                 (2003,
                  12): defaultdict(int,
                             {'superconductivity': 20,
                              'strongly correlated electrons': 26,
                              'other condensed matter': 4,
                              'mesoscale and nanoscale physics': 31,
                              'statistical mechanics': 48,
                              'condensed matter': 1,
                              'soft condensed matter': 10,
                              'materials science': 14,
                              'disordered systems and neural networks': 3,
                              'quantum physics': 3,
                              'quantum gases': 1}),
                 (2008,
                  9): defaultdict(int,
                             {'strongly correlated electrons': 33,
                              'superconductivity': 26,
                              'statistical mechanics': 36,
                              'mesoscale and nanoscale physics': 37,
                              'materials science': 24,
                              'quantum physics': 1,
                              'soft condensed matter': 6,
                              'other condensed matter': 4,
                              'quantum gases': 4,
                              'high energy physics - theory': 2,
                              'disordered systems and neural networks': 1,
                              'physics - physics and society': 1}),
                 (2013,
                  5): defaultdict(int,
                             {'materials science': 53,
                              'statistical mechanics': 47,
                              'mesoscale and nanoscale physics': 63,
                              'strongly correlated electrons': 22,
                              'quantum gases': 11,
                              'soft condensed matter': 19,
                              'quantum physics': 7,
                              'superconductivity': 18,
                              'high energy physics - theory': 1,
                              'disordered systems and neural networks': 1}),
                 (2006,
                  5): defaultdict(int,
                             {'statistical mechanics': 44,
                              'quantum gases': 3,
                              'strongly correlated electrons': 24,
                              'superconductivity': 22,
                              'mesoscale and nanoscale physics': 22,
                              'materials science': 26,
                              'other condensed matter': 8,
                              'soft condensed matter': 17,
                              'disordered systems and neural networks': 2,
                              'quantum physics': 1}),
                 (2007,
                  4): defaultdict(int,
                             {'mesoscale and nanoscale physics': 32,
                              'statistical mechanics': 29,
                              'strongly correlated electrons': 32,
                              'materials science': 32,
                              'high energy physics - theory': 1,
                              'superconductivity': 16,
                              'soft condensed matter': 14,
                              'quantum physics': 3,
                              'disordered systems and neural networks': 1,
                              'other condensed matter': 3,
                              'quantum gases': 1}),
                 (2007,
                  10): defaultdict(int,
                             {'statistical mechanics': 40,
                              'mesoscale and nanoscale physics': 44,
                              'materials science': 28,
                              'superconductivity': 13,
                              'quantum gases': 4,
                              'soft condensed matter': 14,
                              'strongly correlated electrons': 33,
                              'other condensed matter': 7,
                              'quantum physics': 1,
                              'disordered systems and neural networks': 1}),
                 (1998,
                  4): defaultdict(int,
                             {'strongly correlated electrons': 19,
                              'superconductivity': 11,
                              'statistical mechanics': 35,
                              'mesoscale and nanoscale physics': 6,
                              'soft condensed matter': 3,
                              'disordered systems and neural networks': 3,
                              'condensed matter': 2,
                              'materials science': 2}),
                 (2013,
                  11): defaultdict(int,
                             {'mesoscale and nanoscale physics': 53,
                              'soft condensed matter': 21,
                              'disordered systems and neural networks': 3,
                              'superconductivity': 24,
                              'strongly correlated electrons': 30,
                              'materials science': 61,
                              'quantum gases': 10,
                              'statistical mechanics': 32,
                              'high energy physics - theory': 2,
                              'quantum physics': 8,
                              'physics - physics and society': 1}),
                 (2005,
                  9): defaultdict(int,
                             {'materials science': 31,
                              'quantum physics': 2,
                              'disordered systems and neural networks': 5,
                              'soft condensed matter': 15,
                              'strongly correlated electrons': 23,
                              'superconductivity': 22,
                              'statistical mechanics': 45,
                              'mesoscale and nanoscale physics': 24,
                              'other condensed matter': 11,
                              'quantum gases': 1,
                              'high energy physics - theory': 1}),
                 (2004,
                  8): defaultdict(int,
                             {'statistical mechanics': 42,
                              'mesoscale and nanoscale physics': 25,
                              'superconductivity': 19,
                              'soft condensed matter': 9,
                              'quantum physics': 2,
                              'strongly correlated electrons': 22,
                              'quantum gases': 1,
                              'materials science': 27,
                              'other condensed matter': 2,
                              'disordered systems and neural networks': 3}),
                 (1998,
                  7): defaultdict(int,
                             {'mesoscale and nanoscale physics': 13,
                              'materials science': 3,
                              'statistical mechanics': 37,
                              'strongly correlated electrons': 20,
                              'superconductivity': 4,
                              'quantum gases': 1,
                              'condensed matter': 5,
                              'disordered systems and neural networks': 3,
                              'soft condensed matter': 2}),
                 (2010,
                  6): defaultdict(int,
                             {'mesoscale and nanoscale physics': 49,
                              'superconductivity': 31,
                              'strongly correlated electrons': 38,
                              'soft condensed matter': 18,
                              'materials science': 26,
                              'statistical mechanics': 34,
                              'quantum gases': 21,
                              'quantum physics': 3,
                              'high energy physics - theory': 1,
                              'disordered systems and neural networks': 3,
                              'other condensed matter': 2}),
                 (2000,
                  5): defaultdict(int,
                             {'statistical mechanics': 41,
                              'strongly correlated electrons': 18,
                              'superconductivity': 14,
                              'soft condensed matter': 10,
                              'mesoscale and nanoscale physics': 12,
                              'disordered systems and neural networks': 5,
                              'materials science': 9,
                              'condensed matter': 3,
                              'other condensed matter': 1}),
                 (1999,
                  8): defaultdict(int,
                             {'statistical mechanics': 38,
                              'strongly correlated electrons': 21,
                              'superconductivity': 11,
                              'mesoscale and nanoscale physics': 21,
                              'condensed matter': 1,
                              'disordered systems and neural networks': 2,
                              'soft condensed matter': 5,
                              'materials science': 3}),
                 (1997,
                  6): defaultdict(int,
                             {'statistical mechanics': 25,
                              'superconductivity': 6,
                              'strongly correlated electrons': 16,
                              'mesoscale and nanoscale physics': 13,
                              'condensed matter': 3,
                              'materials science': 4,
                              'disordered systems and neural networks': 1,
                              'soft condensed matter': 2}),
                 (2002,
                  8): defaultdict(int,
                             {'materials science': 13,
                              'other condensed matter': 2,
                              'superconductivity': 12,
                              'soft condensed matter': 9,
                              'statistical mechanics': 44,
                              'high energy physics - theory': 1,
                              'mesoscale and nanoscale physics': 17,
                              'strongly correlated electrons': 13,
                              'disordered systems and neural networks': 2,
                              'quantum gases': 2,
                              'condensed matter': 1}),
                 (2006,
                  10): defaultdict(int,
                             {'materials science': 33,
                              'superconductivity': 23,
                              'quantum gases': 3,
                              'soft condensed matter': 15,
                              'strongly correlated electrons': 30,
                              'mesoscale and nanoscale physics': 40,
                              'statistical mechanics': 30,
                              'disordered systems and neural networks': 5,
                              'other condensed matter': 9,
                              'quantum physics': 1}),
                 (2010,
                  11): defaultdict(int,
                             {'statistical mechanics': 44,
                              'materials science': 44,
                              'mesoscale and nanoscale physics': 43,
                              'strongly correlated electrons': 30,
                              'quantum gases': 15,
                              'superconductivity': 21,
                              'soft condensed matter': 23,
                              'quantum physics': 4,
                              'other condensed matter': 1}),
                 (2012,
                  11): defaultdict(int,
                             {'soft condensed matter': 22,
                              'mesoscale and nanoscale physics': 47,
                              'statistical mechanics': 50,
                              'quantum gases': 19,
                              'strongly correlated electrons': 34,
                              'quantum physics': 8,
                              'superconductivity': 20,
                              'materials science': 39,
                              'physics - physics and society': 1,
                              'disordered systems and neural networks': 1,
                              'other condensed matter': 1}),
                 (2011,
                  4): defaultdict(int,
                             {'statistical mechanics': 39,
                              'strongly correlated electrons': 41,
                              'superconductivity': 19,
                              'mesoscale and nanoscale physics': 46,
                              'materials science': 31,
                              'quantum gases': 7,
                              'disordered systems and neural networks': 5,
                              'quantum physics': 7,
                              'soft condensed matter': 14}),
                 (1999,
                  1): defaultdict(int,
                             {'condensed matter': 4,
                              'materials science': 10,
                              'statistical mechanics': 23,
                              'strongly correlated electrons': 16,
                              'superconductivity': 8,
                              'mesoscale and nanoscale physics': 9,
                              'quantum gases': 2,
                              'disordered systems and neural networks': 3,
                              'soft condensed matter': 4}),
                 (2001,
                  7): defaultdict(int,
                             {'statistical mechanics': 38,
                              'mesoscale and nanoscale physics': 18,
                              'superconductivity': 25,
                              'disordered systems and neural networks': 7,
                              'strongly correlated electrons': 32,
                              'materials science': 19,
                              'condensed matter': 3,
                              'soft condensed matter': 7}),
                 (2011,
                  9): defaultdict(int,
                             {'materials science': 32,
                              'strongly correlated electrons': 32,
                              'mesoscale and nanoscale physics': 52,
                              'statistical mechanics': 38,
                              'superconductivity': 28,
                              'quantum gases': 17,
                              'high energy physics - theory': 1,
                              'soft condensed matter': 13,
                              'quantum physics': 5,
                              'disordered systems and neural networks': 4}),
                 (2000,
                  8): defaultdict(int,
                             {'materials science': 11,
                              'statistical mechanics': 40,
                              'mesoscale and nanoscale physics': 12,
                              'superconductivity': 12,
                              'soft condensed matter': 7,
                              'condensed matter': 6,
                              'disordered systems and neural networks': 1,
                              'other condensed matter': 1,
                              'strongly correlated electrons': 16}),
                 (2012,
                  3): defaultdict(int,
                             {'soft condensed matter': 19,
                              'mesoscale and nanoscale physics': 60,
                              'quantum gases': 10,
                              'materials science': 46,
                              'statistical mechanics': 47,
                              'superconductivity': 28,
                              'physics - physics and society': 1,
                              'strongly correlated electrons': 26,
                              'disordered systems and neural networks': 2,
                              'quantum physics': 3}),
                 (2004,
                  1): defaultdict(int,
                             {'materials science': 27,
                              'statistical mechanics': 40,
                              'soft condensed matter': 6,
                              'superconductivity': 16,
                              'strongly correlated electrons': 21,
                              'other condensed matter': 6,
                              'mesoscale and nanoscale physics': 21,
                              'disordered systems and neural networks': 1,
                              'quantum physics': 3}),
                 (2014,
                  2): defaultdict(int,
                             {'mesoscale and nanoscale physics': 38,
                              'statistical mechanics': 41,
                              'strongly correlated electrons': 31,
                              'superconductivity': 30,
                              'materials science': 43,
                              'soft condensed matter': 21,
                              'quantum gases': 13,
                              'quantum physics': 3,
                              'disordered systems and neural networks': 2,
                              'high energy physics - theory': 1}),
                 (2006,
                  3): defaultdict(int,
                             {'strongly correlated electrons': 39,
                              'mesoscale and nanoscale physics': 30,
                              'materials science': 27,
                              'superconductivity': 19,
                              'soft condensed matter': 12,
                              'statistical mechanics': 43,
                              'disordered systems and neural networks': 5,
                              'other condensed matter': 9,
                              'quantum gases': 2}),
                 (2003,
                  7): defaultdict(int,
                             {'mesoscale and nanoscale physics': 23,
                              'soft condensed matter': 8,
                              'strongly correlated electrons': 23,
                              'statistical mechanics': 54,
                              'disordered systems and neural networks': 5,
                              'superconductivity': 25,
                              'materials science': 23,
                              'other condensed matter': 2,
                              'condensed matter': 3,
                              'quantum gases': 2,
                              'quantum physics': 2}),
                 (2006,
                  7): defaultdict(int,
                             {'statistical mechanics': 44,
                              'mesoscale and nanoscale physics': 35,
                              'strongly correlated electrons': 28,
                              'superconductivity': 26,
                              'materials science': 33,
                              'other condensed matter': 10,
                              'quantum physics': 1,
                              'soft condensed matter': 9,
                              'disordered systems and neural networks': 2,
                              'quantum gases': 1}),
                 (1993,
                  7): defaultdict(int,
                             {'materials science': 1, 'condensed matter': 12}),
                 (2013,
                  9): defaultdict(int,
                             {'materials science': 53,
                              'statistical mechanics': 35,
                              'mesoscale and nanoscale physics': 74,
                              'disordered systems and neural networks': 1,
                              'strongly correlated electrons': 31,
                              'soft condensed matter': 24,
                              'quantum gases': 20,
                              'superconductivity': 21,
                              'quantum physics': 6}),
                 (2006,
                  9): defaultdict(int,
                             {'strongly correlated electrons': 28,
                              'other condensed matter': 13,
                              'materials science': 29,
                              'statistical mechanics': 41,
                              'disordered systems and neural networks': 3,
                              'soft condensed matter': 12,
                              'superconductivity': 23,
                              'mesoscale and nanoscale physics': 32,
                              'quantum gases': 1,
                              'high energy physics - theory': 1}),
                 (2010,
                  10): defaultdict(int,
                             {'materials science': 38,
                              'mesoscale and nanoscale physics': 64,
                              'superconductivity': 19,
                              'quantum gases': 18,
                              'physics - physics and society': 2,
                              'physics - biological physics': 1,
                              'soft condensed matter': 17,
                              'disordered systems and neural networks': 4,
                              'strongly correlated electrons': 23,
                              'statistical mechanics': 41,
                              'quantum physics': 4,
                              'high energy physics - theory': 1}),
                 (2010,
                  5): defaultdict(int,
                             {'mesoscale and nanoscale physics': 38,
                              'superconductivity': 23,
                              'strongly correlated electrons': 44,
                              'materials science': 36,
                              'statistical mechanics': 27,
                              'soft condensed matter': 18,
                              'quantum physics': 3,
                              'quantum gases': 19,
                              'disordered systems and neural networks': 3,
                              'other condensed matter': 1,
                              'high energy physics - theory': 2}),
                 (2001,
                  5): defaultdict(int,
                             {'statistical mechanics': 39,
                              'strongly correlated electrons': 17,
                              'mesoscale and nanoscale physics': 22,
                              'other condensed matter': 1,
                              'condensed matter': 3,
                              'superconductivity': 20,
                              'materials science': 9,
                              'soft condensed matter': 6,
                              'disordered systems and neural networks': 3,
                              'quantum physics': 1}),
                 (1996,
                  9): defaultdict(int,
                             {'condensed matter': 68,
                              'mesoscale and nanoscale physics': 1,
                              'materials science': 1}),
                 (2001,
                  9): defaultdict(int,
                             {'statistical mechanics': 36,
                              'superconductivity': 14,
                              'materials science': 10,
                              'mesoscale and nanoscale physics': 24,
                              'strongly correlated electrons': 25,
                              'other condensed matter': 1,
                              'soft condensed matter': 6,
                              'disordered systems and neural networks': 1,
                              'condensed matter': 3,
                              'quantum physics': 1}),
                 (2006,
                  8): defaultdict(int,
                             {'materials science': 22,
                              'quantum gases': 2,
                              'mesoscale and nanoscale physics': 30,
                              'strongly correlated electrons': 24,
                              'soft condensed matter': 7,
                              'superconductivity': 14,
                              'statistical mechanics': 36,
                              'other condensed matter': 5,
                              'disordered systems and neural networks': 2,
                              'high energy physics - theory': 1}),
                 (2011,
                  6): defaultdict(int,
                             {'mesoscale and nanoscale physics': 43,
                              'statistical mechanics': 40,
                              'materials science': 54,
                              'superconductivity': 25,
                              'soft condensed matter': 20,
                              'quantum physics': 4,
                              'strongly correlated electrons': 28,
                              'quantum gases': 14,
                              'disordered systems and neural networks': 2}),
                 (2004,
                  2): defaultdict(int,
                             {'superconductivity': 25,
                              'strongly correlated electrons': 21,
                              'statistical mechanics': 39,
                              'mesoscale and nanoscale physics': 28,
                              'materials science': 20,
                              'other condensed matter': 10,
                              'soft condensed matter': 16,
                              'quantum physics': 1,
                              'disordered systems and neural networks': 3}),
                 (2004,
                  11): defaultdict(int,
                             {'strongly correlated electrons': 25,
                              'statistical mechanics': 41,
                              'superconductivity': 17,
                              'soft condensed matter': 12,
                              'mesoscale and nanoscale physics': 36,
                              'materials science': 25,
                              'other condensed matter': 4,
                              'quantum physics': 2,
                              'quantum gases': 2,
                              'disordered systems and neural networks': 1}),
                 (1999,
                  11): defaultdict(int,
                             {'strongly correlated electrons': 16,
                              'soft condensed matter': 9,
                              'mesoscale and nanoscale physics': 20,
                              'superconductivity': 16,
                              'statistical mechanics': 43,
                              'materials science': 8,
                              'condensed matter': 6,
                              'disordered systems and neural networks': 3}),
                 (2010,
                  7): defaultdict(int,
                             {'soft condensed matter': 15,
                              'materials science': 51,
                              'statistical mechanics': 30,
                              'superconductivity': 20,
                              'mesoscale and nanoscale physics': 58,
                              'strongly correlated electrons': 30,
                              'quantum physics': 8,
                              'quantum gases': 13,
                              'disordered systems and neural networks': 5,
                              'high energy physics - theory': 1}),
                 (1999,
                  6): defaultdict(int,
                             {'mesoscale and nanoscale physics': 14,
                              'statistical mechanics': 32,
                              'strongly correlated electrons': 16,
                              'superconductivity': 13,
                              'high energy physics - theory': 2,
                              'soft condensed matter': 1,
                              'condensed matter': 2,
                              'materials science': 6,
                              'disordered systems and neural networks': 3}),
                 (2012,
                  2): defaultdict(int,
                             {'mesoscale and nanoscale physics': 59,
                              'materials science': 34,
                              'statistical mechanics': 40,
                              'superconductivity': 25,
                              'quantum gases': 20,
                              'disordered systems and neural networks': 3,
                              'strongly correlated electrons': 36,
                              'soft condensed matter': 10,
                              'quantum physics': 8}),
                 (1995, 9): defaultdict(int, {'condensed matter': 37}),
                 (2011,
                  3): defaultdict(int,
                             {'mesoscale and nanoscale physics': 49,
                              'statistical mechanics': 39,
                              'soft condensed matter': 20,
                              'quantum gases': 12,
                              'strongly correlated electrons': 27,
                              'superconductivity': 20,
                              'materials science': 40,
                              'quantum physics': 4,
                              'disordered systems and neural networks': 2,
                              'other condensed matter': 1}),
                 (2005,
                  12): defaultdict(int,
                             {'mesoscale and nanoscale physics': 38,
                              'statistical mechanics': 37,
                              'strongly correlated electrons': 26,
                              'materials science': 19,
                              'soft condensed matter': 16,
                              'other condensed matter': 5,
                              'superconductivity': 13,
                              'high energy physics - theory': 1,
                              'disordered systems and neural networks': 1}),
                 (2013,
                  3): defaultdict(int,
                             {'strongly correlated electrons': 37,
                              'superconductivity': 32,
                              'statistical mechanics': 32,
                              'materials science': 67,
                              'mesoscale and nanoscale physics': 62,
                              'quantum gases': 10,
                              'soft condensed matter': 20,
                              'disordered systems and neural networks': 2,
                              'quantum physics': 3}),
                 (1998,
                  8): defaultdict(int,
                             {'mesoscale and nanoscale physics': 9,
                              'statistical mechanics': 21,
                              'strongly correlated electrons': 16,
                              'superconductivity': 8,
                              'condensed matter': 3,
                              'materials science': 7,
                              'disordered systems and neural networks': 3,
                              'soft condensed matter': 3}),
                 (2000,
                  1): defaultdict(int,
                             {'statistical mechanics': 28,
                              'soft condensed matter': 5,
                              'materials science': 9,
                              'strongly correlated electrons': 18,
                              'disordered systems and neural networks': 4,
                              'mesoscale and nanoscale physics': 16,
                              'superconductivity': 16,
                              'condensed matter': 2,
                              'high energy physics - theory': 1}),
                 (2013,
                  6): defaultdict(int,
                             {'strongly correlated electrons': 37,
                              'soft condensed matter': 26,
                              'materials science': 49,
                              'mesoscale and nanoscale physics': 47,
                              'statistical mechanics': 35,
                              'quantum physics': 3,
                              'quantum gases': 13,
                              'physics - physics and society': 1,
                              'superconductivity': 19,
                              'disordered systems and neural networks': 3}),
                 (2011,
                  7): defaultdict(int,
                             {'soft condensed matter': 22,
                              'quantum physics': 6,
                              'strongly correlated electrons': 27,
                              'quantum gases': 14,
                              'statistical mechanics': 31,
                              'mesoscale and nanoscale physics': 40,
                              'physics - atomic physics': 1,
                              'materials science': 35,
                              'superconductivity': 29,
                              'high energy physics - theory': 1,
                              'disordered systems and neural networks': 1}),
                 (2011,
                  8): defaultdict(int,
                             {'quantum gases': 23,
                              'materials science': 55,
                              'strongly correlated electrons': 28,
                              'soft condensed matter': 15,
                              'statistical mechanics': 43,
                              'mesoscale and nanoscale physics': 53,
                              'superconductivity': 22,
                              'quantum physics': 3,
                              'high energy physics - theory': 1,
                              'other condensed matter': 1}),
                 (2009,
                  2): defaultdict(int,
                             {'soft condensed matter': 13,
                              'mesoscale and nanoscale physics': 34,
                              'statistical mechanics': 34,
                              'materials science': 33,
                              'disordered systems and neural networks': 6,
                              'strongly correlated electrons': 17,
                              'quantum gases': 8,
                              'superconductivity': 22,
                              'quantum physics': 3}),
                 (1998,
                  9): defaultdict(int,
                             {'mesoscale and nanoscale physics': 11,
                              'statistical mechanics': 31,
                              'strongly correlated electrons': 18,
                              'superconductivity': 15,
                              'materials science': 7,
                              'soft condensed matter': 7,
                              'disordered systems and neural networks': 3,
                              'condensed matter': 1}),
                 (2008,
                  4): defaultdict(int,
                             {'mesoscale and nanoscale physics': 38,
                              'superconductivity': 25,
                              'strongly correlated electrons': 31,
                              'soft condensed matter': 6,
                              'disordered systems and neural networks': 2,
                              'materials science': 30,
                              'statistical mechanics': 37,
                              'other condensed matter': 6,
                              'high energy physics - theory': 1,
                              'quantum physics': 1}),
                 (2000,
                  2): defaultdict(int,
                             {'strongly correlated electrons': 16,
                              'statistical mechanics': 39,
                              'superconductivity': 20,
                              'mesoscale and nanoscale physics': 14,
                              'condensed matter': 1,
                              'soft condensed matter': 5,
                              'materials science': 4,
                              'quantum physics': 1,
                              'disordered systems and neural networks': 1,
                              'other condensed matter': 1}),
                 (2000,
                  10): defaultdict(int,
                             {'mesoscale and nanoscale physics': 18,
                              'superconductivity': 13,
                              'strongly correlated electrons': 23,
                              'statistical mechanics': 36,
                              'soft condensed matter': 5,
                              'materials science': 7,
                              'disordered systems and neural networks': 4}),
                 (2001,
                  6): defaultdict(int,
                             {'statistical mechanics': 40,
                              'strongly correlated electrons': 33,
                              'materials science': 16,
                              'superconductivity': 25,
                              'mesoscale and nanoscale physics': 25,
                              'condensed matter': 4,
                              'soft condensed matter': 9,
                              'disordered systems and neural networks': 3,
                              'other condensed matter': 1,
                              'high energy physics - theory': 1}),
                 (2003,
                  6): defaultdict(int,
                             {'materials science': 17,
                              'soft condensed matter': 15,
                              'statistical mechanics': 41,
                              'condensed matter': 3,
                              'strongly correlated electrons': 28,
                              'mesoscale and nanoscale physics': 25,
                              'superconductivity': 34,
                              'other condensed matter': 3,
                              'disordered systems and neural networks': 5}),
                 (2009,
                  4): defaultdict(int,
                             {'strongly correlated electrons': 29,
                              'quantum physics': 5,
                              'materials science': 38,
                              'mesoscale and nanoscale physics': 29,
                              'quantum gases': 9,
                              'superconductivity': 21,
                              'statistical mechanics': 26,
                              'soft condensed matter': 7,
                              'disordered systems and neural networks': 2,
                              'other condensed matter': 2}),
                 (2006,
                  6): defaultdict(int,
                             {'strongly correlated electrons': 40,
                              'mesoscale and nanoscale physics': 34,
                              'superconductivity': 19,
                              'statistical mechanics': 32,
                              'quantum physics': 2,
                              'soft condensed matter': 8,
                              'materials science': 29,
                              'high energy physics - theory': 1,
                              'other condensed matter': 5,
                              'disordered systems and neural networks': 4}),
                 (2004,
                  5): defaultdict(int,
                             {'mesoscale and nanoscale physics': 28,
                              'statistical mechanics': 42,
                              'materials science': 27,
                              'strongly correlated electrons': 26,
                              'quantum physics': 4,
                              'soft condensed matter': 14,
                              'superconductivity': 16,
                              'other condensed matter': 6,
                              'disordered systems and neural networks': 6,
                              'quantum gases': 1}),
                 (1997,
                  3): defaultdict(int,
                             {'superconductivity': 8,
                              'strongly correlated electrons': 17,
                              'statistical mechanics': 17,
                              'materials science': 6,
                              'quantum gases': 1,
                              'mesoscale and nanoscale physics': 7,
                              'condensed matter': 1,
                              'disordered systems and neural networks': 2}),
                 (2007,
                  3): defaultdict(int,
                             {'statistical mechanics': 42,
                              'quantum gases': 1,
                              'strongly correlated electrons': 23,
                              'materials science': 36,
                              'soft condensed matter': 14,
                              'nuclear theory': 1,
                              'disordered systems and neural networks': 4,
                              'superconductivity': 17,
                              'other condensed matter': 9,
                              'mesoscale and nanoscale physics': 32}),
                 (1999,
                  2): defaultdict(int,
                             {'condensed matter': 3,
                              'superconductivity': 14,
                              'materials science': 5,
                              'strongly correlated electrons': 17,
                              'statistical mechanics': 25,
                              'mesoscale and nanoscale physics': 11,
                              'quantum gases': 1,
                              'high energy physics - theory': 1,
                              'soft condensed matter': 1,
                              'disordered systems and neural networks': 1}),
                 (2002,
                  9): defaultdict(int,
                             {'statistical mechanics': 40,
                              'mesoscale and nanoscale physics': 38,
                              'soft condensed matter': 11,
                              'strongly correlated electrons': 16,
                              'superconductivity': 18,
                              'condensed matter': 1,
                              'high energy physics - theory': 2,
                              'materials science': 12,
                              'quantum physics': 1,
                              'disordered systems and neural networks': 3,
                              'quantum gases': 2,
                              'other condensed matter': 1}),
                 (2009,
                  10): defaultdict(int,
                             {'materials science': 40,
                              'mesoscale and nanoscale physics': 34,
                              'superconductivity': 29,
                              'statistical mechanics': 49,
                              'strongly correlated electrons': 27,
                              'quantum gases': 12,
                              'disordered systems and neural networks': 4,
                              'other condensed matter': 1,
                              'soft condensed matter': 17,
                              'quantum physics': 2}),
                 (2010,
                  3): defaultdict(int,
                             {'materials science': 35,
                              'statistical mechanics': 44,
                              'mesoscale and nanoscale physics': 56,
                              'strongly correlated electrons': 31,
                              'quantum physics': 7,
                              'soft condensed matter': 13,
                              'disordered systems and neural networks': 4,
                              'superconductivity': 23,
                              'quantum gases': 12,
                              'high energy physics - theory': 1}),
                 (2008,
                  7): defaultdict(int,
                             {'statistical mechanics': 41,
                              'strongly correlated electrons': 27,
                              'mesoscale and nanoscale physics': 45,
                              'materials science': 29,
                              'superconductivity': 33,
                              'soft condensed matter': 17,
                              'other condensed matter': 12,
                              'quantum gases': 2,
                              'disordered systems and neural networks': 1}),
                 (2002,
                  3): defaultdict(int,
                             {'materials science': 18,
                              'mesoscale and nanoscale physics': 20,
                              'statistical mechanics': 48,
                              'strongly correlated electrons': 20,
                              'superconductivity': 21,
                              'condensed matter': 2,
                              'quantum physics': 1,
                              'high energy physics - theory': 2,
                              'soft condensed matter': 9,
                              'disordered systems and neural networks': 3,
                              'other condensed matter': 1,
                              'quantum gases': 1}),
                 (2012,
                  5): defaultdict(int,
                             {'quantum physics': 4,
                              'mesoscale and nanoscale physics': 54,
                              'materials science': 40,
                              'statistical mechanics': 47,
                              'quantum gases': 15,
                              'superconductivity': 30,
                              'strongly correlated electrons': 40,
                              'soft condensed matter': 13,
                              'high energy physics - theory': 2,
                              'physics - physics and society': 1,
                              'disordered systems and neural networks': 1}),
                 (2013,
                  7): defaultdict(int,
                             {'soft condensed matter': 19,
                              'superconductivity': 31,
                              'strongly correlated electrons': 50,
                              'mesoscale and nanoscale physics': 53,
                              'materials science': 55,
                              'statistical mechanics': 49,
                              'quantum physics': 8,
                              'disordered systems and neural networks': 5,
                              'physics - physics and society': 2,
                              'quantum gases': 16,
                              'high energy physics - theory': 1}),
                 (2000,
                  6): defaultdict(int,
                             {'strongly correlated electrons': 12,
                              'condensed matter': 4,
                              'superconductivity': 12,
                              'statistical mechanics': 41,
                              'materials science': 13,
                              'mesoscale and nanoscale physics': 15,
                              'quantum physics': 2,
                              'soft condensed matter': 10,
                              'disordered systems and neural networks': 2}),
                 (2013,
                  10): defaultdict(int,
                             {'superconductivity': 26,
                              'soft condensed matter': 22,
                              'materials science': 54,
                              'mesoscale and nanoscale physics': 56,
                              'strongly correlated electrons': 35,
                              'statistical mechanics': 46,
                              'disordered systems and neural networks': 3,
                              'quantum gases': 10,
                              'quantum physics': 3,
                              'high energy physics - theory': 2}),
                 (1999,
                  4): defaultdict(int,
                             {'superconductivity': 13,
                              'statistical mechanics': 26,
                              'mesoscale and nanoscale physics': 13,
                              'soft condensed matter': 6,
                              'materials science': 5,
                              'strongly correlated electrons': 21,
                              'condensed matter': 2,
                              'disordered systems and neural networks': 4}),
                 (2001,
                  11): defaultdict(int,
                             {'strongly correlated electrons': 29,
                              'soft condensed matter': 6,
                              'statistical mechanics': 46,
                              'superconductivity': 21,
                              'materials science': 8,
                              'mesoscale and nanoscale physics': 13,
                              'other condensed matter': 2,
                              'quantum physics': 2,
                              'disordered systems and neural networks': 6,
                              'condensed matter': 3}),
                 (1998,
                  10): defaultdict(int,
                             {'mesoscale and nanoscale physics': 21,
                              'materials science': 3,
                              'statistical mechanics': 30,
                              'disordered systems and neural networks': 2,
                              'soft condensed matter': 8,
                              'superconductivity': 10,
                              'high energy physics - theory': 1,
                              'strongly correlated electrons': 14,
                              'condensed matter': 2,
                              'high energy physics - lattice': 1}),
                 (2009,
                  3): defaultdict(int,
                             {'quantum gases': 9,
                              'superconductivity': 31,
                              'statistical mechanics': 31,
                              'strongly correlated electrons': 33,
                              'mesoscale and nanoscale physics': 39,
                              'soft condensed matter': 16,
                              'disordered systems and neural networks': 6,
                              'quantum physics': 2,
                              'materials science': 28,
                              'high energy physics - theory': 1}),
                 (2003,
                  8): defaultdict(int,
                             {'soft condensed matter': 12,
                              'mesoscale and nanoscale physics': 24,
                              'statistical mechanics': 42,
                              'superconductivity': 16,
                              'strongly correlated electrons': 17,
                              'materials science': 16,
                              'other condensed matter': 4,
                              'quantum physics': 2,
                              'condensed matter': 5,
                              'disordered systems and neural networks': 2}),
                 (2003,
                  1): defaultdict(int,
                             {'statistical mechanics': 44,
                              'mesoscale and nanoscale physics': 30,
                              'materials science': 14,
                              'quantum physics': 3,
                              'strongly correlated electrons': 14,
                              'superconductivity': 11,
                              'soft condensed matter': 9,
                              'disordered systems and neural networks': 3,
                              'condensed matter': 2}),
                 (1995,
                  10): defaultdict(int,
                             {'condensed matter': 45, 'materials science': 1}),
                 (2011,
                  1): defaultdict(int,
                             {'mesoscale and nanoscale physics': 48,
                              'disordered systems and neural networks': 2,
                              'materials science': 44,
                              'strongly correlated electrons': 29,
                              'soft condensed matter': 18,
                              'quantum gases': 13,
                              'superconductivity': 25,
                              'statistical mechanics': 29,
                              'quantum physics': 2}),
                 (2013,
                  2): defaultdict(int,
                             {'statistical mechanics': 36,
                              'strongly correlated electrons': 23,
                              'materials science': 41,
                              'disordered systems and neural networks': 2,
                              'superconductivity': 29,
                              'quantum gases': 15,
                              'mesoscale and nanoscale physics': 56,
                              'quantum physics': 8,
                              'soft condensed matter': 21,
                              'high energy physics - theory': 1}),
                 (1995,
                  12): defaultdict(int,
                             {'condensed matter': 42, 'materials science': 1}),
                 (2008,
                  3): defaultdict(int,
                             {'materials science': 34,
                              'strongly correlated electrons': 29,
                              'soft condensed matter': 7,
                              'statistical mechanics': 37,
                              'superconductivity': 22,
                              'mesoscale and nanoscale physics': 39,
                              'quantum gases': 5,
                              'quantum physics': 5,
                              'other condensed matter': 5,
                              'disordered systems and neural networks': 2,
                              'high energy physics - theory': 1}),
                 (2004,
                  7): defaultdict(int,
                             {'statistical mechanics': 40,
                              'mesoscale and nanoscale physics': 38,
                              'superconductivity': 29,
                              'strongly correlated electrons': 28,
                              'disordered systems and neural networks': 4,
                              'materials science': 34,
                              'other condensed matter': 10,
                              'quantum physics': 3,
                              'soft condensed matter': 8}),
                 (2011,
                  11): defaultdict(int,
                             {'statistical mechanics': 40,
                              'soft condensed matter': 16,
                              'mesoscale and nanoscale physics': 53,
                              'quantum gases': 14,
                              'physics - chemical physics': 1,
                              'materials science': 46,
                              'strongly correlated electrons': 38,
                              'superconductivity': 23,
                              'disordered systems and neural networks': 2,
                              'quantum physics': 7,
                              'physics - optics': 1,
                              'physics - physics and society': 1}),
                 (2009,
                  9): defaultdict(int,
                             {'materials science': 33,
                              'soft condensed matter': 16,
                              'mesoscale and nanoscale physics': 39,
                              'superconductivity': 23,
                              'statistical mechanics': 28,
                              'quantum gases': 14,
                              'quantum physics': 2,
                              'strongly correlated electrons': 31,
                              'disordered systems and neural networks': 3,
                              'physics - physics and society': 1}),
                 (2007,
                  12): defaultdict(int,
                             {'disordered systems and neural networks': 4,
                              'mesoscale and nanoscale physics': 34,
                              'statistical mechanics': 31,
                              'strongly correlated electrons': 29,
                              'soft condensed matter': 9,
                              'superconductivity': 11,
                              'materials science': 23,
                              'quantum physics': 3,
                              'other condensed matter': 5}),
                 (1996,
                  5): defaultdict(int,
                             {'condensed matter': 52, 'superconductivity': 1}),
                 (2007,
                  1): defaultdict(int,
                             {'materials science': 22,
                              'quantum gases': 3,
                              'statistical mechanics': 39,
                              'mesoscale and nanoscale physics': 37,
                              'soft condensed matter': 11,
                              'superconductivity': 12,
                              'other condensed matter': 6,
                              'strongly correlated electrons': 24,
                              'quantum physics': 1}),
                 (2008,
                  10): defaultdict(int,
                             {'mesoscale and nanoscale physics': 47,
                              'statistical mechanics': 41,
                              'strongly correlated electrons': 25,
                              'materials science': 49,
                              'superconductivity': 38,
                              'soft condensed matter': 12,
                              'quantum physics': 2,
                              'other condensed matter': 3,
                              'disordered systems and neural networks': 4,
                              'quantum gases': 1}),
                 (2005,
                  8): defaultdict(int,
                             {'strongly correlated electrons': 26,
                              'superconductivity': 32,
                              'other condensed matter': 5,
                              'statistical mechanics': 36,
                              'materials science': 27,
                              'mesoscale and nanoscale physics': 34,
                              'soft condensed matter': 18,
                              'disordered systems and neural networks': 5,
                              'quantum physics': 1}),
                 (2000,
                  4): defaultdict(int,
                             {'soft condensed matter': 9,
                              'superconductivity': 13,
                              'statistical mechanics': 29,
                              'mesoscale and nanoscale physics': 18,
                              'materials science': 10,
                              'strongly correlated electrons': 25,
                              'condensed matter': 4,
                              'disordered systems and neural networks': 3,
                              'other condensed matter': 1}),
                 (1998,
                  3): defaultdict(int,
                             {'materials science': 6,
                              'statistical mechanics': 34,
                              'mesoscale and nanoscale physics': 11,
                              'strongly correlated electrons': 19,
                              'condensed matter': 2,
                              'soft condensed matter': 4,
                              'superconductivity': 6,
                              'quantum gases': 1,
                              'quantum physics': 1}),
                 (2002,
                  1): defaultdict(int,
                             {'disordered systems and neural networks': 4,
                              'materials science': 14,
                              'superconductivity': 22,
                              'statistical mechanics': 34,
                              'mesoscale and nanoscale physics': 17,
                              'soft condensed matter': 12,
                              'strongly correlated electrons': 21,
                              'quantum physics': 1}),
                 (2010,
                  12): defaultdict(int,
                             {'materials science': 46,
                              'statistical mechanics': 37,
                              'mesoscale and nanoscale physics': 54,
                              'soft condensed matter': 20,
                              'strongly correlated electrons': 37,
                              'quantum gases': 19,
                              'superconductivity': 21,
                              'quantum physics': 13,
                              'disordered systems and neural networks': 2,
                              'physics - physics and society': 1,
                              'high energy physics - theory': 1}),
                 (2004,
                  10): defaultdict(int,
                             {'statistical mechanics': 42,
                              'superconductivity': 15,
                              'mesoscale and nanoscale physics': 34,
                              'materials science': 25,
                              'strongly correlated electrons': 23,
                              'other condensed matter': 11,
                              'soft condensed matter': 7,
                              'quantum physics': 1}),
                 (2003,
                  9): defaultdict(int,
                             {'strongly correlated electrons': 22,
                              'statistical mechanics': 47,
                              'materials science': 20,
                              'superconductivity': 21,
                              'mesoscale and nanoscale physics': 24,
                              'disordered systems and neural networks': 2,
                              'soft condensed matter': 9,
                              'condensed matter': 4,
                              'quantum gases': 1,
                              'quantum physics': 1}),
                 (1999,
                  10): defaultdict(int,
                             {'statistical mechanics': 38,
                              'superconductivity': 14,
                              'condensed matter': 5,
                              'materials science': 9,
                              'mesoscale and nanoscale physics': 15,
                              'strongly correlated electrons': 26,
                              'disordered systems and neural networks': 1,
                              'soft condensed matter': 5}),
                 (2008,
                  12): defaultdict(int,
                             {'statistical mechanics': 37,
                              'strongly correlated electrons': 30,
                              'mesoscale and nanoscale physics': 24,
                              'materials science': 27,
                              'other condensed matter': 12,
                              'superconductivity': 29,
                              'soft condensed matter': 19,
                              'quantum physics': 2,
                              'disordered systems and neural networks': 4,
                              'quantum gases': 1,
                              'physics - physics and society': 1}),
                 (2012,
                  8): defaultdict(int,
                             {'materials science': 36,
                              'mesoscale and nanoscale physics': 54,
                              'superconductivity': 35,
                              'quantum gases': 9,
                              'statistical mechanics': 22,
                              'soft condensed matter': 17,
                              'strongly correlated electrons': 27,
                              'disordered systems and neural networks': 5,
                              'quantum physics': 5,
                              'high energy physics - theory': 1}),
                 (2012,
                  6): defaultdict(int,
                             {'strongly correlated electrons': 33,
                              'mesoscale and nanoscale physics': 62,
                              'statistical mechanics': 46,
                              'materials science': 48,
                              'superconductivity': 33,
                              'quantum gases': 17,
                              'disordered systems and neural networks': 5,
                              'soft condensed matter': 13,
                              'nuclear theory': 1,
                              'quantum physics': 7,
                              'high energy physics - theory': 1,
                              'other condensed matter': 1}),
                 (2007,
                  11): defaultdict(int,
                             {'mesoscale and nanoscale physics': 34,
                              'statistical mechanics': 39,
                              'strongly correlated electrons': 31,
                              'materials science': 27,
                              'quantum gases': 2,
                              'quantum physics': 8,
                              'soft condensed matter': 11,
                              'superconductivity': 15,
                              'high energy physics - theory': 1,
                              'other condensed matter': 4,
                              'disordered systems and neural networks': 4}),
                 (1992, 9): defaultdict(int, {'condensed matter': 10}),
                 (2002,
                  5): defaultdict(int,
                             {'strongly correlated electrons': 27,
                              'materials science': 16,
                              'soft condensed matter': 7,
                              'disordered systems and neural networks': 7,
                              'statistical mechanics': 35,
                              'superconductivity': 19,
                              'mesoscale and nanoscale physics': 16,
                              'quantum physics': 2,
                              'other condensed matter': 2,
                              'condensed matter': 1,
                              'quantum gases': 1}),
                 (1996,
                  1): defaultdict(int,
                             {'condensed matter': 25, 'materials science': 2}),
                 (1998,
                  5): defaultdict(int,
                             {'statistical mechanics': 29,
                              'disordered systems and neural networks': 2,
                              'condensed matter': 3,
                              'soft condensed matter': 6,
                              'other condensed matter': 2,
                              'materials science': 6,
                              'mesoscale and nanoscale physics': 9,
                              'strongly correlated electrons': 14,
                              'superconductivity': 12}),
                 (2005,
                  5): defaultdict(int,
                             {'superconductivity': 14,
                              'materials science': 28,
                              'statistical mechanics': 35,
                              'mesoscale and nanoscale physics': 32,
                              'soft condensed matter': 11,
                              'strongly correlated electrons': 22,
                              'quantum physics': 4,
                              'other condensed matter': 9,
                              'quantum gases': 2,
                              'disordered systems and neural networks': 4}),
                 (2011,
                  12): defaultdict(int,
                             {'materials science': 43,
                              'quantum physics': 8,
                              'mesoscale and nanoscale physics': 40,
                              'superconductivity': 24,
                              'statistical mechanics': 44,
                              'strongly correlated electrons': 26,
                              'quantum gases': 13,
                              'soft condensed matter': 13,
                              'mathematical physics': 2,
                              'disordered systems and neural networks': 3,
                              'general relativity and quantum cosmology': 1}),
                 (1995, 4): defaultdict(int, {'condensed matter': 26}),
                 (2002,
                  12): defaultdict(int,
                             {'soft condensed matter': 8,
                              'materials science': 19,
                              'mesoscale and nanoscale physics': 25,
                              'statistical mechanics': 49,
                              'superconductivity': 21,
                              'strongly correlated electrons': 23,
                              'other condensed matter': 3,
                              'disordered systems and neural networks': 2,
                              'condensed matter': 5}),
                 (2001,
                  8): defaultdict(int,
                             {'statistical mechanics': 32,
                              'strongly correlated electrons': 21,
                              'soft condensed matter': 9,
                              'superconductivity': 21,
                              'materials science': 16,
                              'mesoscale and nanoscale physics': 14,
                              'quantum physics': 2,
                              'mathematical physics': 1,
                              'high energy physics - theory': 1,
                              'other condensed matter': 1,
                              'condensed matter': 3,
                              'disordered systems and neural networks': 1}),
                 (2004,
                  3): defaultdict(int,
                             {'strongly correlated electrons': 26,
                              'mesoscale and nanoscale physics': 29,
                              'materials science': 25,
                              'statistical mechanics': 39,
                              'superconductivity': 13,
                              'other condensed matter': 5,
                              'soft condensed matter': 8,
                              'high energy physics - theory': 1,
                              'disordered systems and neural networks': 1,
                              'quantum gases': 2,
                              'quantum physics': 1}),
                 (1997,
                  11): defaultdict(int,
                             {'superconductivity': 8,
                              'strongly correlated electrons': 9,
                              'high energy physics - lattice': 1,
                              'mesoscale and nanoscale physics': 12,
                              'statistical mechanics': 22,
                              'soft condensed matter': 2,
                              'condensed matter': 2,
                              'materials science': 3,
                              'disordered systems and neural networks': 1}),
                 (2005,
                  7): defaultdict(int,
                             {'statistical mechanics': 43,
                              'materials science': 30,
                              'strongly correlated electrons': 32,
                              'soft condensed matter': 11,
                              'superconductivity': 21,
                              'mesoscale and nanoscale physics': 35,
                              'other condensed matter': 8,
                              'quantum physics': 5,
                              'disordered systems and neural networks': 5}),
                 (1999,
                  9): defaultdict(int,
                             {'statistical mechanics': 40,
                              'materials science': 11,
                              'superconductivity': 11,
                              'disordered systems and neural networks': 6,
                              'mesoscale and nanoscale physics': 8,
                              'strongly correlated electrons': 18,
                              'condensed matter': 1,
                              'soft condensed matter': 6,
                              'high energy physics - theory': 1}),
                 (2007,
                  8): defaultdict(int,
                             {'materials science': 32,
                              'mesoscale and nanoscale physics': 46,
                              'statistical mechanics': 39,
                              'soft condensed matter': 17,
                              'other condensed matter': 3,
                              'strongly correlated electrons': 27,
                              'superconductivity': 16,
                              'quantum physics': 2,
                              'disordered systems and neural networks': 2}),
                 (2000,
                  9): defaultdict(int,
                             {'statistical mechanics': 40,
                              'soft condensed matter': 6,
                              'strongly correlated electrons': 24,
                              'materials science': 8,
                              'superconductivity': 11,
                              'mesoscale and nanoscale physics': 18,
                              'disordered systems and neural networks': 2,
                              'condensed matter': 2,
                              'quantum physics': 2}),
                 (1994, 1): defaultdict(int, {'condensed matter': 18}),
                 (2003,
                  4): defaultdict(int,
                             {'mesoscale and nanoscale physics': 24,
                              'strongly correlated electrons': 36,
                              'statistical mechanics': 33,
                              'superconductivity': 28,
                              'materials science': 23,
                              'soft condensed matter': 8,
                              'disordered systems and neural networks': 2,
                              'quantum physics': 1,
                              'other condensed matter': 1}),
                 (2003,
                  3): defaultdict(int,
                             {'mesoscale and nanoscale physics': 23,
                              'statistical mechanics': 47,
                              'superconductivity': 17,
                              'strongly correlated electrons': 18,
                              'materials science': 16,
                              'disordered systems and neural networks': 2,
                              'soft condensed matter': 6,
                              'other condensed matter': 1,
                              'quantum physics': 1,
                              'condensed matter': 2,
                              'quantum gases': 1}),
                 (2003,
                  11): defaultdict(int,
                             {'soft condensed matter': 7,
                              'strongly correlated electrons': 22,
                              'disordered systems and neural networks': 1,
                              'statistical mechanics': 39,
                              'superconductivity': 15,
                              'mesoscale and nanoscale physics': 29,
                              'condensed matter': 1,
                              'materials science': 16,
                              'quantum gases': 1,
                              'other condensed matter': 3,
                              'quantum physics': 3}),
                 (2005,
                  10): defaultdict(int,
                             {'strongly correlated electrons': 37,
                              'statistical mechanics': 38,
                              'soft condensed matter': 14,
                              'materials science': 25,
                              'mesoscale and nanoscale physics': 22,
                              'superconductivity': 29,
                              'other condensed matter': 7,
                              'disordered systems and neural networks': 2,
                              'high energy physics - theory': 2,
                              'quantum physics': 2,
                              'physics - chemical physics': 1}),
                 (1995, 3): defaultdict(int, {'condensed matter': 43}),
                 (2012,
                  7): defaultdict(int,
                             {'materials science': 41,
                              'soft condensed matter': 16,
                              'superconductivity': 34,
                              'statistical mechanics': 42,
                              'strongly correlated electrons': 37,
                              'mesoscale and nanoscale physics': 60,
                              'quantum physics': 7,
                              'quantum gases': 25,
                              'other condensed matter': 1,
                              'disordered systems and neural networks': 3,
                              'high energy physics - theory': 3}),
                 (2009,
                  7): defaultdict(int,
                             {'materials science': 32,
                              'quantum gases': 10,
                              'mesoscale and nanoscale physics': 41,
                              'strongly correlated electrons': 35,
                              'statistical mechanics': 51,
                              'superconductivity': 23,
                              'disordered systems and neural networks': 2,
                              'soft condensed matter': 17,
                              'physics - physics and society': 1,
                              'quantum physics': 3}),
                 (2001,
                  12): defaultdict(int,
                             {'mesoscale and nanoscale physics': 17,
                              'other condensed matter': 1,
                              'strongly correlated electrons': 22,
                              'materials science': 12,
                              'superconductivity': 18,
                              'statistical mechanics': 37,
                              'quantum physics': 4,
                              'disordered systems and neural networks': 6,
                              'condensed matter': 2,
                              'soft condensed matter': 13,
                              'high energy physics - theory': 3}),
                 (1997,
                  12): defaultdict(int,
                             {'statistical mechanics': 25,
                              'disordered systems and neural networks': 4,
                              'mesoscale and nanoscale physics': 19,
                              'strongly correlated electrons': 11,
                              'superconductivity': 7,
                              'soft condensed matter': 3,
                              'materials science': 3,
                              'condensed matter': 1}),
                 (1998,
                  1): defaultdict(int,
                             {'materials science': 3,
                              'strongly correlated electrons': 12,
                              'superconductivity': 9,
                              'statistical mechanics': 21,
                              'soft condensed matter': 3,
                              'disordered systems and neural networks': 1,
                              'mesoscale and nanoscale physics': 11,
                              'condensed matter': 2}),
                 (1992, 7): defaultdict(int, {'condensed matter': 7}),
                 (2013,
                  8): defaultdict(int,
                             {'strongly correlated electrons': 34,
                              'mesoscale and nanoscale physics': 58,
                              'materials science': 46,
                              'superconductivity': 29,
                              'quantum gases': 12,
                              'statistical mechanics': 33,
                              'quantum physics': 4,
                              'high energy physics - theory': 3,
                              'soft condensed matter': 13,
                              'physics - physics and society': 1}),
                 (2009,
                  12): defaultdict(int,
                             {'quantum gases': 16,
                              'mesoscale and nanoscale physics': 47,
                              'strongly correlated electrons': 33,
                              'materials science': 29,
                              'statistical mechanics': 32,
                              'superconductivity': 26,
                              'soft condensed matter': 15,
                              'disordered systems and neural networks': 2,
                              'other condensed matter': 1,
                              'quantum physics': 1}),
                 (1997,
                  4): defaultdict(int,
                             {'statistical mechanics': 22,
                              'disordered systems and neural networks': 3,
                              'soft condensed matter': 2,
                              'condensed matter': 3,
                              'strongly correlated electrons': 7,
                              'superconductivity': 7,
                              'mesoscale and nanoscale physics': 10,
                              'materials science': 2}),
                 (2008,
                  8): defaultdict(int,
                             {'materials science': 21,
                              'quantum gases': 4,
                              'superconductivity': 31,
                              'soft condensed matter': 13,
                              'strongly correlated electrons': 39,
                              'statistical mechanics': 32,
                              'disordered systems and neural networks': 3,
                              'mesoscale and nanoscale physics': 34,
                              'other condensed matter': 5,
                              'high energy physics - theory': 1}),
                 (2014,
                  4): defaultdict(int,
                             {'materials science': 51,
                              'superconductivity': 15,
                              'soft condensed matter': 20,
                              'mesoscale and nanoscale physics': 46,
                              'quantum physics': 4,
                              'statistical mechanics': 32,
                              'strongly correlated electrons': 35,
                              'quantum gases': 12,
                              'disordered systems and neural networks': 4,
                              'high energy physics - theory': 2}),
                 (2011,
                  5): defaultdict(int,
                             {'mesoscale and nanoscale physics': 67,
                              'statistical mechanics': 42,
                              'superconductivity': 34,
                              'soft condensed matter': 17,
                              'quantum gases': 14,
                              'disordered systems and neural networks': 3,
                              'materials science': 39,
                              'strongly correlated electrons': 25,
                              'mathematical physics': 1,
                              'other condensed matter': 1,
                              'quantum physics': 1}),
                 (2004,
                  9): defaultdict(int,
                             {'mesoscale and nanoscale physics': 25,
                              'superconductivity': 13,
                              'statistical mechanics': 36,
                              'strongly correlated electrons': 24,
                              'materials science': 27,
                              'soft condensed matter': 12,
                              'other condensed matter': 5,
                              'quantum physics': 1,
                              'disordered systems and neural networks': 2}),
                 (1999,
                  7): defaultdict(int,
                             {'mesoscale and nanoscale physics': 18,
                              'statistical mechanics': 27,
                              'strongly correlated electrons': 14,
                              'disordered systems and neural networks': 3,
                              'soft condensed matter': 4,
                              'superconductivity': 14,
                              'materials science': 8,
                              'other condensed matter': 1,
                              'condensed matter': 2,
                              'quantum physics': 1}),
                 (2012,
                  4): defaultdict(int,
                             {'soft condensed matter': 23,
                              'superconductivity': 24,
                              'materials science': 45,
                              'statistical mechanics': 36,
                              'mesoscale and nanoscale physics': 63,
                              'quantum gases': 23,
                              'strongly correlated electrons': 41,
                              'quantum physics': 5,
                              'disordered systems and neural networks': 4,
                              'high energy physics - theory': 1,
                              'other condensed matter': 1}),
                 (1993, 11): defaultdict(int, {'condensed matter': 11}),
                 (2001,
                  10): defaultdict(int,
                             {'statistical mechanics': 52,
                              'materials science': 6,
                              'strongly correlated electrons': 20,
                              'mesoscale and nanoscale physics': 24,
                              'superconductivity': 25,
                              'soft condensed matter': 5,
                              'condensed matter': 6,
                              'disordered systems and neural networks': 4,
                              'quantum physics': 1}),
                 (2008,
                  6): defaultdict(int,
                             {'superconductivity': 30,
                              'strongly correlated electrons': 35,
                              'materials science': 23,
                              'mesoscale and nanoscale physics': 44,
                              'quantum physics': 2,
                              'statistical mechanics': 25,
                              'quantum gases': 4,
                              'soft condensed matter': 14,
                              'disordered systems and neural networks': 2,
                              'other condensed matter': 5,
                              'high energy physics - theory': 2}),
                 (2004,
                  12): defaultdict(int,
                             {'strongly correlated electrons': 26,
                              'mesoscale and nanoscale physics': 35,
                              'statistical mechanics': 35,
                              'materials science': 27,
                              'quantum physics': 4,
                              'soft condensed matter': 12,
                              'superconductivity': 24,
                              'quantum gases': 1,
                              'disordered systems and neural networks': 3,
                              'other condensed matter': 4}),
                 (1993, 10): defaultdict(int, {'condensed matter': 20}),
                 (1996, 4): defaultdict(int, {'condensed matter': 45}),
                 (2002,
                  11): defaultdict(int,
                             {'mesoscale and nanoscale physics': 20,
                              'strongly correlated electrons': 21,
                              'soft condensed matter': 23,
                              'statistical mechanics': 38,
                              'superconductivity': 20,
                              'quantum physics': 3,
                              'materials science': 16,
                              'condensed matter': 4,
                              'disordered systems and neural networks': 2,
                              'quantum gases': 1,
                              'other condensed matter': 1}),
                 (2014,
                  3): defaultdict(int,
                             {'mesoscale and nanoscale physics': 50,
                              'soft condensed matter': 24,
                              'disordered systems and neural networks': 6,
                              'superconductivity': 25,
                              'statistical mechanics': 40,
                              'high energy physics - theory': 4,
                              'materials science': 59,
                              'quantum physics': 5,
                              'quantum gases': 16,
                              'strongly correlated electrons': 37}),
                 (2010,
                  4): defaultdict(int,
                             {'materials science': 23,
                              'superconductivity': 29,
                              'mesoscale and nanoscale physics': 48,
                              'quantum gases': 10,
                              'statistical mechanics': 39,
                              'strongly correlated electrons': 29,
                              'soft condensed matter': 18,
                              'quantum physics': 1,
                              'high energy physics - theory': 1,
                              'disordered systems and neural networks': 3}),
                 (1993,
                  6): defaultdict(int,
                             {'condensed matter': 18, 'materials science': 1}),
                 (1997,
                  10): defaultdict(int,
                             {'superconductivity': 10,
                              'statistical mechanics': 31,
                              'mesoscale and nanoscale physics': 13,
                              'disordered systems and neural networks': 2,
                              'strongly correlated electrons': 17,
                              'condensed matter': 3,
                              'materials science': 5,
                              'soft condensed matter': 1}),
                 (2000,
                  7): defaultdict(int,
                             {'strongly correlated electrons': 32,
                              'statistical mechanics': 39,
                              'mesoscale and nanoscale physics': 11,
                              'soft condensed matter': 9,
                              'materials science': 5,
                              'disordered systems and neural networks': 4,
                              'superconductivity': 6,
                              'condensed matter': 4}),
                 (1995,
                  8): defaultdict(int,
                             {'condensed matter': 33, 'materials science': 2}),
                 (1995,
                  2): defaultdict(int,
                             {'condensed matter': 27,
                              'high energy physics - theory': 1,
                              'materials science': 2}),
                 (2004,
                  4): defaultdict(int,
                             {'statistical mechanics': 52,
                              'superconductivity': 15,
                              'mesoscale and nanoscale physics': 27,
                              'strongly correlated electrons': 35,
                              'materials science': 22,
                              'other condensed matter': 3,
                              'quantum gases': 4,
                              'disordered systems and neural networks': 2,
                              'soft condensed matter': 7,
                              'quantum physics': 1}),
                 (2007,
                  5): defaultdict(int,
                             {'superconductivity': 20,
                              'statistical mechanics': 39,
                              'other condensed matter': 9,
                              'materials science': 27,
                              'mesoscale and nanoscale physics': 32,
                              'quantum physics': 4,
                              'strongly correlated electrons': 21,
                              'soft condensed matter': 19,
                              'quantum gases': 1,
                              'disordered systems and neural networks': 2}),
                 (2003,
                  2): defaultdict(int,
                             {'statistical mechanics': 34,
                              'superconductivity': 30,
                              'mesoscale and nanoscale physics': 26,
                              'condensed matter': 4,
                              'soft condensed matter': 15,
                              'materials science': 11,
                              'other condensed matter': 1,
                              'strongly correlated electrons': 22,
                              'disordered systems and neural networks': 4,
                              'quantum physics': 1}),
                 (2001,
                  4): defaultdict(int,
                             {'materials science': 7,
                              'statistical mechanics': 30,
                              'superconductivity': 30,
                              'mesoscale and nanoscale physics': 17,
                              'strongly correlated electrons': 12,
                              'soft condensed matter': 9,
                              'quantum physics': 1,
                              'condensed matter': 1}),
                 (1999,
                  3): defaultdict(int,
                             {'statistical mechanics': 32,
                              'strongly correlated electrons': 18,
                              'materials science': 8,
                              'high energy physics - theory': 1,
                              'soft condensed matter': 8,
                              'superconductivity': 18,
                              'mesoscale and nanoscale physics': 14,
                              'condensed matter': 2}),
                 (2012,
                  9): defaultdict(int,
                             {'mesoscale and nanoscale physics': 41,
                              'other condensed matter': 1,
                              'statistical mechanics': 48,
                              'strongly correlated electrons': 20,
                              'materials science': 44,
                              'superconductivity': 24,
                              'quantum physics': 7,
                              'soft condensed matter': 20,
                              'mathematical physics': 3,
                              'quantum gases': 13,
                              'disordered systems and neural networks': 1}),
                 (2000,
                  12): defaultdict(int,
                             {'statistical mechanics': 33,
                              'strongly correlated electrons': 17,
                              'mesoscale and nanoscale physics': 15,
                              'soft condensed matter': 11,
                              'superconductivity': 15,
                              'disordered systems and neural networks': 1,
                              'high energy physics - lattice': 1,
                              'condensed matter': 2,
                              'quantum physics': 2,
                              'materials science': 4}),
                 (2005,
                  6): defaultdict(int,
                             {'materials science': 27,
                              'other condensed matter': 11,
                              'soft condensed matter': 18,
                              'superconductivity': 17,
                              'statistical mechanics': 27,
                              'strongly correlated electrons': 32,
                              'mesoscale and nanoscale physics': 31,
                              'disordered systems and neural networks': 3,
                              'quantum physics': 8,
                              'quantum gases': 2}),
                 (2002,
                  4): defaultdict(int,
                             {'soft condensed matter': 8,
                              'strongly correlated electrons': 15,
                              'statistical mechanics': 38,
                              'superconductivity': 18,
                              'condensed matter': 2,
                              'materials science': 14,
                              'mesoscale and nanoscale physics': 13,
                              'disordered systems and neural networks': 3,
                              'high energy physics - phenomenology': 1,
                              'high energy physics - theory': 1}),
                 (2002,
                  6): defaultdict(int,
                             {'statistical mechanics': 37,
                              'mesoscale and nanoscale physics': 25,
                              'strongly correlated electrons': 24,
                              'soft condensed matter': 8,
                              'materials science': 11,
                              'disordered systems and neural networks': 5,
                              'quantum physics': 3,
                              'superconductivity': 10,
                              'condensed matter': 2}),
                 (2000,
                  11): defaultdict(int,
                             {'statistical mechanics': 42,
                              'strongly correlated electrons': 14,
                              'superconductivity': 16,
                              'quantum gases': 3,
                              'mesoscale and nanoscale physics': 17,
                              'disordered systems and neural networks': 2,
                              'soft condensed matter': 3,
                              'condensed matter': 2,
                              'materials science': 8}),
                 (2010,
                  2): defaultdict(int,
                             {'statistical mechanics': 26,
                              'materials science': 42,
                              'strongly correlated electrons': 24,
                              'superconductivity': 27,
                              'soft condensed matter': 16,
                              'mesoscale and nanoscale physics': 52,
                              'quantum gases': 9,
                              'quantum physics': 5,
                              'disordered systems and neural networks': 2}),
                 (1999,
                  5): defaultdict(int,
                             {'statistical mechanics': 33,
                              'mesoscale and nanoscale physics': 18,
                              'superconductivity': 15,
                              'strongly correlated electrons': 14,
                              'condensed matter': 2,
                              'materials science': 5,
                              'soft condensed matter': 5,
                              'disordered systems and neural networks': 1}),
                 (2001,
                  3): defaultdict(int,
                             {'superconductivity': 30,
                              'statistical mechanics': 39,
                              'soft condensed matter': 8,
                              'strongly correlated electrons': 20,
                              'materials science': 8,
                              'high energy physics - theory': 2,
                              'mesoscale and nanoscale physics': 15,
                              'condensed matter': 3,
                              'disordered systems and neural networks': 5,
                              'quantum physics': 1,
                              'other condensed matter': 1}),
                 (2001,
                  1): defaultdict(int,
                             {'condensed matter': 4,
                              'mesoscale and nanoscale physics': 13,
                              'disordered systems and neural networks': 3,
                              'statistical mechanics': 31,
                              'soft condensed matter': 11,
                              'materials science': 7,
                              'strongly correlated electrons': 22,
                              'superconductivity': 9,
                              'high energy physics - theory': 1}),
                 (2009,
                  8): defaultdict(int,
                             {'mesoscale and nanoscale physics': 44,
                              'materials science': 23,
                              'strongly correlated electrons': 22,
                              'quantum gases': 9,
                              'statistical mechanics': 23,
                              'superconductivity': 19,
                              'physics - physics and society': 1,
                              'soft condensed matter': 9,
                              'quantum physics': 1}),
                 (2006,
                  4): defaultdict(int,
                             {'statistical mechanics': 32,
                              'superconductivity': 18,
                              'mesoscale and nanoscale physics': 26,
                              'materials science': 21,
                              'soft condensed matter': 12,
                              'strongly correlated electrons': 24,
                              'other condensed matter': 7,
                              'quantum gases': 1,
                              'disordered systems and neural networks': 5}),
                 (2001,
                  2): defaultdict(int,
                             {'strongly correlated electrons': 26,
                              'statistical mechanics': 31,
                              'superconductivity': 21,
                              'mesoscale and nanoscale physics': 12,
                              'materials science': 8,
                              'soft condensed matter': 6,
                              'disordered systems and neural networks': 2,
                              'quantum physics': 1,
                              'condensed matter': 2}),
                 (2009,
                  6): defaultdict(int,
                             {'mesoscale and nanoscale physics': 37,
                              'statistical mechanics': 28,
                              'materials science': 28,
                              'strongly correlated electrons': 33,
                              'soft condensed matter': 16,
                              'quantum gases': 16,
                              'superconductivity': 14,
                              'quantum physics': 4,
                              'nonlinear sciences - chaotic dynamics': 1,
                              'disordered systems and neural networks': 1}),
                 (2002,
                  10): defaultdict(int,
                             {'disordered systems and neural networks': 6,
                              'statistical mechanics': 43,
                              'soft condensed matter': 9,
                              'mesoscale and nanoscale physics': 17,
                              'other condensed matter': 2,
                              'materials science': 21,
                              'strongly correlated electrons': 23,
                              'superconductivity': 23,
                              'condensed matter': 3,
                              'quantum physics': 1}),
                 (1993,
                  12): defaultdict(int,
                             {'condensed matter': 28, 'materials science': 1}),
                 (2011,
                  2): defaultdict(int,
                             {'statistical mechanics': 38,
                              'materials science': 47,
                              'soft condensed matter': 19,
                              'mesoscale and nanoscale physics': 42,
                              'superconductivity': 30,
                              'quantum physics': 5,
                              'quantum gases': 10,
                              'strongly correlated electrons': 24,
                              'disordered systems and neural networks': 5,
                              'mathematical physics': 1,
                              'high energy physics - theory': 1}),
                 (2009,
                  5): defaultdict(int,
                             {'statistical mechanics': 33,
                              'strongly correlated electrons': 28,
                              'materials science': 25,
                              'mesoscale and nanoscale physics': 35,
                              'soft condensed matter': 14,
                              'superconductivity': 24,
                              'quantum gases': 6,
                              'quantum physics': 5,
                              'disordered systems and neural networks': 4,
                              'high energy physics - theory': 1,
                              'physics - physics and society': 1,
                              'other condensed matter': 1}),
                 (2005,
                  3): defaultdict(int,
                             {'mesoscale and nanoscale physics': 36,
                              'superconductivity': 26,
                              'statistical mechanics': 48,
                              'soft condensed matter': 11,
                              'materials science': 31,
                              'nonlinear sciences - chaotic dynamics': 1,
                              'strongly correlated electrons': 24,
                              'disordered systems and neural networks': 2,
                              'other condensed matter': 4,
                              'quantum gases': 1}),
                 (2010,
                  1): defaultdict(int,
                             {'materials science': 34,
                              'statistical mechanics': 37,
                              'mesoscale and nanoscale physics': 46,
                              'strongly correlated electrons': 23,
                              'soft condensed matter': 12,
                              'superconductivity': 34,
                              'quantum gases': 13,
                              'quantum physics': 6,
                              'disordered systems and neural networks': 2}),
                 (2002,
                  2): defaultdict(int,
                             {'strongly correlated electrons': 17,
                              'superconductivity': 17,
                              'statistical mechanics': 35,
                              'mesoscale and nanoscale physics': 18,
                              'soft condensed matter': 10,
                              'materials science': 10,
                              'quantum physics': 1,
                              'condensed matter': 1,
                              'quantum gases': 1,
                              'disordered systems and neural networks': 1,
                              'other condensed matter': 1,
                              'high energy physics - theory': 1}),
                 (2004,
                  6): defaultdict(int,
                             {'superconductivity': 21,
                              'mesoscale and nanoscale physics': 39,
                              'strongly correlated electrons': 36,
                              'soft condensed matter': 8,
                              'statistical mechanics': 28,
                              'other condensed matter': 10,
                              'materials science': 22,
                              'quantum physics': 2,
                              'disordered systems and neural networks': 4,
                              'quantum gases': 1}),
                 (1995,
                  7): defaultdict(int,
                             {'condensed matter': 31, 'materials science': 1}),
                 (2009,
                  1): defaultdict(int,
                             {'strongly correlated electrons': 18,
                              'materials science': 31,
                              'superconductivity': 19,
                              'statistical mechanics': 18,
                              'mesoscale and nanoscale physics': 27,
                              'quantum gases': 6,
                              'disordered systems and neural networks': 3,
                              'soft condensed matter': 13,
                              'quantum physics': 2,
                              'other condensed matter': 1}),
                 (2007,
                  6): defaultdict(int,
                             {'strongly correlated electrons': 31,
                              'statistical mechanics': 29,
                              'mesoscale and nanoscale physics': 33,
                              'superconductivity': 20,
                              'materials science': 30,
                              'other condensed matter': 15,
                              'soft condensed matter': 14,
                              'disordered systems and neural networks': 5,
                              'quantum physics': 2,
                              'quantum gases': 1}),
                 (2003,
                  10): defaultdict(int,
                             {'materials science': 19,
                              'strongly correlated electrons': 28,
                              'statistical mechanics': 43,
                              'soft condensed matter': 12,
                              'mesoscale and nanoscale physics': 37,
                              'high energy physics - theory': 1,
                              'superconductivity': 18,
                              'disordered systems and neural networks': 6,
                              'other condensed matter': 4,
                              'quantum gases': 1}),
                 (1995, 11): defaultdict(int, {'condensed matter': 30}),
                 (1996, 2): defaultdict(int, {'condensed matter': 39}),
                 (1994,
                  6): defaultdict(int,
                             {'condensed matter': 25, 'materials science': 1}),
                 (2008,
                  2): defaultdict(int,
                             {'superconductivity': 15,
                              'other condensed matter': 1,
                              'strongly correlated electrons': 20,
                              'soft condensed matter': 12,
                              'statistical mechanics': 35,
                              'mesoscale and nanoscale physics': 29,
                              'quantum gases': 2,
                              'materials science': 14,
                              'disordered systems and neural networks': 2,
                              'quantum physics': 4}),
                 (1999,
                  12): defaultdict(int,
                             {'statistical mechanics': 33,
                              'mesoscale and nanoscale physics': 16,
                              'superconductivity': 17,
                              'strongly correlated electrons': 25,
                              'high energy physics - lattice': 1,
                              'high energy physics - theory': 2,
                              'materials science': 6,
                              'soft condensed matter': 2,
                              'quantum gases': 1,
                              'disordered systems and neural networks': 3,
                              'condensed matter': 1}),
                 (2014,
                  1): defaultdict(int,
                             {'superconductivity': 15,
                              'statistical mechanics': 33,
                              'mesoscale and nanoscale physics': 46,
                              'materials science': 68,
                              'soft condensed matter': 21,
                              'quantum physics': 9,
                              'strongly correlated electrons': 29,
                              'quantum gases': 7,
                              'other condensed matter': 1}),
                 (2006,
                  12): defaultdict(int,
                             {'mesoscale and nanoscale physics': 32,
                              'other condensed matter': 8,
                              'materials science': 31,
                              'superconductivity': 15,
                              'strongly correlated electrons': 19,
                              'statistical mechanics': 35,
                              'soft condensed matter': 9,
                              'quantum physics': 4,
                              'disordered systems and neural networks': 1,
                              'quantum gases': 1}),
                 (1996,
                  11): defaultdict(int,
                             {'condensed matter': 53,
                              'superconductivity': 1,
                              'strongly correlated electrons': 2,
                              'materials science': 2,
                              'mesoscale and nanoscale physics': 2}),
                 (2007,
                  2): defaultdict(int,
                             {'materials science': 28,
                              'mesoscale and nanoscale physics': 29,
                              'superconductivity': 19,
                              'strongly correlated electrons': 18,
                              'statistical mechanics': 20,
                              'quantum physics': 1,
                              'soft condensed matter': 11,
                              'other condensed matter': 7,
                              'disordered systems and neural networks': 3,
                              'quantum gases': 1,
                              'physics - biological physics': 1,
                              'high energy physics - theory': 1}),
                 (1992, 12): defaultdict(int, {'condensed matter': 11}),
                 (1997,
                  8): defaultdict(int,
                             {'superconductivity': 8,
                              'statistical mechanics': 16,
                              'disordered systems and neural networks': 3,
                              'mesoscale and nanoscale physics': 8,
                              'strongly correlated electrons': 6,
                              'condensed matter': 2,
                              'materials science': 2,
                              'high energy physics - theory': 1,
                              'soft condensed matter': 2}),
                 (1994, 3): defaultdict(int, {'condensed matter': 24}),
                 (2007,
                  7): defaultdict(int,
                             {'other condensed matter': 8,
                              'mesoscale and nanoscale physics': 39,
                              'statistical mechanics': 30,
                              'materials science': 32,
                              'strongly correlated electrons': 27,
                              'disordered systems and neural networks': 3,
                              'superconductivity': 17,
                              'quantum gases': 3,
                              'soft condensed matter': 14,
                              'quantum physics': 1,
                              'high energy physics - theory': 1}),
                 (2005,
                  1): defaultdict(int,
                             {'statistical mechanics': 47,
                              'strongly correlated electrons': 22,
                              'superconductivity': 16,
                              'mesoscale and nanoscale physics': 27,
                              'materials science': 22,
                              'soft condensed matter': 10,
                              'other condensed matter': 3,
                              'quantum physics': 1,
                              'disordered systems and neural networks': 3,
                              'high energy physics - theory': 1}),
                 (1993, 1): defaultdict(int, {'condensed matter': 14}),
                 (1996,
                  12): defaultdict(int,
                             {'condensed matter': 43,
                              'materials science': 1,
                              'mesoscale and nanoscale physics': 2}),
                 (1998,
                  6): defaultdict(int,
                             {'strongly correlated electrons': 11,
                              'soft condensed matter': 5,
                              'statistical mechanics': 31,
                              'mesoscale and nanoscale physics': 10,
                              'materials science': 8,
                              'superconductivity': 11,
                              'disordered systems and neural networks': 2,
                              'high energy physics - lattice': 1,
                              'condensed matter': 1}),
                 (2006,
                  11): defaultdict(int,
                             {'strongly correlated electrons': 34,
                              'superconductivity': 18,
                              'materials science': 30,
                              'statistical mechanics': 37,
                              'mesoscale and nanoscale physics': 31,
                              'soft condensed matter': 9,
                              'disordered systems and neural networks': 6,
                              'other condensed matter': 4,
                              'quantum physics': 1}),
                 (2006,
                  1): defaultdict(int,
                             {'materials science': 20,
                              'superconductivity': 16,
                              'strongly correlated electrons': 27,
                              'statistical mechanics': 29,
                              'mesoscale and nanoscale physics': 32,
                              'other condensed matter': 6,
                              'soft condensed matter': 9,
                              'disordered systems and neural networks': 1,
                              'high energy physics - theory': 1}),
                 (1994,
                  12): defaultdict(int,
                             {'condensed matter': 28, 'materials science': 1}),
                 (2006,
                  2): defaultdict(int,
                             {'statistical mechanics': 36,
                              'mesoscale and nanoscale physics': 27,
                              'materials science': 26,
                              'strongly correlated electrons': 22,
                              'superconductivity': 17,
                              'quantum gases': 1,
                              'disordered systems and neural networks': 3,
                              'soft condensed matter': 5,
                              'other condensed matter': 3,
                              'high energy physics - theory': 1}),
                 (1998,
                  2): defaultdict(int,
                             {'strongly correlated electrons': 18,
                              'superconductivity': 7,
                              'statistical mechanics': 21,
                              'mesoscale and nanoscale physics': 8,
                              'disordered systems and neural networks': 3,
                              'soft condensed matter': 1,
                              'condensed matter': 2,
                              'materials science': 2,
                              'high energy physics - theory': 1}),
                 (1997,
                  5): defaultdict(int,
                             {'mesoscale and nanoscale physics': 10,
                              'statistical mechanics': 24,
                              'strongly correlated electrons': 15,
                              'superconductivity': 9,
                              'condensed matter': 2,
                              'materials science': 6,
                              'high energy physics - theory': 1,
                              'other condensed matter': 1,
                              'disordered systems and neural networks': 1}),
                 (1997,
                  7): defaultdict(int,
                             {'mesoscale and nanoscale physics': 15,
                              'superconductivity': 6,
                              'disordered systems and neural networks': 4,
                              'soft condensed matter': 4,
                              'condensed matter': 2,
                              'statistical mechanics': 23,
                              'materials science': 5,
                              'strongly correlated electrons': 12}),
                 (2005,
                  2): defaultdict(int,
                             {'superconductivity': 19,
                              'strongly correlated electrons': 24,
                              'disordered systems and neural networks': 3,
                              'statistical mechanics': 28,
                              'mesoscale and nanoscale physics': 32,
                              'other condensed matter': 9,
                              'materials science': 18,
                              'soft condensed matter': 8,
                              'quantum physics': 3,
                              'quantum gases': 1}),
                 (1997,
                  2): defaultdict(int,
                             {'strongly correlated electrons': 18,
                              'condensed matter': 1,
                              'statistical mechanics': 6,
                              'superconductivity': 12,
                              'mesoscale and nanoscale physics': 10,
                              'materials science': 4,
                              'soft condensed matter': 1}),
                 (1996,
                  6): defaultdict(int,
                             {'condensed matter': 50, 'materials science': 1}),
                 (1996,
                  10): defaultdict(int,
                             {'condensed matter': 53,
                              'superconductivity': 3,
                              'high energy physics - theory': 1,
                              'materials science': 3}),
                 (1996,
                  7): defaultdict(int,
                             {'condensed matter': 38,
                              'materials science': 3,
                              'mesoscale and nanoscale physics': 1}),
                 (1992, 11): defaultdict(int, {'condensed matter': 3}),
                 (1998,
                  11): defaultdict(int,
                             {'mesoscale and nanoscale physics': 15,
                              'statistical mechanics': 25,
                              'materials science': 6,
                              'strongly correlated electrons': 16,
                              'soft condensed matter': 2,
                              'condensed matter': 3,
                              'superconductivity': 14,
                              'high energy physics - theory': 2,
                              'disordered systems and neural networks': 2}),
                 (2000,
                  3): defaultdict(int,
                             {'statistical mechanics': 32,
                              'soft condensed matter': 6,
                              'condensed matter': 2,
                              'superconductivity': 20,
                              'mesoscale and nanoscale physics': 20,
                              'strongly correlated electrons': 17,
                              'materials science': 7,
                              'disordered systems and neural networks': 3,
                              'other condensed matter': 1}),
                 (1995, 5): defaultdict(int, {'condensed matter': 30}),
                 (1997,
                  1): defaultdict(int,
                             {'strongly correlated electrons': 11,
                              'disordered systems and neural networks': 5,
                              'mesoscale and nanoscale physics': 13,
                              'materials science': 2,
                              'superconductivity': 3,
                              'statistical mechanics': 13,
                              'soft condensed matter': 1,
                              'condensed matter': 2}),
                 (1992, 4): defaultdict(int, {'condensed matter': 3}),
                 (1994,
                  9): defaultdict(int,
                             {'condensed matter': 27, 'materials science': 1}),
                 (1994, 2): defaultdict(int, {'condensed matter': 28}),
                 (1996,
                  8): defaultdict(int,
                             {'condensed matter': 41,
                              'materials science': 1,
                              'mesoscale and nanoscale physics': 1}),
                 (1995, 6): defaultdict(int, {'condensed matter': 30}),
                 (1994, 11): defaultdict(int, {'condensed matter': 33}),
                 (1995,
                  1): defaultdict(int,
                             {'condensed matter': 37, 'materials science': 1}),
                 (1997,
                  9): defaultdict(int,
                             {'statistical mechanics': 21,
                              'strongly correlated electrons': 20,
                              'mesoscale and nanoscale physics': 10,
                              'condensed matter': 3,
                              'materials science': 4,
                              'soft condensed matter': 5,
                              'superconductivity': 6,
                              'disordered systems and neural networks': 1}),
                 (1994, 5): defaultdict(int, {'condensed matter': 22}),
                 (1992, 8): defaultdict(int, {'condensed matter': 9}),
                 (1994, 4): defaultdict(int, {'condensed matter': 25}),
                 (1994, 10): defaultdict(int, {'condensed matter': 31}),
                 (1994, 8): defaultdict(int, {'condensed matter': 33}),
                 (1993, 2): defaultdict(int, {'condensed matter': 7}),
                 (1996, 3): defaultdict(int, {'condensed matter': 51}),
                 (1994,
                  7): defaultdict(int,
                             {'condensed matter': 31,
                              'high energy physics - lattice': 1}),
                 (1992, 10): defaultdict(int, {'condensed matter': 6}),
                 (1993, 3): defaultdict(int, {'condensed matter': 13}),
                 (1993, 9): defaultdict(int, {'condensed matter': 13}),
                 (1993, 4): defaultdict(int, {'condensed matter': 12}),
                 (1993, 8): defaultdict(int, {'condensed matter': 10}),
                 (1992, 5): defaultdict(int, {'condensed matter': 6}),
                 (1993,
                  5): defaultdict(int,
                             {'condensed matter': 9, 'materials science': 1}),
                 (1992, 6): defaultdict(int, {'condensed matter': 4})})


