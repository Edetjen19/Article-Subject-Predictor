import json

# Load the provided JSON file
with open("train.json", "r") as file:
    data = json.load(file)



unique_subjects = set(article['subject'] for article in data)

# Saving the unique subjects to a text file
file_path = "unique_subjects.txt"
with open(file_path, "w") as file:
    for subject in unique_subjects:
        file.write(subject + "\n")

