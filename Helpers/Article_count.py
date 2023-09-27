from datetime import datetime
import json

# Load the provided JSON file
with open("train.json", "r") as file:
    data = json.load(file)



def count_articles(month: int, year: int, subject: str, data: list) -> int:
    
    
    count = 0
    for article in data:
        # Extracting month and year from the date field of the article
        article_date = datetime.strptime(article['date'], '%Y-%m-%dT%H:%M:%S.%fZ')
        if article_date.month == month and article_date.year == year and article['subject'] == subject:
            count += 1

    return count

# Testing the function with the provided check cases
quantum_physics_count = count_articles(4, 2014, "quantum physics", data)
superconductivity_count = count_articles(4, 2014, "superconductivity", data)

#print the results
print(quantum_physics_count, superconductivity_count)
