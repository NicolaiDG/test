# Describe your project

## Dataset

Ik ga de dataset gebruiken van 'Apple Quality'. Ik zou een API call doen naar de Kaggle URL om de data te extracten.

De link naar de dataset vind u hier:

https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality

## Project Explanation

Wat ik wil bereiken is een een simpele Machine Learning algoritme die de kwaliteit tussen Appels klassificieert tussen goed en slecht adhv verschillende eigenschappen van die appels.

De data die ik heb bevat bepaalde categorische eigenschappen van appels en ook een dimensie quality met good / bad. Het doel is moest iemand appels verwerken met bepaalde informatie van appels zoals de size , sweetness , crunchiness , Juiciness ,... dat de algoritme bepaald dat die appels goed zijn of niet. 

Het is een vrij simpele algoritme, maar de kern erachter is om te focussen dat er een goeie pipeline wordt gebouwd dus daarom koos ik voor een simpele algoritme die een grote slagingskans heeft. Ik kan hierop hyperparameters zoeken en tuning erop uitvoeren om zo het model sterker en efficiënter te maken.

De dimensies in de dataset zijn:
* A_id
* Size
* Weight
* Sweetness
* Crunchiness
* Juiciness
* Ripeness
* Acidity
* Quality

## Flows & Actions 

### Data inlezen 

Ik zou een API request doen naar Kaggle om de data in te lezen in csv formaat zodat ik het kan gebruiken in mijn project.

### Preprosessing 
Ik zou de data filteren op basis van de dimensies die ik nodig heb voor mijn project, de NULL values eruit filteren en bijvoorbeeld de data van 'Quality' in de dataset converteren naar numerieke waarde aangezien de data alleen uit goed of slecht bevat zodat ik hier makkelijk mee kan werken. Ik zou ook de datatypes controleren van de dimensies en eventueel naar de bijpassende datatype converteren. Ik zou ook de data splitsen in training en testset. 

### Model seargh
Ik zou een task maken die probeert meerdere modellen te testen en beste accuracy te vinden zodat die model gekozen kan worden om mijn project te maken.

### Model training
Trainen van een klassificatiemodel op basis van de best gekozen model van mijn model seargh task.

### Hyperparameter tuning
De beste hyperparameters zoeken voor de geselecteerde klassificatiemodel en mogelijk de model hertrainen met behulp van MLFlow

### Evaluatie
Het model evalueren en beoordelen en eventueel de model extra finetunen.

### Service implementatie
Tools als MLflow en Prefect om continu appels input te verwerken en automatisch te klassificeren. Mogelijk een frontend ervoor ontwerpen.  



