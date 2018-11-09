# Recommendation tool for Beer
## Troy Kayne

I got the idea to make a recommendation tool for beer after discussing and practicing unsupervised learning techniques. I think it would be great for people to discover new beers that match their favorite qualities in taste and source.

I have found a dataset on kaggle that has a couple thousand records of beer and its associated brewery. Unfortunately, this dataset has a large amount of missing ibu values. The ibu is a crucial feature for my recommendations. However, there is a way to classify an average ibu value with respect to the style of the beer, or maybe regression is a better strategy in filling in this value. I am not going to prioritize my time on fixing this one feature. If I have to, I will find a free web API on programmable web that can potentially give me better data to work with.

I intend to use unsupervised learning strategies to compare a given preference of beer features with the rest of my dataset. My best option is to use cosine similarity to output the most similar vectors.

## Question

How many beers are out there that are similar to an input of desired beer qualities?

### MVP
 - Conduct an accurate SVD/PCA model to find the most similar beer options.
 - Input examples would be people's favorites brewery, ibu, abv, style, and maybe location.

### MVP+
 - Create a pipeline that cleanly organizes the steps of fit/transform/predict

### MVP++
 - Put pipeline into production by embedding it in a Flask Application
