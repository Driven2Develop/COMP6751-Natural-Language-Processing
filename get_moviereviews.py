import nltk

nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

# Accessing and viewing the movie reviews line by line
# print("Negative Reviews")
# for index, fileid in enumerate([name for name in movie_reviews.fileids() if name.startswith("neg")][:50]):
#     review_format = "{}. {}".format(index+1, movie_reviews.raw(fileid))
#     print(review_format)


print("Positive Reviews")
for index, fileid in enumerate([name for name in movie_reviews.fileids() if name.startswith("pos")][:50]):
    review_format = "{}. {}".format(index+1, movie_reviews.raw(fileid))
    print(review_format)