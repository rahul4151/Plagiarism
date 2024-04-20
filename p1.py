from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import fitz

f1 = input("Enter file name : ")
d1 = fitz.open(f1)
s1 = ""
for d in d1:
    s1 = s1 + d.get_text()

f2 = input("Enter file name : ")
d2 = fitz.open(f1)
s2 = ""
for d in d2:
    s2 = s2 + d.get_text()

cv = CountVectorizer()
vectors = cv.fit_transform([s1,s2])

features = pd.DataFrame(vectors.toarray(),columns=cv.get_feature_names_out())

cs = cosine_similarity(vectors)
ans = cs[0,1]
print("Similarity is :",round(ans*100,2),"%")
