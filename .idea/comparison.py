from Levenshtein import distance
import time
from sentence_transformers import CrossEncoder

text_a = "insertion sort"
text_b = "bubble sort"
model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

start = time.time()
levDistance = distance(text_a, text_b)
end = time.time()
print(f"Time: {end-start} s")
print(levDistance)

# start = time.time()
# scores = model.predict([
#     (text_a, text_b)
# ])
# end = time.time()
# print(f"Time: {end-start} s")
# print(scores)