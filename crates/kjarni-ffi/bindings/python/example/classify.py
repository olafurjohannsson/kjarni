from kjarni import Classifier

classifier = Classifier("distilbert-sentiment")
result = classifier.classify("i love kjarni")
print(result.all_scores)

