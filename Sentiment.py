# -*- coding: utf-8 -*-
"""
Created on Wed May 17 18:07:06 2017

@author: Sergio
"""
#%%
import pandas
import numpy


products = pandas.read_csv('amazon_2.csv', low_memory=False, sep=',')
products = products.fillna({'name':''})  # fill in N/A's in the review column
products = products.fillna({'review':''})  # fill in N/A's in the review column
products.dtypes
products.review=products.review.astype(str)




#%%
#1. Eliminar signos de puntuación
#2. Eliminar reviews neutrales (rating 3).
#3. Definir reviews con rating >=4 como positivas (+1) y reviews con rating <= 2 negativas (-1)

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

products['review_clean'] = products['review'].apply(remove_punctuation)
#products['review_clean'] = products['review_clean'].map(lambda x: x.lower())
products['review_clean'] = products['review_clean'].apply(lambda x: x.lower())

# Eliminar reviews neutrales
products = products[products['rating'] != 3]
# Sentimiento positivo +1 y sentimiento negativo ‐1
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)


#%%

# Muestras train-test
numpy.random.seed(1) 
msk = numpy.random.rand(len(products)) < 0.8
train_data = products[msk]
test_data = products[~msk]
print len(train_data)
print len(test_data)    


#%%
# Entrenamiento de un clasificador logístico

# significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
#      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
#       'work', 'product', 'money', 'would', 'return']

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
# Representación de documentos como bolsa de palabras
#vectorizer = CountVectorizer(vocabulary=significant_words) # limit to 20 words

# Representacion de documentos de training a partir de la bolsa de palabras
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# Representacion de documentos de test a partir de la bolsa de palabras
test_matrix = vectorizer.transform(test_data['review_clean'])

# Entrenamiento de clasificador logístico
from sklearn import linear_model
model = linear_model.LogisticRegression()
model.fit(train_matrix, train_data['sentiment'])


#%%
# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true=test_data['sentiment'], y_pred=model.predict(test_matrix))
print "Test Accuracy: %s" % accuracy

#%%
# Baseline: Predicción de la clase mayoritaria
baseline = len(test_data[test_data['sentiment'] == 1])/float(len(test_data))
print "Baseline accuracy (majority class classifier): %s" % baseline

#%%
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cmat = confusion_matrix(y_true=test_data['sentiment'],y_pred=model.predict(test_matrix),labels=model.classes_) # use the same order of class as the LR model.
print ' target_label | predicted_label | count '
print '‐‐‐‐‐‐‐‐‐‐‐‐‐‐+‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐+‐‐‐‐‐‐‐'
# Imprimir matriz de confusión
for i, target_label in enumerate(model.classes_):
 for j, predicted_label in enumerate(model.classes_):
  print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j])



#%%
# Precision y Recall
from sklearn.metrics import precision_score
precision = precision_score(y_true=test_data['sentiment'],y_pred=model.predict(test_matrix))
print "Precision de datos de test: %s" % precision


from sklearn.metrics import recall_score
recall = recall_score(y_true=test_data['sentiment'],y_pred=model.predict(test_matrix))
print "Recall de datos de test: %s" % recall



#%%
# Precision recall tradeoff

# Varying the threshold

def apply_threshold(probabilities, threshold):
### YOUR CODE GOES HERE
# +1 if >= threshold and ‐1 otherwise.
  return [1 if x >= threshold else -1 for x in probabilities]

probabilities = model.predict_proba(test_matrix)[:,1]
predictions_with_default_threshold = apply_threshold(probabilities, 0.5)
predictions_with_high_threshold = apply_threshold(probabilities, 0.9)

print "Number of positive predicted reviews (threshold = 0.5): %s" % \
(sum([x for x in predictions_with_default_threshold if x == 1]))

print "Number of positive predicted reviews (threshold = 0.9): %s" % \
(sum([x for x in predictions_with_high_threshold if x == 1]))


precision_with_default_threshold = precision_score(y_true=test_data['sentiment'],y_pred=predictions_with_default_threshold)
recall_with_default_threshold = recall_score(y_true=test_data['sentiment'],y_pred=predictions_with_default_threshold)
precision_with_high_threshold = precision_score(y_true=test_data['sentiment'],y_pred=predictions_with_high_threshold)
recall_with_high_threshold = recall_score(y_true=test_data['sentiment'],y_pred=predictions_with_high_threshold)

print "Precision (threshold = 0.5): %s" % precision_with_default_threshold
print "Recall (threshold = 0.5) : %s" % recall_with_default_threshold

print "Precision (threshold = 0.9): %s" % precision_with_high_threshold
print "Recall (threshold = 0.9) : %s" % recall_with_high_threshold

#%%
# Precision recall curve
threshold_values = numpy.linspace(0.5, 0.999999, num=100)
precision_all = []
recall_all = []


for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)
    precision = precision_score(y_true=test_data['sentiment'],y_pred=predictions)
    recall = recall_score(y_true=test_data['sentiment'],y_pred=predictions)
    print 'Metrics Threshold %s Precision %s Recall %s' % (threshold, precision, recall)  
    precision_all.append(precision)
    recall_all.append(recall)


import matplotlib.pyplot as plt
#%matplotlib inline

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.xlim(0.94, 1.0)   #0.78
    plt.ylim(0.0, 1.0)
    plt.rcParams.update({'font.size': 16})

plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')


