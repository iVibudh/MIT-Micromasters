import project1 as p1
import utils
import numpy as np

#-------------------------------------------------------------------------------
# Data loading. There is no need to edit code in this section.
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))


#-------------------------------------------------------------------------------
# Problem 9.1
#-------------------------------------------------------------------------------
# Usual Case
# dictionary = p1.bag_of_words(train_texts)
#
# train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
# val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
# test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)
#
# # Best Model is Pegasos
# # Best Parameters T = 25, L = 0.0100
# T = 25
# L = 0.01
#
#
# avg_peg_train_accuracy, avg_peg_val_accuracy = \
#    p1.classifier_accuracy(p1.pegasos, train_bow_features, test_bow_features, train_labels, test_labels,T=T,L=L)
# print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
# print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))

# Case with stopwords
stopwords = []
with open('stopwords.txt','r') as f:
    for line in f:
        for word in line.split():
           stopwords.append(word)
print(len(stopwords))
stopwords_set = set(stopwords)

def remove_stopwords(train_texts):
    train_texts_list = []
    for line in train_texts:
        line_clean = ""

        line = str(line)
        for w in line.split():
            if w not in stopwords_set:
                line_clean = line_clean + w + " "
        train_texts_list.append(line_clean)
    train_texts_tuple = tuple(train_texts_list)
    return(train_texts_tuple)

train_texts_no_stop = remove_stopwords(train_texts)
# val_texts_no_stop = remove_stopwords(val_texts)
# test_texts_no_stop = remove_stopwords(test_texts)
#
# dictionary = p1.bag_of_words(train_texts_no_stop)
# print(type(dictionary))
# print(dictionary)
# train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
# val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
# test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)
# print(train_bow_features)
# print(len(train_bow_features))
# print(len(train_bow_features[1]))
# T = 25
# L = 0.01
#
#
# avg_peg_train_accuracy, avg_peg_val_accuracy = \
#    p1.classifier_accuracy(p1.pegasos, train_bow_features, test_bow_features, train_labels, test_labels,T=T,L=L)
# print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
# print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))
#
#-------------------------------------------------------------------------------
# Problem 9.2
#-------------------------------------------------------------------------------
# Use a different extract_bow_feature_function
# Use the same learning algorithm and the same feature as the last problem.
# However, when you compute the feature vector of a word, use its count in each document rather than a binary indicator.

dictionary_no_stop = p1.bag_of_words(train_texts_no_stop)

train_bow_features = p1.extract_bow_feature_vectors_count(train_texts, dictionary_no_stop)
print(train_bow_features)

print(len(train_bow_features))
print(len(train_bow_features[1]))

val_bow_features = p1.extract_bow_feature_vectors_count(val_texts, dictionary_no_stop)
test_bow_features = p1.extract_bow_feature_vectors_count(test_texts, dictionary_no_stop)
T = 25
L = 0.01


avg_peg_train_accuracy, avg_peg_val_accuracy = \
   p1.classifier_accuracy(p1.pegasos, train_bow_features, val_bow_features, train_labels, val_labels,T=T,L=L)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))