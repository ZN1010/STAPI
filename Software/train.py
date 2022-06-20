import glob
from bs4 import BeautifulSoup, Comment
import bs4
import re
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
import sys
import joblib
import csv
from itertools import zip_longest
import xgboost as xgb

import torch
from transformers import BertTokenizer, BertModel
import nltk
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc
import sklearn

"""
This function takes an HTML element as input
and checks whether it is a special HTML tag or not.
Special HTML tags are those for formatting purposes (https://www.w3schools.com/html/html_formatting.asp).
We believe that these formatting tags should not break the original
text (e.g., a paragraph) into 2 separate parts.
"""
def checkSpecialTag(e):
	# {"b", "a", "strong", "i", "small", "mark", "del", "ins", "sub", "sup"}
	if e.name in {"a", "i", "small", "mark", "del", "ins", "sub", "sup", "u", "strong", "b"}:
		return True
	return False

"""
This function takes an HTML element as input
and checks whether it has special style or not.
An HTML element that has special style will have CSS code
"display: block;" and will appear as the first child of its parent.
We believe that an HTML element with the special style should be
separated from its parent.
"""
def checkSpecialStyle(e):
	display_block, isFirst = False, False
	if e.has_attr("style"):
		if "display" in e['style'] and "block" in e['style']:
			display_block = True

	if display_block and e.parent.find(text=True, recursive=False) != None:
		temp = list(e.parent)
		if isinstance(temp[0], bs4.element.Tag) and e.find(text=True, recursive=False) == temp[0].find(text=True, recursive=False):
			isFirst = True
	return display_block and isFirst



"""
This function builds filter's feature list for a single
Web page. 
"""
def build_filter_features(raw_text, d):
	size, rst = len(raw_text), []
	for i in range(size):
		word_length = len(raw_text[i].split())
		relative_position = (i+1) / size

		current_tag = d[i].name
		parent_tag, next_tag = d[i].parent.name, "None"
		if i < size - 1:
			next_tag = d[i+1].name
		tag_id = "None"
		# if d[i].has_attr("class"):
		# 	tag_class = d[i]['class']
		if d[i].has_attr("id"):
			tag_id = d[i]['id']
		rst.append([word_length, relative_position, current_tag, parent_tag, next_tag, tag_id])
	return rst


"""
This function builds typographic classifier's feature list for a single
Web page. 
"""
def build_classifier_features(nonzero_text, d, idx_interested):
	size, rst = len(nonzero_text), []
	for i in range(size):
		word_length = len(nonzero_text[i].split())

		ratio_length_next_length, next_tag = 0, "None"
		if i < size - 1:
			ratio_length_next_length = word_length / len(nonzero_text[i+1].split())
			next_tag = d[idx_interested[i+1]].name

		num_punc_symbol = len("".join([c if not c.isalnum() and not c.isspace() else "" for c in nonzero_text[i]]))

		current_tag = d[idx_interested[i]].name

		rst.append([word_length, ratio_length_next_length, num_punc_symbol, current_tag, next_tag])
	return rst

"""
This function creates labels for each Web document based
on the gold standard. 0 for throwing away, 1 for keeping the original text,
and 2 for including its children.
"""
def create_labels(raw_text, raw_text_test, d_test):
	size = len(raw_text)
	labels, start_idx = [[0] * 2 for x in range(size)], 0
	for target_idx in range(len(raw_text_test)):
		for i in range(start_idx, size):
			if raw_text[i].lower() == raw_text_test[target_idx].lower():
				labels[i][0] = 1
				labels[i][1] = typographic_element[d_test[target_idx].name]
				start_idx = i + 1
				break
	return labels


"""
This function embeds a sentence.
"""
def embed_sentence(sentence, bert_tokenizer, model):
	marked_text = "[CLS] " + sentence + " [SEP]"
	tokenized_text = bert_tokenizer.tokenize(marked_text)
	if len(tokenized_text) > 512:
		tokenized_text[511] = '[SEP]'
		tokenized_text = tokenized_text[:512]

	indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
	segments_ids = [1] * len(tokenized_text)
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])

	with torch.no_grad():
		outputs = model(tokens_tensor, segments_tensors)
		hidden_states = outputs[2]
	layer_collection = []
	for i in range(len(hidden_states)):
		if i == 0:
			continue
		layer_collection.append(torch.mean(hidden_states[i][0], dim=0).numpy())
	return np.mean(layer_collection, axis=0)



"""
This function embeds a text segment (with one
or multiple sentences).
"""
def embed_raw_text(text_collection, bert_tokenizer):
	rst = []
	model = BertModel.from_pretrained('bert-base-cased', output_hidden_states = True)
	model.eval()
	for i in tqdm(range(len(text_collection))):
		a_list = nltk.tokenize.sent_tokenize(text_collection[i])
		if len(a_list) <= 1:
			rst.append(embed_sentence(text_collection[i], bert_tokenizer, model))
		else:
			sentence_collection = []
			for e in a_list:
				sentence_collection.append(embed_sentence(e, bert_tokenizer, model))
			rst.append(np.mean(sentence_collection, axis=0))
	return rst


"""
This function is a helper function for drawing
the ROC curve of the typographic classifier.
"""
def plot_multiclass_roc(clf, X_test, y_test, n_classes):
    y_score = clf.predict_proba(X_test)
    plt.rcParams.update({'font.size': 15})
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.rcParams.update({'font.size': 18})
    
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
#     fig, ax = plt.subplots(figsize=plt.rcParams.get('figure.figsize'))
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve of Typographic Classifier')
    l = ["title", "prose text", "miscellany"]
    for i in range(3):
        ax.plot(fpr[i], tpr[i], label='ROC curve (AUC = %0.2f) of %s' % (roc_auc[i], l[i]))
    ax.legend(loc="lower right")
    sns.despine()
    plt.show()


"""
This function reads the content of HTML 
and proposes the input for filter.
"""
def process_file(contents, d, skip):
	s = re.sub('<br>', '<p>', contents)
	s = re.sub('</br>', '<p>', s)
	s = re.sub('<br/>', '<p>', s)
	s = s.replace("\n", " ")
	s = s.replace("&nbsp;", " ")
	soup = BeautifulSoup(s, 'lxml')

	# Remove Comments from HTML Tags
	comments = soup.findAll(text=lambda text:isinstance(text, Comment))
	for comment in comments:
		comment.extract()

	body = soup.body

	# replace <a> tag with its child or its text
	for a in body("a"):
		text = a.find(text=True, recursive=False)
		if text == None:
			a.replaceWithChildren()
			continue
		a.unwrap()
    
	if skip != "":
		for script in body(skip):
			script.decompose()
    
	for script in body("script"):
		script.decompose()
	for style in body("style"):
		style.decompose()
	for nav in body("nav"):
		nav.decompose()
	for footer in body("footer"):
		footer.decompose()
	for hr in body("hr"):
		hr.decompose()

	# Newly Added from Java Version
	for button in body("button"):
		button.decompose()
	for table in body("table"):
		table.decompose()
	for img in body("img"):
		img.decompose()
	for figure in body("figure"):
		figure.decompose()
	for code in body("code"):
		code.decompose()
	for details in body("details"):
		details.decompose()
	for svg in body("svg"):
		svg.decompos

	children = [c for c in body.findChildren() if len(list(c)) != 0]

	
	size, idx, raw_text, idx_back, i = len(children), 0, [], -1, 0
	
	# preserve the order of each HTML element
	while idx < size:
		temp = [x for x in list(children[idx]) if not (isinstance(x, str) and x.strip() == "")]
		
		# if len(temp) == 0 or children[idx].find(text=True, recursive=False) == None:
		if len(temp) == 0:
			idx += 1
			continue
		
		if idx_back < idx and isinstance(temp[0], bs4.element.Tag):
			idx_back = idx + 1
			while idx_back < size and isinstance(list(children[idx_back]), bs4.element.Tag):
				idx_back += 1
			if idx_back < size:
				children[idx: idx_back+1] = children[idx: idx_back+1][::-1]
				temp = list(children[idx])

		if checkSpecialTag(children[idx]) and not checkSpecialStyle(children[idx]):
			idx += 1
			continue
		for j in range(len(temp)):
			if checkSpecialTag(temp[j]) and not checkSpecialStyle(temp[j]):
				temp[j] = str(temp[j].text)

		text = "".join([t for t in temp if isinstance(t, str)])
		children_d = children[idx]

		# remove some unreadable characters
		text = text.replace("\xa0", " ")

		trimmed = text.lower().strip().replace("\n", " ").replace("  ", " ")
		idx += 1
		if trimmed == "none" or trimmed == "":
			continue

		# remove extra white spaces
		removed = text.strip().replace("\n", " ").replace("  ", " ").split()
		raw_text.append(" ".join(removed))
		d[i] = children_d
		i += 1
	return raw_text, d

"""
This function reads an HTML document. It takes
the directory of the file as input. It returns
a list of text segments along with their HTML elements.
"""
def read_file(file, skip=""):
	with open(file, "r", encoding="utf-8", errors='ignore') as f:
		contents = f.read()
		raw_text, d = process_file(contents, {}, skip)
		
	return raw_text, d


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          m=0):


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""
    
    plt.rcParams.update({'font.size': 15})

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    
    if m == 0:
        plt.yticks(np.arange(2)+0.5,('Irrelevance','Pertinence'), va="center")
    else:
        plt.yticks(np.arange(3)+0.5,("Title", "Prose Text", "Miscellany"), va="center")


# set a seed to get deterministic results
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

typographic_element = {"h2": 1, "p": 2, "div": 3}
avoid = {'Data/Misc/squarespace/Misc.html','Data/Misc/Feat/Misc.html','Data/TOS/CBSLocal/TOS.html','Data/TOS/MS/TOS.html','Data/TOS/meetup/TOS.html', 'Data/PP/conservativereview.com/priv.html'}
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

data_misc = glob.glob(os.path.join('Data/Misc', '*', 'Misc.html'))
data_tos = glob.glob(os.path.join('Data/TOS', '*', 'TOS.html'))
data_pp = glob.glob(os.path.join('Data/PP', '*', 'priv.html'))

vali_test_idx = np.random.choice(291, 116, False)
vali_idx, test_idx = vali_test_idx[:58], vali_test_idx[58:]
data, filter_features, classifier_features, labels_train, data_train, data_vali, data_test = data_pp + data_misc + data_tos, [], [], [], [], [], []
count = 0
s33 = set(vali_idx)
for e in test_idx:
    s33.add(e)
    
for i in range(len(data)):
    if data[i] in avoid:
        continue
        
    if i in vali_idx:
        data_vali.append(data[i])
    elif i in test_idx:
        data_test.append(data[i])
    else:
        data_train.append(data[i])

print("test id:", test_idx)

text_train, text_vali, text_test = [], [], []
for file in data_train:
    if file in avoid:
        continue
    raw_text, d = read_file(file)
    text_train += raw_text
    file_break = file.split("/")
    file_break[-1] = "gold.html"
    test_file = "/".join([f for f in file_break])
    raw_text_test, d_test = read_file(test_file)
    len1 = len(raw_text_test)
    file_label = create_labels(raw_text, raw_text_test, d_test)

    filter_features += build_filter_features(raw_text, d)

    idx_interested = [i for i in range(len(raw_text)) if file_label[i][0] == 1]
    nonzero_text = [raw_text[i] for i in idx_interested]
    classifier_features += build_classifier_features(nonzero_text, d, idx_interested)
    labels_train = labels_train + file_label

filter_features_vali, classifier_features_vali, labels_vali = [], [], []
for file in data_vali:
    if file in avoid:
        continue
    raw_text, d = read_file(file)
    text_vali += raw_text
    file_break = file.split("/")
    file_break[-1] = "gold.html"
    test_file = "/".join([f for f in file_break])
    raw_text_test, d_test = read_file(test_file)
    file_label = create_labels(raw_text, raw_text_test, d_test)

    filter_features_vali += build_filter_features(raw_text, d)

    idx_interested = [i for i in range(len(raw_text)) if file_label[i][0] == 1]
    nonzero_text = [raw_text[i] for i in idx_interested]
    classifier_features_vali += build_classifier_features(nonzero_text, d, idx_interested)
    labels_vali = labels_vali + file_label

filter_features_test, classifier_features_test, labels_test, size_each_test = [], [], [], []
for file in data_test:
    raw_text, d = read_file(file)
    size_each_test.append(len(raw_text))
    text_test += raw_text
    file_break = file.split("/")
    file_break[-1] = "gold.html"
    test_file = "/".join([f for f in file_break])
    raw_text_test, d_test = read_file(test_file)
    file_label = create_labels(raw_text, raw_text_test, d_test)
    len2 = np.count_nonzero(np.array(file_label) > 0)
    filter_features_test += build_filter_features(raw_text, d)

    idx_interested = [i for i in range(len(raw_text)) if file_label[i][0] == 1]
    nonzero_text = [raw_text[i] for i in idx_interested]
    classifier_features_test += build_classifier_features(nonzero_text, d, idx_interested)

    labels_test = labels_test + file_label

# create 3 lists that hold all the sets of categorical features from filter and classifier
f_set, c_set = [], []
for i in range(4):
    s = set()
    for f in filter_features:
        s.add(f[i + 2])
    for f in filter_features_vali:
        s.add(f[i + 2])
    for f in filter_features_test:
        s.add(f[i + 2])
    order_l = list(s)
    order_l.sort(reverse=True)
    f_set.append(order_l)

for i in range(2):
    s = set()
    for f in classifier_features:
        s.add(f[i + 3])
    for f in classifier_features_vali:
        s.add(f[i + 3])
    for f in classifier_features_test:
        s.add(f[i + 3])
    order_l = list(s)
    order_l.sort(reverse=True)
    c_set.append(order_l)

# save categorical_features of both classifier and filter to a csv file
with open('output/categorical_features.csv', 'w') as f:
    w = csv.writer(f)
    for r1, r2, r3, r4, r5, r6 in zip_longest(f_set[0], f_set[1], f_set[2], f_set[3], c_set[0], c_set[1]):
        w.writerow([r1, r2, r3, r4, r5, r6])

bert_feature = embed_raw_text(text_train, tokenizer)
bert_feature_vali = embed_raw_text(text_vali, tokenizer)
bert_feature_test = embed_raw_text(text_test, tokenizer)
bert_feature, bert_feature_vali = np.array(bert_feature), np.array(bert_feature_vali) 
bert_feature_test = np.array(bert_feature_test)

filter_numerical, filter_categorical, size_F  = [], [], len(filter_features)
size_F_vali = len(filter_features_vali)

for row in filter_features:
    filter_numerical.append([row[0], row[1]])
    filter_categorical.append([row[2], row[3], row[4], row[5]])

for row in filter_features_vali:
    filter_numerical.append([row[0], row[1]])
    filter_categorical.append([row[2], row[3], row[4], row[5]])
    
for row in filter_features_test:
    filter_numerical.append([row[0], row[1]])
    filter_categorical.append([row[2], row[3], row[4], row[5]])

enc_filter = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_F = enc_filter.fit_transform(filter_categorical)

train_x_F, train_y_F = np.hstack((np.array(filter_numerical)[:size_F, :], encoded_F[:size_F, :], bert_feature)), np.array([l[0] for l in labels_train])
vali_x_F, vali_y_F = np.hstack((np.array(filter_numerical)[size_F:(size_F+size_F_vali), :], encoded_F[size_F:(size_F+size_F_vali), :], bert_feature_vali)), np.array([l[0] for l in labels_vali])
test_x_F, test_y_F = np.hstack((np.array(filter_numerical)[(size_F+size_F_vali):, :], encoded_F[(size_F+size_F_vali):, :], bert_feature_test)), np.array([l[0] for l in labels_test])
joblib.dump(enc_filter, 'output/encoder_F.joblib')
print("The encoder for the filter is successfully saved!")

l = [x for x in range(2, encoded_F[:size_F, :].shape[1] + 2)]
num_0 = np.count_nonzero(train_y_F == 0)
print("Upsampling for preparing the data for the filter")
smote_nc = SMOTENC(categorical_features=l, sampling_strategy={1: num_0}, random_state=0)
train_x_F_over, train_y_F_over = smote_nc.fit_sample(train_x_F, train_y_F)

##### Test on Filter #####
target_names = ['class 0', 'class 1']
xgb_F = xgb.XGBClassifier(learning_rate=0.1)
print("Fitting the training model for the filter...")
xgb_F.fit(train_x_F, train_y_F)
pred_F = xgb_F.predict(test_x_F)
print(accuracy_score(test_y_F, pred_F))
print(classification_report(test_y_F, pred_F, target_names=target_names, digits=5))
matrix_F = confusion_matrix(test_y_F, pred_F, labels=[0, 1])
categories = ["Irrelevance", "Pertinence"]
make_confusion_matrix(matrix_F, categories=categories, sum_stats=False, title="Confusion Matrix of Filter")

joblib.dump(xgb_F, 'output/xgb_F.pkl')
print("The filter is successfully saved!")

plt.rcParams.update({'font.size': 15})
sklearn.metrics.plot_roc_curve(xgb_F, test_x_F, test_y_F, response_method="predict_proba",drop_intermediate=False, name="ROC Curve")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("ROC Curver of Filter\n(Positive label: Pertinence)")
plt.show()


bert_feature_C = np.array([bert_feature[i] for i in range(len(labels_train)) if labels_train[i][0] == 1])
bert_feature_vali_C = np.array([bert_feature_vali[i] for i in range(len(labels_vali)) if labels_vali[i][0] == 1])
bert_feature_test_C = np.array([bert_feature_test[i] for i in range(len(labels_test)) if labels_test[i][0] == 1])

classifier_numerical, classifier_categorical, size_C, size_C_vali = [], [], len(classifier_features), len(classifier_features_vali)
for row in classifier_features:
    classifier_numerical.append([row[0], row[1], row[2]])
    classifier_categorical.append([row[3], row[4]])

for row in classifier_features_vali:
    classifier_numerical.append([row[0], row[1], row[2]])
    classifier_categorical.append([row[3], row[4]])

for row in classifier_features_test:
    classifier_numerical.append([row[0], row[1], row[2]])
    classifier_categorical.append([row[3], row[4]])

enc_classifier = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_C = enc_classifier.fit_transform(classifier_categorical)

train_x_C, train_y_C = np.hstack((np.array(classifier_numerical)[:size_C, :], encoded_C[:size_C, :], bert_feature_C)), np.array([l[1] for l in labels_train if l[0] == 1])
vali_x_C, vali_y_C = np.hstack((np.array(classifier_numerical)[size_C: (size_C+size_C_vali), :], encoded_C[size_C: (size_C+size_C_vali), :], bert_feature_vali_C)), np.array([l[1] for l in labels_vali if l[0] == 1])
test_x_C, test_y_C = np.hstack((np.array(classifier_numerical)[(size_C+size_C_vali):, :], encoded_C[(size_C+size_C_vali):, :], bert_feature_test_C)), np.array([l[1] for l in labels_test if l[0] == 1])
joblib.dump(enc_classifier, 'output/encoder_C.joblib')
print("The encoder for the typographic classifier is successfully saved!")


num_2 = np.count_nonzero(train_y_C == 2)
l = [x for x in range(3, encoded_C[:size_C, :].shape[1] + 3)]
print("Upsampling for preparing the data for the typographic classifier")
sm = SMOTENC(categorical_features=l, sampling_strategy={1: num_2, 3: num_2}, random_state = 0)
train_x_C_over, train_y_C_over = sm.fit_sample(train_x_C, train_y_C)

##### Test on Classifier #####
target_names = ['class 1', 'class 2', 'class 3']
xgb_C = xgb.XGBClassifier(objective='multi:softmax', learning_rate=0.65)
print("Fitting the training model for the typographic classifier...")
xgb_C.fit(train_x_C_over, train_y_C_over)
pred_C = xgb_C.predict(test_x_C)
print(accuracy_score(test_y_C, pred_C))
print(classification_report(test_y_C, pred_C, target_names=target_names, digits=8))
matrix_C, categories = confusion_matrix(test_y_C, pred_C, labels=[1, 2, 3]), ["Title", "Prose Text", "Miscellany"]
make_confusion_matrix(matrix_C, categories=categories, sum_stats=False, title="Confusion Matrix of Typographic Classifier", m=1)

joblib.dump(xgb_C, 'output/xgb_C.pkl')
print("The typographic classifier is successfully saved!")

plot_multiclass_roc(xgb_C, test_x_C, test_y_C, n_classes=3)


