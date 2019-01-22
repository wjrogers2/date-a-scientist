import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import timeit
#from matplotlib import pyplot as plt

# NOTE: The only computer I had access to to do this programming was a chromebook.  
# I was able to get Anaconda installed, however as I was running in a 
# terminal in the Chrome app, it did not handle graphics very well.  
# When I tried to import pyplot it could result in an exception.  
# I was able to get around this by running my snippets of code in 
# one of the Jupyter notebooks from the optional projects and copying plot 
# to a file.  Therefore I do not have any plots in this code, however all 
# of the code I used to create the plots in my presentation are in the comments 
# below at the applicable steps.


#Create your df here:
df = pd.read_csv("profiles.csv")
print("END LOAD OF CSV")
# Create Column Mappings
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
drug_mapping = {"never": 0, "sometimes": 1, "often" : 2}
smoke_mapping = {"no": 0, "sometimes":1, "when drinking":2, "trying to quit":3, "yes":4}
sex_mapping = {"m":0, "f":1}
education_mapping = {"graduated from college/university":3, 
                     "graduated from masters program":4, 
                     "working on college/university":3,
                     "working on masters program":4,
                     "graduated from two-year college":2,
                     "graduated from high school":1,
                     "graduated from ph.d program":5,
                     "graduated from law school":4,
                     "working on two-year college":2,
                     "dropped out of college/university":1,
                     "working on ph.d program":5,
                     "college/university":3,
                     "graduated from space camp":0,
                     "dropped out of space camp":0,
                     "graduated from med school":5,
                     "working on space camp":0,
                     "working on law school":4,
                     "two-year college":2,
                     "working on med school":5,
                     "dropped out of two-year college":1,
                     "dropped out of masters program":3,
                     "masters program":4,
                     "dropped out of ph.d program":3,
                     "dropped out of high school":0,
                     "high school":1,
                     "working on high school":1,
                     "space camp":0,
                     "ph.d program":5,
                     "law school":4,
                     "dropped out of law school":3,
                     "dropped out of med school":3,
                     "med school":5}

# images
# 1, histogram of income from 0-200k
#test = df.drop(df[df.income < 0].index)
#test = test['income']
#print(len(test))
#plt.hist(test,bins=50)
#plt.xlabel("Income")
#plt.ylabel("Frequency")
#plt.xlim(0, 200000)
#plt.show()

df["drinks_map"] = df.drinks.map(drink_mapping)
df["smokes_map"] = df.smokes.map(smoke_mapping)
df["drugs_map"] = df.drugs.map(drug_mapping)
df["sex_map"] = df.sex.map(sex_mapping)
df["education_map"] = df.education.map(education_mapping)

essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
# create essay_len column
df["essay_len"] = all_essays.apply(lambda x: len(x))

#We also created a column with average word length and a column with the frequency of the words "I" or "me" appearing in the essays.
# remove all line break characters to get accurate word length calculations, and remove some common punctuation characters

all_essays = all_essays.apply(lambda x: x.replace("<br />"," ").replace(",","").replace(".","").replace(":","").replace(")","").replace("!","").replace("(",""))

all_essays = all_essays.apply(lambda x: x.split())
average_word_lengths = []
me_i_counts = []
for essay in all_essays:
    word_len_sum = 0
    me_i_count = 0
    for word in essay:
        word_len_sum += len(word)
        if word.lower() == "me" or word.lower() == "i":
            me_i_count+=1
    if len(essay) > 0:
        average_word_lengths.append(word_len_sum/len(essay))
    else:
        average_word_lengths.append(0)
    me_i_counts.append(me_i_count)
    
df["avg_word_length"] = average_word_lengths
df["me_i_count"] = me_i_counts

print("END OF INITIALIZING")
#---------------------------------------------------------------
# classification (k-nearest neighbors, support vector machines, naive bayes)
#---------------------------------------------------------------

#---------------------------------------------------------------
# CAN WE PREDICT SEX WITH EDUCATION LEVEL AND INCOME
rows_to_check = df.dropna(subset=['education_map', 'income','sex_map'])
feature_data = rows_to_check[['education_map', 'income','sex_map']]

# remove all the people who didn't put their income
feature_data = feature_data.drop(feature_data[feature_data.income < 0].index)

x = feature_data.values
min_max_scaler = pre.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

#plt.scatter(feature_data['education_map'],feature_data['income'] , c=feature_data['sex_map'],alpha=0.5)
#plt.xlabel("Scaled Education Map")
#plt.ylabel("Scaled Income")
#plt.show()

# split into training and test sets
sex_data = feature_data['sex_map']
feature_data = feature_data[['education_map','income']]
train_data, test_data, train_sex, test_sex = train_test_split(feature_data, sex_data, test_size = 0.2, random_state = 1)

print("PRE K-NEIGHBOR CLASSIFIER")
kscores = []
times = []
for i in range(1,200):
    start = timeit.default_timer()
    kneigh = KNeighborsClassifier(n_neighbors=i)
    kneigh.fit(train_data,train_sex)
    stop = timeit.default_timer()
    kscores.append(kneigh.score(test_data,test_sex))
    times.append(stop-start)

max_value = max(kscores)
max_index = kscores.index(max_value) # result is k=69


print("Max time of a single k-neighbors run = ", max(times)) # 0.22169
print("Time of optimal k-neighbor classifier = ", times[max_index]) #0.09988899
print("Total time for k-neighbor loop = ", sum(times)) # 17.61
print("Optimal k value = ", max_index)
print("Optimal accuracy = ", max_value)

start = timeit.default_timer()
svcclass = SVC(C=0.5)
svcclass.fit(train_data,train_sex)
stop = timeit.default_timer()
svc_time = stop-start
print("SVC Time to Run = ", svc_time) # 4.412 
print("SVC score: ", svcclass.score(test_data,test_sex))
# SVC got 72.69% while K=69 got 73.34%
    
#plt.plot(range(1,200),kscores)
#plt.xlabel("K-value")
#plt.ylabel("Accuracy")
#plt.title("Sex Classifier Accuracy")
#plt.show()
    
#---------------------------------------------------------------
#regression (multiple linear regression, k-nearest neighbor regression)
#---------------------------------------------------------------

#---------------------------------------------------------------
# CAN WE PREDICT AGE WITH FREQUENCY OF "I" OR "ME" IN ESSAYS?
rows_to_check = df.dropna(subset=['age','me_i_count'])
feature_data = rows_to_check[['age','me_i_count']]
feature_data = feature_data.drop(feature_data[feature_data.age >= 100].index)

#plt.scatter(feature_data['me_i_count'],feature_data['age'],alpha=0.5)
#plt.xlabel("Number of \'me\' or \'I\' in essays")
#plt.ylabel("Age")
#plt.show()

# don't need to scale the data because only one category
#x = feature_data.values
#min_max_scaler = pre.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

age_data = feature_data['age']
feature_data = feature_data['me_i_count']
train_data, test_data, train_age, test_age = train_test_split(feature_data, age_data, test_size = 0.2, random_state = 1)

# reshape for mlr fit
train_data = np.array(train_data)
test_data = np.array(test_data)
train_age = np.array(train_age)
test_age = np.array(test_age)
train_data = train_data.reshape(-1, 1)
train_age = train_age.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)
test_age = test_age.reshape(-1, 1)

start = timeit.default_timer()
mlr = LinearRegression()
model=mlr.fit(train_data,train_age)
stop = timeit.default_timer()
print("Time to run linear regression: ", stop-start)
age_predict = mlr.predict(test_data)
print("Score of training data: ", mlr.score(train_data,train_age)) # 0.003605
print("Score of test data: ", mlr.score(test_data,test_age)) # 0.004836

#plt.scatter(test_age, age_predict, alpha=0.5)
#plt.xlabel("Actual Age")
#plt.ylabel("Predicted Age")

#plt.scatter(feature_data, age_data, alpha=0.5)
#plt.plot(test_data,age_predict)
#plt.xlabel("Number of \'me' or 'I' in essays")
#plt.ylabel("Age")

kscores = []
times = []
predictions = []
for i in range(1,200):
    start = timeit.default_timer()
    kneigh = KNeighborsRegressor(n_neighbors=i, weights="distance")
    kneigh.fit(train_data,train_age)
    stop = timeit.default_timer()
    predictions.append(kneigh.predict(test_data))
    kscores.append(kneigh.score(test_data,test_age))
    times.append(stop-start)

print("Max k-neighbor regressor time: ", max(times)) # 0.5699978
print("Min k-neighbor regressor time: ", min(times)) # 0.3845
print("Total kneighbor regressor time: ", sum(times)) # 78.429

#plt.plot(range(1,200),kscores)
#plt.xlabel("K-value")
#plt.ylabel("R^2")
#plt.title("Age Regressor Score")
#plt.show()

#plt.scatter(test_age,predictions[50],alpha=0.5)
#plt.xlabel("Actual Age")
#plt.ylabel("Regressor Predicted Age (k=50)")
#plt.show()
