# resume-screening
#standard classification of the resume for appropriate job description and provide the analysis so the companies with huge resumes can have a clear idea regarding the people who have applied.#


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(&#39;darkgrid&#39;)
import warnings
warnings.filterwarnings(&#39;ignore&#39;)
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
df = pd.read_csv(&#39;resume-dataset/UpdatedResumeDataSet.csv&#39;)
df.head(10)
category = df[&#39;Category&#39;].value_counts().reset_index()
category
def cleanResume(resumeText):
resumeText = re.sub(&#39;http\S+\s*&#39;, &#39; &#39;, resumeText) # remove URLs
resumeText = re.sub(&#39;RT|cc&#39;, &#39; &#39;, resumeText) # remove RT and cc
resumeText = re.sub(&#39;#\S+&#39;, &#39;&#39;, resumeText) # remove hashtags
resumeText = re.sub(&#39;@\S+&#39;, &#39; &#39;, resumeText) # remove mentions
resumeText = re.sub(&#39;[%s]&#39; % re.escape(&quot;&quot;&quot;!&quot;#$%&amp;&#39;()*+,-./:;&lt;=&gt;?@[\]^_`{|}~&quot;&quot;&quot;),
&#39; &#39;, resumeText) # remove punctuations
resumeText = re.sub(r&#39;[^\x00-\x7f]&#39;,r&#39; &#39;, resumeText)
resumeText = re.sub(&#39;\s+&#39;, &#39; &#39;, resumeText) # remove extra whitespace
return resumeText
df[&#39;cleaned&#39;] = df[&#39;Resume&#39;]. apply (lambda x:cleanResume(x))
df.head()
label = LabelEncoder()
df[&#39;Category&#39;] = label.fit_transform(df[&#39;Category&#39;])
df.head(20)
text = df[&#39;cleaned&#39;].values
target = df[&#39;Category&#39;].values
word_vectorizer = TfidfVectorizer(
sublinear_tf=True,
stop_words=&#39;english&#39;,
max_features=1500)
word_vectorizer.fit(text)

WordFeatures = word_vectorizer.transform(text)
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, target,
random_state=24, test_size=0.2)
model = OneVsRestClassifier(KNeighborsClassifier())
model.fit(X_train, y_train)
OneVsRestClassifier(estimator=KNeighborsClassifier())
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test, y_pred))
print(f&#39;---------------------------------\n| Training Accuracy :- {(model.score(X_train,
y_train)*100).round(2)}% |&#39;)
print(f&#39;---------------------------------\n| Validation Accuracy :- {(model.score(X_test,
y_test)*100).round(2)}% |\n---------------------------------&#39;)
plt.figure(figsize=(12,8))
sns.barplot(x=category[&#39;Category&#39;], y=category[&#39;index&#39;], palette=&#39;cool&#39;)
plt.show()
plt.figure(figsize=(12,8))
plt.pie(category[&#39;Category&#39;], labels=category[&#39;index&#39;],
colors=sns.color_palette(&#39;cool&#39;), autopct=&#39;%.0f%%&#39;)
plt.title(&#39;Category Distribution&#39;)
plt.show()
