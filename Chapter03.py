#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=True)
mnist.keys()


# In[19]:


X, y = mnist["data"], mnist["target"]
X.shape


# In[26]:


X.head()


# In[31]:


import numpy as np

X = np.array(X)


# In[32]:


X[:5]


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")

plt.show()


# In[34]:


y[0]


# In[35]:


y = y.astype(np.uint8)


# In[36]:


y


# In[42]:


X[:5]


# In[51]:


plt.figure(figsize=(10, 6))
for i in range(10):
    for j in range(10):
        some_digit = X[i * 10 + j]
        some_digit_image = some_digit.reshape(28, 28)
        plt.subplot(10, 10, i*10 + j + 1)  # subplot 위치를 먼저 잡고 그래프 그려주기!
        plt.imshow(some_digit_image, cmap="binary")
        plt.axis("off")
plt.show()


# In[52]:


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# In[53]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[55]:


from sklearn.linear_model import SGDClassifier

some_digit = X[0]
model = SGDClassifier()
model.fit(X_train, y_train)
model.predict([some_digit])


# In[57]:


from sklearn.model_selection import cross_val_score

model.decision_function([some_digit])
cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")


# ## 연습 문제
# 
# ### Spam or Ham

# In[68]:


import zipfile
import tarfile
import os
import urllib.request


# In[69]:


DOWNLOAD_ROOT = 'https://spamassassin.apache.org/old/publiccorpus/'
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")

def fetch_spam_data(ham_url=HAM_URL, spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()


# In[74]:


PATH = "C:/Users/csp/Desktop/DataSets/spamorham/"
HAM_PATH = PATH + "20030228_easy_ham.tar.bz2"
SPAM_PATH = PATH + "20030228_spam.tar.bz2"
NEW_PATH = os.path.join("DataSets", "spam")

for filename in (HAM_PATH, SPAM_PATH):
    tar_bz2_file = tarfile.open(filename)
    tar_bz2_file.extractall(path=NEW_PATH)
    tar_bz2_file.close()


# In[77]:


HAM_DIR = os.path.join(NEW_PATH, "easy_ham")
SPAM_DIR = os.path.join(NEW_PATH, "spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]


# In[78]:


len(ham_filenames)


# In[79]:


len(spam_filenames)


# In[82]:


import email
import email.policy

def load_email(is_spam, filename, spam_path=NEW_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


# In[83]:


ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]


# In[86]:


print(ham_emails[1].get_content().strip())


# In[1]:


import pandas as pd


# In[ ]:




