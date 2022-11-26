from nltk.corpus import stopwords
import csv,re
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# import spacy.cli
# spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

# with open('stsbenchmark/sts-train.csv', 'r', encoding = 'utf-8') as infile, open('repaired.csv','w', encoding='utf-8') as outfile:
#     for line in infile.readlines():
#         error = 0
#         try:
#             line = line.replace('","', '')
#             # line = line.replace('",', '')
#             outfile.write(line)
#         except:
#             error += 1
        
#     print(error)

with open('repaired.csv', 'r', encoding = 'utf-8-sig') as file:
    text1 = []
    text2 = []
    # check = []
    reader = csv.reader(file)
    count=1
    error = 0
    for row in reader:
        first_element = row[0]
        # check.append(first_element)
        
        try:
            sent2 = first_element.split('\t')[6]
            sent1 = first_element.split('\t')[5]
            
            sent1 = re.sub(r'\[[0-9]*\]',' ',sent1)
            sent1 = re.sub(r'\s+',' ',sent1)
            sent1 = sent1.lower()
            sent1 = tokenizer.tokenize(sent1)
            
            for i in range(len(sent1)):
                sent1 = [word for word in sent1 if word not in stopwords.words('english')]
            
            sent2 = re.sub(r'\[[0-9]*\]',' ',sent2)
            sent2 = re.sub(r'\s+',' ',sent2)
            sent2 = sent2.lower()
            sent2 = tokenizer.tokenize(sent2)
            
            for i in range(len(sent2)):
                sent2 = [word for word in sent2 if word not in stopwords.words('english')]
            
            text2.append(sent2)
            text1.append(sent1)
        except:
            error+=1 
        
    print(error)
    
import pandas as pd

data = pd.DataFrame({'text1': text1, 'text2': text2})

copydata=data.copy()
copydata.shape

# def remove_punc(copydata):
#   pattern = r'[' + string.punctuation + ']'
#   copydata['text1']=data['text1'].map(lambda m:re.sub(pattern," ",m))
#   copydata['text2']=data['text2'].map(lambda m:re.sub(pattern," ",m))
#   return copydata


# def lower(copydata):
#   copydata['text1']=copydata['text1'].map(lambda m:m.lower())
#   copydata['text2']=copydata['text2'].map(lambda m:m.lower())
#   return copydata


# def tokenization(text):
#     tokens = re.split(' ',text)
#     return tokens

# def token(copydata):
#   copydata['text1']= copydata['text1'].apply(lambda x: tokenization(x))
#   copydata['text2']= copydata['text2'].apply(lambda x: tokenization(x))
#   return copydata


# sw=nltk.corpus.stopwords.words('english')

# def remove_SW(copydata):
#    copydata['text1']=copydata['text1'].apply(lambda x: [item for item in x if item not in sw])
#    copydata['text2']=copydata['text2'].apply(lambda x: [item for item in x if item not in sw])
#    return copydata


# def remove_digits(copydata):
#   copydata['text1']=copydata['text1'].apply(lambda x: [item for item in x if not item.isdigit()])
#   copydata['text2']=copydata['text2'].apply(lambda x: [item for item in x if not item.isdigit()])
#   return copydata


# lemmatizer = WordNetLemmatizer()

# def lemmatize(copydata):
#   copydata['text1']=copydata['text1'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
#   copydata['text2']=copydata['text2'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
#   return copydata


# def remove_empty_tokens(copydata):
#   copydata['text1']=copydata['text1'].apply(lambda x: [item for item in x if item !=''])
#   copydata['text2']=copydata['text2'].apply(lambda x: [item for item in x if item !=''])
#   return copydata


# def remove_single_letters(copydata):
#   copydata['text1']=copydata['text1'].apply(lambda x: [item for item in x if len(item) > 1])
#   copydata['text2']=copydata['text2'].apply(lambda x: [item for item in x if len(item) > 1])
#   return copydata


# def detoken(copydata):
#   copydata['text1']= copydata['text1'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
#   copydata['text2']= copydata['text2'].apply(lambda x: TreebankWordDetokenizer().detokenize(x))
#   return copydata

# def replace_spaces(x,space,second):
#   result = x.replace(space, second)
#   return result
# def remove_space(copydata):
#   copydata['text1']= copydata['text1'].apply(lambda x: replace_spaces(x,'  ',' '))
#   copydata['text2']= copydata['text2'].apply(lambda x: replace_spaces(x,'  ',' '))
#   return copydata
# def count_vcr():
#   for i in range(len(copydata)):
#     doc1=copydata['text1'][i]
#     doc2=copydata['text2'][i]
#     docs=(doc1,doc2)
#     matrix = CountVectorizer().fit_transform(docs)
#     cosine_sim = cosine_similarity(matrix[0], matrix[1])
#     similarity.append(cosine_sim)
#   return similarity

# def similarity_fn():
#   for i in range(len(copydata)):
#     doc1=copydata['text1'][i]
#     doc2=copydata['text2'][i]
#     docs=(doc1,doc2)
#     tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
#     cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
#     similarity.append(cosine_sim)
#   return similarity

# copydata=copydata.pipe(remove_punc).pipe(token).pipe(remove_SW).pipe(remove_digits).pipe(lemmatize).pipe(remove_empty_tokens).pipe(remove_single_letters)

# bow_converter = CountVectorizer()
# copydata.pipe(detoken).pipe(remove_space)
# similarity=[]
# similarity=count_vcr()
# data_cvr=copydata.copy()
# data_cvr['Similarity']=similarity
# data_cvr[:5]

# tfidf_vectorizer = TfidfVectorizer()
# similarity=[]
# similarity=similarity_fn()
# data_tf=copydata.copy()
# data_tf['Similarity']=similarity

# all_data=data_cvr.copy()
# all_data['Count-Vec Similarity']=all_data['Similarity']
# all_data=all_data.drop('Similarity',axis=1)
# all_data['Tf-idf Similarity']=data_tf['Similarity']
