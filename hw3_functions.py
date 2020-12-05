#----------------------------------import------------------------------

from bs4 import BeautifulSoup
import requests
import math
import os
import shutil
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.firefox import GeckoDriverManager
import codecs
from langdetect import detect
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
import json
from numpy import linalg as la
import time
import random
import seaborn as sns
import string
import heapq 
import matplotlib.pyplot as plt
import editdistance
nltk.download('stopwords')

def import_files():
    
    with open('vocabulary.txt') as voc:
        vocabulary = json.load(voc)
    with open('inverted_index.txt', 'r') as index:
        inverted_index = json.load(index)
    with open('inverted_index_score.txt') as index_score:
        inverted_index_score = json.load(index_score)
    with open('inverted_index_score.txt') as index_score:
        inverted_index_score = json.load(index_score)
    with open('d_vector.txt') as d:
        d_vector = json.load(d)
    with open('articles_with_plot.txt') as d:
        articles = json.load(d)
    
    return vocabulary, inverted_index, inverted_index_score, d_vector, articles

vocabulary,inverted_index, inverted_index_score, d, articles = import_files()

#------------------------------ 1.1 -------------------------------------

def urls(): 
    # since we want to get the first 300 pages we will start from page 1
    page = requests.get("https://www.goodreads.com/list/show/1.Best_Books_Ever?page=1") 
    soup = BeautifulSoup(page.content, features='lxml')
    urlfile=open('url.tx')
    # in order to get last page as 300th we stop when "next page" is 302
    while (soup.find('a', class_ = "next_page", rel = "next").attrs['href'] != "/list/show/1.Best_Books_Ever?page=302"):
        for a in soup.find_all('a', href=True, itemprop="url", class_="bookTitle"):
            urlfile.write(str(a['href'])+"\n")
    urlfile.close()
    return 1

def download():
    url = open('url.txt', 'r') 
    Lines = url.readlines()
    n_books=len(Lines)/300
    n_pages=300
    
    driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())

    # creating empty folders (page_1 to page_300)
    for n in range(1,n_pages+1):
        os.mkdir(f"page{n}") 
    
    # downloading each html
    for i in range(len(Lines)):
        line=Lines[i]
        page = driver.get("https://www.goodreads.com/"+str(line))
        soup = BeautifulSoup(driver.page_source, features='lxml')
        text = soup.prettify()
        page_number = int(i/n_books)+1 # round up the ratio to assign folder
        f = open(f"{i}.html", "w")
        f.write(text)
        f.close()
        shutil.move(f"{i}.html", f"page{page_number}/{i}.html")
    return 1

#------------------------------ 1.2 -------------------------------------
# this function will be used for bookCharacters and bookSetting 
# it converts resultset type of object to a list
def resultset_to_list(result_set):
    result =[]
    for i in result_set:
        result.extend(i)
    result = [i.strip() for i in result]
    return ','.join(result)


def parse(book_directory,url_line):
    
    book = codecs.open(book_directory, 'r')
    soup = BeautifulSoup(book.read())
    url = open('url.txt', 'r') 
    Lines = url.readlines()
    # initializing all variables
    bookTitle=bookSeries=bookAuthors=ratingValue=ratingCount=reviewCount=Plot=NumberofPages=publishingDate=bookCharacters=bookSetting=bookUrl=''
    bookUrl=Lines[url_line]
    
    # checking booktitle
    try:
        bookTitle=soup.find_all('h1')[0].contents[0].split('\n')[1].strip()
    except:
        # we assume these htmls are corrupted so we store their urls in order to download them again
        print(f" \n ERROR: html not valid. Try to download this html again: {bookUrl} \n")
        broken_urls=open('broken_url.txt', 'a') 
        broken_urls.write(f'https://www.goodreads.com{bookUrl}')
        broken_urls.write(book_directory)
        broken_urls.close()
    
    # checking plot
    try:
        plot = []
        # we use try because we have some books that have more sections in the plot
        # while some others only have one section (due to the presence of the "more" button in the web page)
        try:
            # pages with extra plot information
            plot_first = soup.find_all('div', id='description')[0].contents[3]
            for i in plot_first.stripped_strings:
                plot.append(" "+ str(i))
                Plot = [''.join(plot[:])][0]
        except:
            # pages with no extra plot information
            plot_first = soup.find_all('div', id='description')[0]
            for i in plot_first.stripped_strings:
                plot.append(" "+ str(i))
                Plot = [''.join(plot[:])][0]
    except:
        print(f" Warning: No plot available for {book_directory} ")
    
    # checking secondary fields
    # each html may have a slightly different structure so we tried to generalize the code to extract as much informationas possible
    try:
        bookAuthors=soup.find_all('span',itemprop="name")[0].contents[0].strip()
    except:
        print(f' Warning: missing bookAuthors for {book_directory}')
    try:
        NumberofPages = str(soup.find_all('span',itemprop="numberOfPages")[0].contents[0].split('\n')[1].strip().split(' ')[0])
    except:
        print(f' Warning: missing NumberofPages for {book_directory}')
    try:
        bookSeries = soup.find_all('a', class_ ='greyText', href = True)[0].contents[0].strip()
    except:
        print(f' Warning: missing bookSeries for {book_directory}')
    try:
        ratingValue = str(soup.find_all('span',itemprop="ratingValue")[0].contents[0].split('\n')[1].strip())
    except:
        print(f' Warning: missing ratingValue for {book_directory}')
    try:
        ratingCount= str(soup.find(itemprop ='ratingCount').get('content'))
    except:
        print(f' Warning: missing ratingCount for {book_directory}')
    try:
        reviewCount= str(soup.find(itemprop ='reviewCount').get('content'))
    except:
        print(f' Warning: missing reviewCount for {book_directory}')
    try:
        publishingDate= soup.find_all('div', class_ = "row")[1].contents[0].split('\n')[2].strip()
    except:
        print(f' Warning: missing publishingDate for {book_directory}')
    try:
        characters = soup.select('div[class="infoBoxRowItem"] > a[href^="/characters"]')
        bookCharacters = resultset_to_list(characters)
    except:
        print(f' Warning: missing bookCharacters for {book_directory}')
    try:
        setting = soup.select('div[class="infoBoxRowItem"] > a[href^="/places"]')
        bookSetting = resultset_to_list(setting)
    except:
        print(f' Warning: missing fields in {book_directory}')
    
    #storing all parsed info into a dataframe
    l=[[bookSeries,bookAuthors,ratingValue,ratingCount,reviewCount,Plot,NumberofPages,publishingDate,bookCharacters,bookSetting,bookUrl]]
    colnames=['bookSeries','bookAuthors','ratingValue','ratingCount','reviewCount','Plot','NumberofPages','publishingDate','bookCharacters','bookSetting','bookUrl']
    df=pd.DataFrame(l,columns=colnames)
    df.index=[bookTitle]
    
    return df

def generate_tsv(pages_interval):
    
    for page_number in pages_interval: # folder number
        for book_number in range(100): # book number in folder
            i=100*(page_number-1)+book_number
            book_html = f'{i}.html' # absolute book number
            book_directory = f'{os.getcwd()}/page{page_number}/{book_html}' # path to book html
            parsing=parse(book_directory,i) # parsing function
            try:
                if detect(parsing.Plot[0]) == 'en': # checking language
                    parsing.to_csv(f'article_{i}.tsv',sep='\t',index=True) # coverting dataframe to tsv file
            except:
                print(f"Warning: Language detection failed for {book_directory}")

def fix_broken_links():
    f = open('broken_url.txt','r')
    driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
    lines= f.readlines()
    for line in range(0,len(lines),2):
        url=lines[line]
        path=lines[line+1]
        i=path.split('/')[-1].split('.')[0]
        page_number = path.split('/')[-2]
        page = driver.get(url)
        soup = BeautifulSoup(driver.page_source, features='lxml')
        text = soup.prettify()
        f = open(f"{i}.html", "w")
        f.write(text)
        f.close()
        shutil.move(f"{i}.html", f"{page_number}/{i}.html")
    return 1

#------------------------------ 2.1.1 -------------------------------------

def cleaning(field):
    
    # lower case
    field = field.lower()
    field_terms = []
    
    # removing punctuation
    pattern = nltk.RegexpTokenizer(r"\w+")
    tokenized = pattern.tokenize(field)
    
    # set of words which are not stopwords
    stop_words = stopwords.words('english')
    for e in stop_words:
        if e in tokenized:
            tokenized.remove(e)
    
    # stemming
    ps = PorterStemmer() 
    for w in tokenized:
        stemmed = ps.stem(w)
        field_terms.append(stemmed)
    
    # return cleaned terms with their original occurrences
    return field_terms

def build_vocabulary(items):
    
    terms_ids = set() # to save all unique terms
    article_dict = {} # it is created to find the inverted index, will be used with vocabulary_dict

    for article in items: # we iterate all article_i.tsv files
        if article.endswith(".tsv"): # checking it is a tsv file
            df=pd.read_csv(f"art/{article}",sep='\t',index_col=0) # we read it as a pandas dataframe
            try: # we are not sure that all files have a plot
                plot = df.Plot[0]
                t = set(cleaning(plot)) # we extract the plot and clean it
                terms_ids.update(t) # we add the new set of words to the general set
                article_dict[article.split('.')[0]]= t # maps each words of an article to the article
            except:
                print(f'Warning: {article} does not have a plot') 

    vocabulary_dict = {} # key of this dictionary will be integers(term_id) corresponds to unique word
    for term_id in terms_ids: # for each unique word we assign it an integer (its position in the list)
        i=list(terms_ids).index(term_id)
        vocabulary_dict[i] = term_id
    
    with open('vocabulary.txt', 'w') as file:
        file.write(json.dumps(vocabulary_dict))
    file.close()
    
    return vocabulary_dict,article_dict,terms_ids

def build_inverted_index(vocabulary_dict,article_dict):
    
    inverted_index={} #key of the dictionary is term_id, values are the article names
        
    for vocab in vocabulary_dict.keys(): # for each word           
        for article in article_dict.keys(): # for each article
            if vocabulary_dict[vocab] in article_dict[article]: # checks if the article contains the word
                if vocab in inverted_index: # we update key
                    inverted_index[vocab].append(article)
                else: # we create new key
                    inverted_index[vocab] = [article] 

    with open('inverted_index.txt', 'w') as file:
         file.write(json.dumps(inverted_index))
    file.close()
    
    return inverted_index

#------------------------------ 2.1.2 -------------------------------------

def prefiltering(my_input):
    
    common_articles = []
    
    for word in my_input:
        i = list(vocabulary.values()).index(word) # index of word in the vocabolary file (index = position)
        common_articles.append(set(inverted_index[str(i)])) 
    
    common_articles = [e +'.tsv' for e in set.intersection(*common_articles)]
    
    return common_articles

def query(my_input,vocabulary,inverted_index):
    
    my_input = cleaning(my_input)
    # initializing list to save results of queries, output dataframe, frequency
    new_list = [] 
    df = pd.DataFrame([])

    # for each query word there is a place to put the article names that has the term
    common_articles = prefiltering(my_input)
    for doc in common_articles:
        df_doc = pd.read_csv(f"art/{doc}",index_col=0,header=0,sep='\t') # reding article tsv
        df = pd.concat([df,df_doc], axis=0, join='outer') # adding new line

    df = df[['Plot', 'bookUrl']]
    df = df.reset_index()
    df = df.rename(columns={'index':'bookTitle','bookUrl':'Url'})
    
    return df

#------------------------------ 2.2.1 -------------------------------------


def occurrence_vector(article):
    
    df=pd.read_csv(f"art/{article}",sep='\t',index_col=0)
    plot = cleaning(df.Plot[0])
    occurrence = []
    for term in vocabulary.values():
        occurrence.append(plot.count(term))
    return occurrence

def occurrence_matrix(articles):
    
    occ_matrix=[]
    articles_to_be_deleted=[]
    for article in articles:
        try:
            occ_matrix.append(occurrence_vector(article))
        except:
            # we add to this list the articles with special book title name
            # e.g. python function, True, False ... 
            articles_to_be_deleted.append(article)
            
    for article in articles_to_be_deleted:
        del articles[articles.index(article)]
    
    with open('articles_with_plot.txt', 'w') as file:
        file.write(json.dumps(articles))
    file.close()
    
    return occ_matrix, articles

def N_vector(occ_matrix): 
    N=[] 
    art_len = len(occ_matrix)
    vocab_len = len(occ_matrix[0])
    for i in range(vocab_len):
        # value of each term is the number of articles that contain the term
        total=0
        for j in range(art_len):
            if occ_matrix[j][i]>0: # we store a result only if the term appears
                total+=1
        N.append(total)
    try:
        # applying the idf formula for normalization
        N = [math.log(len(occ_matrix)/i) for i in N]
    except:
        print('division by zero error')
    return N

#each inner list is for article, each value of inner list is score for the term in this article
def matrix_with_score(occ_matrix, my_N_vector):
    
    n=len(occ_matrix)
    occ_matrix_with_score=[]
    for i in occ_matrix:
        # we mupltiply occurrence by idf
        occ_matrix_with_score.append([a*b for a,b in zip(i,my_N_vector)])
    
    return occ_matrix_with_score

def d_vector(OMWS):
    
    d=[]
    for doc in OMWS:
        d.append(1/math.sqrt(sum([i**2 for i in doc]))) # applying formula
    
    with open('d_vector.txt', 'w') as file:
        file.write(json.dumps(d))
    file.close()
    
    return d

def build_inverted_index_2(OMWS):    
    
    inverted_index_s={}
    art_len = len(OMWS)
    term_len = len(OMWS[0])
    
    for i in range(term_len):
        # for each term we initalize a new dictionary
        inverted_index_s[i] ={}
        for j in range(art_len):
            if OMWS[j][i]>0:
                # we add a new key and value
                inverted_index_s[i][articles[j]] = OMWS[j][i]
    
    # we save the index
    with open('inverted_index_score.txt', 'w') as file:
        file.write(json.dumps(inverted_index_s))
    file.close()
    
    return inverted_index_s

#------------------------------ 2.2.2 -------------------------------------

def cosine_similarity(article,my_input,d_vector):
     
    query_scores = []

    for term in my_input:
        i = list(vocabulary.values()).index(term) # id of the term
        dict_articles = inverted_index_score[str(i)] # articles associated to that term
        query_scores.append(dict_articles[article]) # scores associated to query terms in a specific article 
    
    # we retrieve the index of article in the d vector and extract the corresponding value
    article_index = articles.index(article)
    # we define all the paramters
    d = d_vector[article_index] 
    q = math.sqrt(len(my_input))
    summation = sum(query_scores)
    
    # we apply the cosine similarity formula and avoid division by zero
    if summation != 0:
        cosine_similarity = (1/q) * d * summation
    else: 
        cosine_similarity = 0 
        
    return cosine_similarity

def heap(df,k):
    df_sorted = pd.DataFrame([])
    # dictionary with similarity score as key and and row index as value
    heap_dict = pd.Series(df.index,df.Similarity).to_dict()
    largest_sorted = heapq.nlargest(k, heap_dict)
    # we add new line according to the similarity score order
    for score in largest_sorted:
        idx = heap_dict[score]
        df_sorted = pd.concat([df_sorted,df.loc[[idx]]], axis=0, join='outer')
    return df_sorted

def top_k_query(my_input,k):
    
    # initializing output dataframe and list fo articles matching the query
    df = pd.DataFrame([])
    my_input = set(cleaning(my_input))
    common_articles= prefiltering(my_input)

    # we iterate through all articles
    for article_idx in range(len(common_articles)):
        article = common_articles[article_idx] # article file name
        df_doc = pd.read_csv(f"art/{article}",index_col=0,header=0,sep='\t') # reading article tsv
        similarity = cosine_similarity(article,my_input,d)
        df_doc['Similarity'] = similarity # adding new column to the article dataframe
        df = pd.concat([df,df_doc], axis=0, join='outer') # adding new line to the output dataframe
    
    try:
        df = df[['Plot', 'bookUrl','Similarity']] 
        df = df.reset_index()
        df = df.rename(columns={'index':'bookTitle','bookUrl':'Url'})
        df = heap(df,k)
    except:
        print ("None of the documents matches the query")
    
    return df

#------------------------------ 3.1 -------------------------------------

def query_newscore(plot_query,title_query,author_query,year_query,min_nratings,rate_query,min_npages,k):
    
    df= pd.DataFrame([])
    common_articles = prefiltering(set(cleaning(plot_query)))
    
    # taking into account plot only of those articles that have all queries in the plot
    for article_idx in range(len(common_articles)):
        
        article = common_articles[article_idx] # naem of the article
        df_art = pd.read_csv('art/'+article,index_col=0,header=0,sep='\t')
        
        # PLOT
        # for this we calculate again cosine similarity
        plot_similarity = cosine_similarity(article,set(cleaning(plot_query)),d)
        
        
        # BOOK TITLE
        try:
            # we set it to zero and change it only when if condition is satisfied
            title_score = 0
            title = df_art.index[0]
            title_query = str(title_query)
            ed_title = editdistance.eval(title.lower(), title_query.lower()) # python editdistance module
            # consider only the given input that has ed less than or equal to length of title
            if ed_title <= len(title):
                title_score = ( len(title) - ed_title ) / len( title ) # normalization
        except:
            title_score = 0
        
        
        # AUTHOR
        # execption needed since not all books have title by construction
        try:
            # we set it to zero and change it only when if condition is satisfied
            author_score = 0
            author = df_art.bookAuthors[0]
            author_query = str(author_query)
            ed_author = editdistance.eval(author.lower(), author_query.lower())
            # consider only the given input that has ed less than or equal to length of author
            if ed_author <= len(author):
                author_score = ( len( author ) - ed_author ) / len( author ) # normalization
        except:
            author_score = 0
        
        # YEAR
        try:
            year = int(df_art.publishingDate[0].split(' ')[-1]) # extracting year only
            year_score = 0
            if 2020-year <= year_query: # checking threshold
                year_score = 1
        except:
            year_score = 0

        
        # NUMBER OF RATINGS
        try:
            rcount = int(df_art.ratingCount[0])
            nratings_score = 0
            if rcount >= min_nratings : # checking threshold
                nratings_score = 1
        except:
            nratings_score = 0

        # AVARAGE RATING
        try:
            rating = float(df_art.ratingValue[0])
            rate_score = 0
            if rating >= rate_query : # checking threshold
                rate_score = 1
        except:
            rate_score = 0
        
        # PAGES
        try:
            pages = int(df_art.NumberofPages[0])
            pages_score = 0
            if pages >= min_npages : # checking threshold
                pages_score = 1
        except:
            pages_score = 0
        
        # FINAL SCORE   
        matches = sum([year_score,nratings_score,rate_score,pages_score])/4 # normalizing match score
        final_score = (plot_similarity + title_score + author_score + matches)/4 # normalizing total score
        df_art['Similarity'] = final_score
        df = pd.concat([df,df_art], axis=0, join='outer') # adding new line to the output dataframe
    
    # otput dataframe
    df = df[['Plot', 'bookUrl','Similarity']] 
    df = df.reset_index()
    df = df.rename(columns={'index':'bookTitle','bookUrl':'Url'})
    df = heap(df,int(k)) # heap structure
    
    return df

#------------------------------ 4 -------------------------------------

def find_series():
    series_set = set() # avoit to consider the same series twice
    i = -1
    while len(series_set) < 10:
        i += 1
        try:
            df_art = pd.read_csv('art/article_'+ str(i) + '.tsv',index_col=0,header=0,sep='\t')
            series = df_art.bookSeries[0][1:-1]  # exlude parenthesys
            if "#" in series: # this symbol is in all valid series
                series_number = series.split('#')[1] # number of the book
                series = series.split('#')[0] # name of the series
                if len(series_number) == 1: # we check it is not more books together
                    series_set.add(series)   
        except:
            continue
    return list(series_set)


def find_all_books(articles):
    series_list= find_series()
    
    # empty dictionary with name of the series as keys and lists as values
    series_dict = {new_list: [] for new_list in series_list}
    
    # we iterate all articles and try to extract their series
    for article in articles:
        df_art = pd.read_csv('art/'+article,index_col=0,header=0,sep='\t')
        try: 
            series_name = df_art.bookSeries[0][1:-1].split('#')[0]
            number = df_art.bookSeries[0][1:-1].split('#')[1]
            n_pages =  df_art.NumberofPages[0]
            year =  df_art.publishingDate[0]
            # if the series is in the list of selected series and it is a single book
            if series_name in series_list and len(number) == 1:
                # we append a tuple with 3 elements for each series
                # the ordinal of the book, the date of pubblication and the number of pages
                idx = series_list.index(series_name)
                series_dict[series_name].append((number,year,n_pages))
        except:
            continue
    return series_dict


def cumulative_pages(articles):
    series_dict = find_all_books(articles)
    
    #empty dictionary that will have series names as keys and a list of cumulative pages as values
    all_cum_pages = {}
    all_years = {}
    for series in series_dict.keys():
        sorted_series = sorted(series_dict[series]) # sort by book number
        cum_pages = [0] # setting first value of number of pages
        years = []
        for book in sorted_series:
            cum_pages.append(cum_pages[-1] + book[2]) # adding new pages to last value
            years.append(book[1])
        all_cum_pages[series]=cum_pages[1:] # assigning value to key, we ignore the first zero
        all_years[series]=years
    return all_cum_pages,all_years


#------------------------------ 5 -------------------------------------


def MaxSubseqRecursive(s, alphabet): 
    
    
    # if we have checked all the elements of at list one string we exit the function
    if len(s) == 0 or len(alphabet) == 0: 
        return 0
    
    # of the last element of the two strings is the same we we add 1 
    # we call the function on a substring which does not include the last element of both 
    elif s[len(s)-1] == alphabet[len(alphabet)-1]: 
        return 1 + MaxSubseqRecursive(s[:-1], alphabet[:-1])
    
    # if the last element is different we do not add 1 and we divide the problem in two parts
    # 1. we call the function on the reduced alphabet string 
    # 2. we call the function on the reduced input string 
    else: 
        return max(MaxSubseqRecursive(s, alphabet[:-1]), MaxSubseqRecursive(s[:-1], alphabet))


def MaxSubseqDynamic(s, alphabet): 
    
    # inizializing empty matrix 
    matrix = []
    for i in range(len(s) + 1):
        matrix.append([0]*(len(alphabet) + 1))
    
    # for each row we have a different character of the string 
    # +1 is needed for the empty string 
    for row in range(len(s)+1): 
        
        # for each column we have a different character of the alphabet string
        # +1 is needed for the empty sring
        for col in range(len(alphabet)+1): 
            
            # if either the column or the row is the empty string we keep the zero
            if row == 0 or col == 0 : 
                continue
            
            # if I have a match I add 1 to the value in the cell diagonal to that of my current position
            # I update my current position 
            elif s[row-1] == alphabet[col-1]: 
                matrix[row][col] = matrix[row-1][col-1] +1
            
            # if I do not have a match I take the maximum value among the neighbors 
            # I update my current position
            else: 
                matrix[row][col] = max(matrix[row-1][col], matrix[row][col-1]) 
  
    return matrix[row][col] 

def run_algorithms(str_len):    
    
    time_interval_1 = []
    results1 = []
    time_interval_2 = []
    results2 = []
    strings = []

    alphabet="ABCDEFGHIGKLMNOPQRSTUVWXYZ"
    
    for n in str_len:
        
        # random string of length n
        s =''.join(random.choice(string.ascii_uppercase) for _ in range(n))
        strings.append(s)
        
        # recursive
        start1 = time.time()
        results1.append(MaxSubseqRecursive(s,alphabet))
        end1 = time.time()
        time_interval_1.append(end1-start1)
        
        
        # dynamic
        start2 = time.time()
        results2.append(MaxSubseqDynamic(s,alphabet))
        end2 = time.time()
        time_interval_2.append(end2-start2)

    
    return time_interval_1, time_interval_2 ,results1, results2, strings

def run_dynamic(str_len):    
    
    time_interval_2 = []
    results2 = []

    alphabet="ABCDEFGHIGKLMNOPQRSTUVWXYZ"
    
    for n in str_len:
        
        
        s =''.join(random.choice(string.ascii_uppercase) for _ in range(n))
        strings.append(s)
        start2 = time.time()
        results2.append(MaxSubseqDynamic(s,alphabet))
        end2 = time.time()
        time_interval_2.append(end2-start2)

    
    return time_interval_2 ,results2

