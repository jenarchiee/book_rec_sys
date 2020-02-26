import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

## Content based method

books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books = books.head(10000).rename(columns={"Book-Title": "BookTitle", "Book-Author": "BookAuthor"})

books['index'] = books.index

# Choose key feature as content based information of the book
content_based = ['BookTitle', 'BookAuthor']

# Create a function for combining the values of these columns into a single string 
def combine_content(row):
    return row['BookTitle']+" "+row['BookAuthor']

# Clean data, fill all the NaN values with blank string
for content_based in content_based:
    books[content_based] = books[content_based].fillna('')

# Apply content_based() over each rows and store the combined string in "content_based" column   
books["content_based"] = books.apply(combine_content,axis=1)

# Feed combined strings(book content) to CountVectorizer() object
count_matrix = CountVectorizer().fit_transform(books["content_based"])

# obtain the cosine similarity matrix from the count matrix
cosine_sim = cosine_similarity(count_matrix)

#  Define functions to get book title from book index and viceversa
def get_title_from_index(index):
    return books[books.index == index]["BookTitle"].values[0]
def get_index_from_title(title):
    return books[books.BookTitle == title]["index"].values[0]

book_user_likes = "Classical Mythology"

book_index = get_index_from_title(book_user_likes)

# Access the row corresponding to given book to find all the similarity scores for that book 
# Enumerate over it
recommended_books = list(enumerate(cosine_sim[book_index]))

# Sort the list recommended_books according to similarity scores in descending order
# Discard the first element as the book itself after sorting 
sorted_books = sorted(recommended_books,key=lambda x:x[1],reverse=True)[1:]

# run a loop to print first 5 from sorted_books list 
i=0
print("Top 5 similar books to "+book_user_likes+" are:\n")
for element in sorted_books:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>5:
        break





