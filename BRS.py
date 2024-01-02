import pandas as pd
import numpy as np
ratings=pd.read_csv(r"C:\Users\YASH\Desktop\pandas\Ratings.csv")
users=pd.read_csv(r"C:\Users\YASH\Desktop\pandas\Users.csv")

gdown.download('https://drive.google.com/file/d/12o0lVTcez156b_fyKAn_Bjf4CsxyFhp9/view?usp=drive_link', 'Books.csv', quiet=False)
books = pd.read_csv('Books.csv')



books.isnull().sum()
ratings.isnull().sum()
users.isnull().sum()

books.duplicated().sum()
ratings.duplicated().sum()
users.duplicated().sum()

ratings_with_name=ratings.merge(books,on='ISBN')
num_rating=ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()

avg_rating=ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating.rename(columns={'Book-Rating':'avgrating'},inplace=True)

popular_df=num_rating.merge(avg_rating,on='Book-Title')
popular_df=popular_df[popular_df['Book-Rating']>=250].sort_values('avgrating',ascending=False).head(50)
popular_df=popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','Book-Rating','avgrating']]

x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index

filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]
y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt.fillna(0,inplace=True)

from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(pt)

def recommend(book_name):
    index = np.where(pt.index.str.casefold() == book_name.casefold())[0]
    
    if len(index) == 0:
        return []  
    
    index = index[0]
    similar_items = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    return data



import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config( layout="wide")


st.markdown("<h1 style='text-align: center;color: green;'>Book Recommender System</h1>", unsafe_allow_html=True)



selected_option=option_menu(None, ["Home", "Recommend"],  
    menu_icon="cast", default_index=0, orientation="horizontal")

   



if selected_option == "Home":
    st.title("Top 50 Books")  
    cols = st.columns(3)
    for index, row in popular_df.iterrows():
            cols[index % 3].image(row['Image-URL-M'],width=100, use_column_width=False)
            cols[index % 3].write(row['Book-Title'])
              
    

elif selected_option == "Recommend":
    st.title('Recommend Books')
    book_titles = books['Book-Title'].tolist()
    user_input = st.text_input("Enter a book name:")
    button_clicked=st.button("Submit")
    if button_clicked:
        if user_input:
                recommended_books = recommend(user_input)

                if recommended_books:
                    st.write("Recommended Books:")
                    for book in recommended_books:
                        st.write(f"Title: {book[0]}, Author: {book[1]}")
                        st.image(book[2], caption=book[0], use_column_width=False,width=100)
        
                else:
                    st.write("No recommendations found.")
 





