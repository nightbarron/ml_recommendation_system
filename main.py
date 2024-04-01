import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy
from gensim import corpora, models, similarities
from surprise import dump
import random
import xuly_tiengviet as xt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

# Load model and data
def load_model():
    # Read gemsim model
    dictionary = corpora.Dictionary.load('content_based_model/gemsim_dictionary.dict')
    tfidf = models.TfidfModel.load('content_based_model/gemsim_tfidf.model')
    index = similarities.SparseMatrixSimilarity.load('content_based_model/gemsim_index.index')

    # Concatenate all files to surprise model
    filePath = 'collaborative_filtering/surprise_model'
    with open(filePath, 'wb') as file:
        for i in range(60):
            with open(f'{filePath}_{i}', 'rb') as f:
                file.write(f.read())
    
    # Read surprise model
    predictions, algo = dump.load('collaborative_filtering/surprise_model')
    
    # Read data
    data1 = pd.read_csv('data/Products_ThoiTrangNam_preprocessing1.csv')
    data2 = pd.read_csv('data/Products_ThoiTrangNam_preprocessing2.csv')
    products = pd.concat([data1, data2])
    ratings = pd.read_csv('data/Products_ThoiTrangNam_rating_preprocessing.csv')

    # Fill null
    products['description'] = products['description'].fillna('')
    products['description_tokenize'] = products['description_tokenize'].fillna('')
    products['sub_category_tokenize'] = products['sub_category_tokenize'].fillna('')

    return dictionary, tfidf, index, predictions, algo, products, ratings

def DrawWordCloud(predictProducts):
    # Get top 50 words of top 3 similar products
    words = []
    for p in predictProducts:
        text = p['product_name_tokenize'] + ' ' + p['description_tokenize'] + ' ' + p['sub_category_tokenize']
        text = text.values[0]
        words += text.split()

    # Count the frequency of words
    word_freq = Counter(words)
    word_freq.most_common(50)

    text = ''
    for word in word_freq.most_common(50):
        text += word[0] + ' '

    # Create wordcloud
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = None,
                    min_font_size = 10).generate(text)

    # plot the WordCloud image
    fig = plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    st.pyplot(fig)

def gioi_thieu():
    st.markdown('## Giới thiệu Project')
    st.markdown("""### Mục tiêu:
- Xây dựng hệ thống gợi ý sản phẩm dựa trên nội dung và khách hàng
- Dự đoán Content-based và Collaborative Filtering
            
### Công nghệ sử dụng:
- Content-based Filtering: Gensim
- Collaborative Filtering: Surprise
        
### Quy trình xây dựng mô hình:
- Thu thập dữ liệu
- Tiền xử lý dữ liệu tiếng việt
- Xây dựng mô hình
- Đánh giá mô hình => Chọn mô hình tốt nhất
- Lưu mô hình
- Thống kê dữ liệu
- Xây dựng giao diện và dự đoán
        
### Tác giả:
- Huỳnh Thái Bảo
- Đặng Lê Hoàng Tuấn""")
    return

def insight(products, ratings):
    st.markdown('## INSIGHT tập dữ liệu')
    st.markdown('### 1. Dữ liệu sản phẩm')
    st.write(products.head())

    st.write('- Biểu đồ số lượng sản phẩm theo danh mục')
    fig = plt.figure(figsize=(10, 5))
    product_sub_cateogry = products.groupby('sub_category').agg({'product_id': 'count', 'rating': 'mean', 'price': 'mean'}).reset_index()
    product_sub_cateogry = product_sub_cateogry.rename(columns = {'sub_category' : 'Sub Category', 'product_id' : 'Count Sub', 'rating' : 'Rating', 'price' : 'Price'})
    product_sub_cateogry.sort_values(by = 'Count Sub', ascending = False, inplace = True)
    product_sub_cateogry[['Rating', 'Price']] = round(product_sub_cateogry[['Rating', 'Price']],1)
    sns.barplot(data = product_sub_cateogry, x = 'Sub Category', y = 'Count Sub', color='skyblue')
    plt.title('Count for Sub_Category',loc = 'left',  fontweight = 'heavy', fontsize = 16)
    plt.grid(axis = 'y', linestyle ='--')
    sns.despine(left=True, bottom=True)
    plt.xticks(rotation = 70)
    plt.tight_layout()
    st.pyplot(fig)

    st.write('- Biểu đồ giá trung bình theo danh mục')
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(data = product_sub_cateogry.sort_values(by = 'Price', ascending = False), x = 'Sub Category', y = 'Price')
    plt.title('Avg price for Sub_Category',loc = 'left',  fontweight = 'heavy', fontsize = 16)
    plt.grid(axis = 'y', linestyle ='--')
    sns.despine(left=True, bottom=True)
    plt.xticks(rotation = 70)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""`Comment:`  
- Ta có tất cả 48.700 sản phẩm thời trang nam khác nhau thuộc 17 phân loại sản phẩm đang được bán trên shopee.  
- Tuy nhiên, chỉ có 6 phân loại có đa dạng sản phẩm nhất đó là: Đồ Bộ, Trang Phục Truyền Thống, Vớ/Tất, Đồ Hóa Trang, Cà Vạt & Nơ Cổ, Kính Mát Nam với số lượng sản phẩm của mỗi phân loại đều trên 4.000.  
- Trong khi đó các phân loại còn lại với số lượng sản phẩm khá khiêm tốn so với nhóm trên (chỉ từ 1.600 sản phẩm mỗi loại).  
- Ở phân tích xu hướng mua của khách hàng trên shopee, chúng ta có 5 phân loại đang được mua nhiều nhất là: Đồ Bộ, Vớ/Tất, Áo, Áo Khoác, Quần Dài Âu. 
- Mặc dù vậy chỉ có Đồ Bộ, Vớ/Tất là có nhiều sản phẩm để đáp ứng nhu cầu của khách hàng, từ dữ liệu như trên, các nhà bán hàng ở shopee có thể đẩy mạnh hơn vào các sản phẩm thuộc nhóm Áo, Áo Khoác, Quần Dài Âu nhằm đa dạng nhóm sản phẩm đang có lượt mua cao từ khách hàng nhưng không có quá nhiều sản phẩm để giúp họ đa dạng hóa sự lựa chọn.""")

    st.markdown('### 2. Dữ liệu đánh giá')
    st.write(ratings.head())

    st.write('- Biểu đồ phân phối đánh giá')
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(data = ratings, x = 'rating', palette = 'viridis')
    plt.title('Rating Distribution',loc = 'left',  fontweight = 'heavy', fontsize = 16)
    plt.grid(axis = 'y', linestyle ='--')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    st.pyplot(fig)

    st.write('- Top 10 user mua nhiều nhất')
    top10_user = ratings['user'].value_counts().head(10)
    st.write(top10_user)

    st.write('- Top 10 sản phẩm được mua nhiều nhất')
    top10_product = ratings['product_id'].value_counts().head(10)
    top10_product = pd.merge(top10_product, products[['product_id', 'product_name']], on = 'product_id', how = 'left')
    st.write(top10_product)

    st.write('- Top 10 sản phẩm được đánh giá cao nhất (trung bình đánh giá)')
    top10_product_rating = ratings.groupby('product_id').agg({'rating': 'mean'}).sort_values(by = 'rating', ascending = False).head(10)
    top10_product_rating = pd.merge(top10_product_rating, products[['product_id', 'product_name']], on = 'product_id', how = 'left')
    st.write(top10_product_rating)

    st.write('- Top 10 sản phẩm được đánh giá thấp nhất (trung bình đánh giá)')
    top10_product_rating = ratings.groupby('product_id').agg({'rating': 'mean'}).sort_values(by = 'rating', ascending = True).head(10)
    top10_product_rating = pd.merge(top10_product_rating, products[['product_id', 'product_name']], on = 'product_id', how = 'left')
    st.write(top10_product_rating)

    st.write('- Số lượng đánh giá theo sub_category')
    plt.clf()
    rating_sub_cateogry = ratings.merge(products[['product_id', 'sub_category']], on = 'product_id', how = 'left')
    rating_sub_cateogry = rating_sub_cateogry.groupby('sub_category').agg({'rating': 'count'}).reset_index()
    rating_sub_cateogry = rating_sub_cateogry.rename(columns = {'rating' : 'Count Rating'})
    rating_sub_cateogry.sort_values(by = 'Count Rating', ascending = False, inplace = True)
    sns.barplot(data = rating_sub_cateogry, x = 'sub_category', y = 'Count Rating')
    plt.title('Count for Sub_Category',loc = 'left',  fontweight = 'heavy', fontsize = 16)
    plt.grid(axis = 'y', linestyle ='--')
    sns.despine(left=True, bottom=True)
    plt.xticks(rotation = 70)
    plt.tight_layout()
    st.pyplot(fig)

    st.write('- Trung bình đánh giá theo sub_category')
    rating_sub_cateogry = ratings.merge(products[['product_id', 'sub_category']], on = 'product_id', how = 'left')
    rating_sub_cateogry = rating_sub_cateogry.groupby('sub_category').agg({'rating': 'mean'}).reset_index()
    rating_sub_cateogry = rating_sub_cateogry.rename(columns = {'rating' : 'Rating'})
    rating_sub_cateogry.sort_values(by = 'Rating', ascending = False, inplace = True)
    sns.barplot(data = rating_sub_cateogry, x = 'sub_category', y = 'Rating')
    plt.title('Avg Rating for Sub_Category',loc = 'left',  fontweight = 'heavy', fontsize = 16)
    plt.grid(axis = 'y', linestyle ='--')
    sns.despine(left=True, bottom=True)
    plt.xticks(rotation = 70)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""`Comment:`  
- Có 994.487 sản phẩm về Thời Trang Nam đang được mua và đánh giá trong tập dữ liệu được cung cấp.  
- Xu hướng tiêu dùng thời trang nam tập trung chủ yếu ở 3 nhóm sản phẩm: Đồ bộ, vớ/tất, áo, áo khoác và quần dài Âu.  
- Mặc cho xu hướng mua hàng ở 4 nhóm trên đang chiếm đa số nhưng chỉ có Vớ/Tất tạo được mức độ hài lòng của khách hàng ở mức cao trong khi đó Quần Dài Âu, Áo Khoác và Áo lại đứng cuối cùng về mức độ hài lòng.  
- Ngược lại, tuy Cà Vạt & Nơ Cổ thuộc một trong những nhóm rất kén lượt mua nhưng mang lại trãi nghiệm vô cùng tốt từ khách hàng, có thể nhận định rằng đây là nhóm hàng tuy kén người dùng nhưng chất lượng tương đối cao.  
- Các nhóm sản phẩm có giá tiền top đầu đồng thời rất ít lượt mua gồm có 3 nhóm: Đồ hóa trang, áo vest và blazer, đồ ngủ. Trong khi Đồ Hóa Trang và Đồ Ngủ là những nhóm hầu như không xuất hiện trong tủ đồ của nam giới thì Áo Vest và Blazer lại thuộc nhóm đồ có xu hướng mua tại cửa hàng nhiều hơn là đặt online (vì để sở hữu một bộ Vest đẹp và chất lượng lại đi kèm với giá mua tương đối cao nên việc mua online mang đến nhiều rủi ro hơn).  
=> Có thể xác định xu hướng Thời Trang Nam của người tiêu dùng vẫn phụ thuộc vào nhu cầu của từng cá nhân, họ không mua vì giá rẻ hay mua vì sản phẩm có lượt rating cao. Họ cần một sản phẩm phù hợp với mình hơn.
""")

    return

def SuggestProductBasedProductID(user_clicked_product_id, products, index, tfidf, dictionary, num_of_suggest=3):
    # Get index of user_clicked_product_id
    index_user_clicked_product_id = products[products['product_id'] == user_clicked_product_id].index[0]
    product = products.iloc[index_user_clicked_product_id]
    # Convert description to vector
    text = product['product_name_tokenize'] + ' ' + product['description_tokenize'] + ' ' + product['sub_category_tokenize']
    text_gem = text.split()

    # Find top 3 similar products
    sims = index[tfidf[dictionary.doc2bow(text_gem)]]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    result = []

    for i in range(1, num_of_suggest+1):
        # Convert index to product_id
        product_id = products.iloc[sims[i][0]]['product_id']
        result.append(products[products['product_id'] == product_id])

    return result

def SuggestProductBasedUserContent(user_query, products, index, tfidf, dictionary, num_of_suggest=3):
    # Convert user query to vector
    user_query = xt.stepByStep(user_query)
    user_query_gem = user_query.split()
    user_query_bow = dictionary.doc2bow(user_query_gem)
    user_query_tfidf = tfidf[user_query_bow]

    # Find top 3 similar products
    sims = index[user_query_tfidf]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    result = []

    for i in range(1, num_of_suggest+1):
        # Convert index to product_id
        product_id = products.iloc[sims[i][0]]['product_id']
        result.append(products[products['product_id'] == product_id])

    return result

def SuggestProductBasedUserRating(user_id, ratings, products, predictions, algo, num_of_suggest=3):
    df_select = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >=3)]
    df_select = df_select.set_index('product_id')

    df_score = ratings[["product_id"]]
    df_score['EstimateScore'] = df_score['product_id'].apply(lambda x: algo.predict(user_id, x).est) # est: get EstimateScore
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
    df_score = df_score.drop_duplicates()

    # Get top x products
    result = []
    df_topx = df_score[df_score.EstimateScore>=3].head(num_of_suggest)

    for product_id in df_topx['product_id']:
        result.append(products[products['product_id'] == product_id])

    return result

def content_based_filtering(dictionary, tfidf, index, products):
    st.markdown('## Content-based Filtering')
    st.markdown("""### Mô tả:
- Content-based Filtering dựa trên nội dung của sản phẩm để đưa ra gợi ý sản phẩm tương tự.
- Mô hình sử dụng Gensim để xây dựng mô hình.
- Mô hình sẽ đưa ra 3 sản phẩm tương tự với sản phẩm được chọn.
""")
    st.write('### 1. Dựa vào sản phẩm đã click:')

    # Make this like search bar
    product_id_list = products['product_id'].values
    user_clicked_product_id = st.selectbox('Chọn một sản phẩm', product_id_list)
    st.write('Thông tin sản phẩm đã chọn:')
    selected_product = products[products['product_id'] == user_clicked_product_id]
    selected_product = selected_product[['product_name', 'description', 'link', 'sub_category', 'price']]
    st.write(selected_product)

    if user_clicked_product_id:
        try:
            # user_clicked_product_id = int(user_clicked_product_id)
            predictProducts = SuggestProductBasedProductID(user_clicked_product_id, products, index, tfidf, dictionary)
            filterPredictProducts = []
            for p in predictProducts:
                filterPredictProducts.append(p[['product_name', 'description', 'link', 'sub_category', 'price']])
            st.write("Các sản phẩm dự đoán: ")
            # Convert to DataFrame
            predictProducts_DF = pd.concat(filterPredictProducts)
            st.write(predictProducts_DF)
            st.write("WordCloud của các sản phẩm dự đoán: ")
            DrawWordCloud(predictProducts)
        except Exception as e:
            st.write(e)
            st.write('Product_id không hợp lệ')
    # Gợi ý theo user nhập
            
    st.write('### 2. Dựa vào từ khóa:')
    user_query = st.text_input('Nhập từ khóa tìm kiếm: ')
    if user_query:
        try:
            predictProducts = SuggestProductBasedUserContent(user_query, products, index, tfidf, dictionary)
            filterPredictProducts = []
            for p in predictProducts:
                filterPredictProducts.append(p[['product_name', 'description', 'link', 'sub_category', 'price']])
            st.write("Các sản phẩm tìm kiếm: ")
            # Convert to DataFrame
            predictProducts_DF = pd.concat(filterPredictProducts)
            st.write(predictProducts_DF)
            st.write("WordCloud của các sản phẩm tìm kiếm: ")
            DrawWordCloud(predictProducts)
        except Exception as e:
            st.write(e)
            st.write('Từ khóa không hợp lệ')
    return

def collaborative_filtering(predictions, algo, products, ratings):
    st.markdown('## Collaborative Filtering')
    st.markdown("""### Mô tả:
- Collaborative Filtering dựa trên dữ liệu đánh giá của người dùng để đưa ra gợi ý sản phẩm.
- Mô hình sử dụng Surprise để xây dựng mô hình.
- Mô hình sẽ đưa ra 3 sản phẩm tương tự với sản phẩm được chọn.
""")
    st.write('### Hãy login để xem gợi ý sản phẩm dựa trên dữ liệu đánh giá của bạn')
    user_lst = ratings['user'].unique()
    random_user_lst = random.sample(list(user_lst), 50)
    user_query = st.text_input('Tìm kiếm user_id: ')

    if user_query:
        # Tìm top 5 user gần giống sử dụng string matching
        try:
            result_lst = []
            for user in user_lst:
                if str(user_query) in str(user):
                    result_lst.append(user)
                if len(result_lst) == 5:
                    break
            st.write('Các users tìm thấy: ', result_lst)
            st.write('[*] Sao chép user_id để điền vào ô login bên dưới')
        except Exception as e:
            st.write(e)
            st.write('Không tìm thấy user')


    user = st.text_input('Login bằng user_id: ')

    if user:
        st.write('User Loged in: ', user)
        try:
            user_id = ratings[ratings['user'] == user].iloc[0]['user_id']
            predictProducts = SuggestProductBasedUserRating(user_id, ratings, products, predictions, algo)
            st.write("Các sản phẩm dự đoán: ")
            # Convert to DataFrame
            filterPredictProducts = []
            for p in predictProducts:
                filterPredictProducts.append(p[['product_name', 'description', 'link', 'sub_category', 'price']])
            predictProducts_DF = pd.concat(filterPredictProducts)
            st.write(predictProducts_DF)
            st.write("WordCloud của các sản phẩm dự đoán: ")
            DrawWordCloud(predictProducts)
        except Exception as e:
            st.write(e)
            st.write('User không hợp lệ')

    return




#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def main():
    # Load model
    dictionary, tfidf, index, predictions, algo, products, ratings = load_model()
    products['product_id'] = products['product_id'].astype(str)
    ratings['product_id'] = ratings['product_id'].astype(str)
    ratings['user_id'] = ratings['user_id'].astype(str)
    ratings['user'] = ratings['user'].astype(str)

    # Title
    st.title('Project 2: Recommendation System')
    # st.markdown('## Content-based and Collaborative Filtering')
    menu = ["Giới Thiệu Project", "INSIGHT tập dữ liệu", "Content-based Filtering", "Collaborative Filtering"]
    choice = st.sidebar.selectbox('Danh mục', menu)
    if choice == menu[0]:
        gioi_thieu()
    elif choice == menu[1]:
        insight(products, ratings)
    elif choice == menu[2]:
        content_based_filtering(dictionary, tfidf, index, products)
    elif choice == menu[3]:
        collaborative_filtering(predictions, algo, products, ratings)

    return


if __name__ == "__main__":
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.environ['PYSPARK_LOG_LEVEL'] = 'ERROR'
    main()