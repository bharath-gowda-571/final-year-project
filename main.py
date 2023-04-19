from flask import Flask, render_template, request, redirect, url_for
# import datetime
# from newsapi import NewsApiClient
# import redis
# from mongo_utils import get_database
from json import loads
# from bson.json_util import dumps
# from bson import json_util
# from fastcoref import spacy_component
# import spacy
import json
# import pickle
# import torch
# from transformers import BertForSequenceClassification, BertTokenizer
# from nltk import tokenize
# import numpy as np
# import yfinance as yf
# from datetime import datetime,timedelta
# import pandas as pd
app = Flask(__name__)
app.debug=True

# Redis
# redisConnector=redis.Redis(host="localhost",password="",decode_responses=True)
# Reading company search terms
companies=open("companies.txt").readlines()
company_names={}
for company in companies:
    names=company.strip().split(",")
    company_names[names[0]]=names

# device=torch.device('cuda')
# model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert').to(device)
# tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert',device=device)
# model_2_class=pickle.load(open("models/random_search_v1",'rb'))
# model_4_class=pickle.load(open("models/random_search_v1_4",'rb'))
# Fastcoref 
# nlp=spacy.load("en_core_web_sm")
# nlp.add_pipe(
#     "fastcoref",
#     config={'model_architecture':'FCoref','device':"cuda:0"}
# )
def search_in_sentence(words, sentence):
    return any(word.lower() in sentence.lower() for word in words)

def get_sentiment(sentence):
    classes=np.array([1,0,-1])
    inputs = tokenizer(sentence, padding = True, truncation = True, return_tensors='pt').to(device)
    outputs = model(**inputs)
    prediction = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()
    return classes[np.argmax(prediction)]
def get_sentiment_class(score):
    if 0<score<=0.05:
        return "Neutral",0
    elif -0.05<score<=0:
        return "Neutral",0
    elif 0.05<score<=0.2:
        return "Slightly Positive",1
        
    elif -0.2<score<=-0.05:
        return "Slightly Negative",-1
        
    elif score>0.2:
        return "Positive",2

    elif score<=-0.2:
        return "Negative",-2


def calculate_indicators(price_data):
    price_data.sort_values(by = ['symbol','datetime'], inplace = True)

    price_data['change_in_price'] = price_data['close'].diff()
    mask = price_data['symbol'] != price_data['symbol'].shift(1)

    price_data['change_in_price'] = np.where(mask == True, np.nan, price_data['change_in_price'])

    price_data[price_data.isna().any(axis = 1)]
    n = 14

    up_df, down_df = price_data[['symbol','change_in_price']].copy(), price_data[['symbol','change_in_price']].copy()

    up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0

    down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0

    down_df['change_in_price'] = down_df['change_in_price'].abs()

    ewma_up = up_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())
    ewma_down = down_df.groupby('symbol')['change_in_price'].transform(lambda x: x.ewm(span = n).mean())

    relative_strength = ewma_up / ewma_down

    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

    price_data['down_days'] = down_df['change_in_price']
    price_data['up_days'] = up_df['change_in_price']
    price_data['RSI'] = relative_strength_index
    n = 14

    low_14, high_14 = price_data[['symbol','low']].copy(), price_data[['symbol','high']].copy()

    low_14 = low_14.groupby('symbol')['low'].transform(lambda x: x.rolling(window = n).min())
    high_14 = high_14.groupby('symbol')['high'].transform(lambda x: x.rolling(window = n).max())

    k_percent = 100 * ((price_data['close'] - low_14) / (high_14 - low_14))

    price_data['low_14'] = low_14
    price_data['high_14'] = high_14
    price_data['k_percent'] = k_percent
    n = 14

    low_14, high_14 = price_data[['symbol','low']].copy(), price_data[['symbol','high']].copy()

    low_14 = low_14.groupby('symbol')['low'].transform(lambda x: x.rolling(window = n).min())
    high_14 = high_14.groupby('symbol')['high'].transform(lambda x: x.rolling(window = n).max())

    r_percent = ((high_14 - price_data['close']) / (high_14 - low_14)) * - 100

    price_data['r_percent'] = r_percent
    ema_26 = price_data.groupby('symbol')['close'].transform(lambda x: x.ewm(span = 26).mean())
    ema_12 = price_data.groupby('symbol')['close'].transform(lambda x: x.ewm(span = 12).mean())
    macd = ema_12 - ema_26

    # Calculate the EMA
    ema_9_macd = macd.ewm(span = 9).mean()

    # Store the data in the data frame.
    price_data['MACD'] = macd
    price_data['MACD_EMA'] = ema_9_macd
    n = 9

    price_data['Price_Rate_Of_Change'] = price_data.groupby('symbol')['close'].transform(lambda x: x.pct_change(periods = n))
    
    def obv(group):
        # Grab the volume and close column.
        volume = group['volume']
        change = group['close'].diff()

        # intialize the previous OBV
        prev_obv = 0
        obv_values = []

        # calculate the On Balance Volume
        for i, j in zip(change, volume):

            if i > 0:
                current_obv = prev_obv + j
            elif i < 0:
                current_obv = prev_obv - j
            else:
                current_obv = prev_obv

            # OBV.append(current_OBV)
            prev_obv = current_obv
            obv_values.append(current_obv)
        
        # Return a panda series.
        return pd.DataFrame({'On Balance Volume': obv_values}, index=group.index)
    
    obv_groups = price_data.groupby('symbol').apply(obv)
    price_data['On Balance Volume'] = obv_groups.reset_index(level=0, drop=True)

@app.route('/')
def home():
    return render_template('home.html', site_name='My Website')

@app.route('/search', methods=['GET', 'POST'])
def search():
    
    if request.method == 'POST':
        selected_company = request.form['company']
        return redirect(url_for('result', company=company_names[selected_company]))
    
    return render_template('search.html', site_name='My Website', company_names=list(company_names.keys()))

@app.route('/result')
def result():
    passed_company = request.args.get('company')
    alt_names=company_names[passed_company]
    symbol=alt_names[-1]
    alt_names=[name.strip() for name in alt_names]
    query_string="|".join(alt_names)
    query_string=query_string.replace(" ","\s")
    query_string=query_string.replace("&","\&")
    query_string="(\s|\.|\,|^)("+query_string+")(\s|\.|\,|$)"
    cache_key=passed_company.replace(" ","_")
    print(query_string)
    # db=get_database("articles")
    # stored_dic=redisConnector.get(cache_key)
    stored_dic=None
    with open('final_dictionary.json', 'r') as f:
            stored_dic = json.load(f)

    if stored_dic!=None:
        print("from cache")
        # stored_dic= loads(stored_dic)
        technical_indicators=stored_dic["indicators"]
        sorted_dates=stored_dic["dates"]
        dic=stored_dic['articles']
#     else:
#         dic={}
#         print("Not from cache")
        
# # Print the loaded data
#         results=db["articles"].aggregate([
#         {
#             '$match': {
#                 '$or': [
#                     {
#                         'title': {
#                             '$regex': query_string, 
#                             '$options': 'i'
#                         }
#                     }, {
#                         'content': {
#                             '$regex': query_string, 
#                             '$options': 'i'
#                         }
#                     }
#                 ]
#             }
#         }, {
#             '$project': {
#                 'datetime': 1, 
#                 'date': {
#                     '$dateToString': {
#                         'format': '%Y-%m-%d', 
#                         'date': {
#                             '$dateFromString': {
#                                 'dateString': '$datetime'
#                             }
#                         }
#                     }
#                 }, 
#                 'content': 1, 
#                 'url': 1, 
#                 'title': 1
#             }
#         }, {
#             '$group': {
#                 '_id': '$date', 
#                 'documents': {
#                     '$push': '$$ROOT'
#                 }, 
#                 'count': {
#                     '$sum': 1
#                 }
#             }
#         }, {
#             '$sort': {
#                 '_id': -1
#             }
#         }, {
#             '$limit': 7
#         }
#         ])

#         sentiment_dic={}
#         for doc in results:
#             print(doc)
#             dic.setdefault(doc['_id'],[])
#             dic[doc['_id']]=doc['documents']
#             sentiment_dic.setdefault(doc['_id'],{"sentiment":0,"count":0})
#             for document in dic[doc['_id']]:
#                 docs=nlp.pipe(
#                     [document['content']],
#                     component_cfg={"fastcoref": {'resolve_text': True}}
#                 )
#                 document['resolved_text']=list(docs)[0]._.resolved_text
#                 doesTitleContainCompanyName=False
                
#                 title_sentiment=0
#                 sentiments=0
#                 count=0
#                 if search_in_sentence(alt_names,document['title']):
#                     title_sentiment=get_sentiment(document['title'])
#                     doesTitleContainCompanyName=True
#                 for sentence in tokenize.sent_tokenize(document['resolved_text']):
#                     if doesTitleContainCompanyName:
#                         sentiments+=get_sentiment(sentence)
#                         count+=1
#                     else:
#                         if search_in_sentence(alt_names,sentence):
#                             sentiments+=get_sentiment(sentence)
#                             count+=1
#                 if doesTitleContainCompanyName:
#                     final_sentiment=(sentiments+2*title_sentiment)/(count+1)
#                 else:
#                     if count==0:
#                         final_sentiment=0
#                     else:
#                         final_sentiment=sentiments/count
#                 document["sentiment_score"]=round(final_sentiment,2)
#                 document['sentiment_class'],document['sentiment_class_value']=get_sentiment_class(document['sentiment_score'])
#                 sentiment_dic[doc['_id']]['sentiment']+=document['sentiment_class_value']
#                 sentiment_dic[doc['_id']]['count']+=1
            
#         sorted_dates=sorted(list(dic.keys()))
#         print(sorted_dates)
#         start_date=(datetime.strptime(sorted_dates[0],"%Y-%m-%d")-timedelta(days=60)).strftime("%Y-%m-%d")
#         end_date=(datetime.strptime(sorted_dates[-1],"%Y-%m-%d")+timedelta(days=10)).strftime("%Y-%m-%d")
#         ticker=alt_names[-1]
#         company_ticker=yf.Ticker(ticker+".NS")
#         df=None
#         df=company_ticker.history(start=start_date,end=end_date)
#         df=df.reset_index()
#         df=df.rename(columns={"Date":'datetime','Open':"open",'High':'high','Low':'low','Close':'close','Volume':'volume',})
#         df['symbol']=ticker
#         df=df[['symbol','datetime','close','high','low','open','volume']]
#         df['datetime'] = pd.to_datetime(df['datetime'])

# # extract the date and replace it in the same column
#         df['datetime'] = df['datetime'].dt.date

#         calculate_indicators(df)
#         technical_indicators={}
#         # Assuming the dataframe is called df and the date column is called "Date"
#         for date in sorted_dates:
#             search_date = pd.to_datetime(date)

#             # Find the row with the matching date
#             row = df.loc[df['datetime'] == search_date]

#             # If the row is empty, find the closest date that is less than the search date
#             if row.empty:
#                 closest_date = df.loc[df['datetime'] < search_date]['datetime'].max()
#                 row = df.loc[df['datetime'] == closest_date]

#             # Convert the row to a dictionary
#             row_dict = row.to_dict('records')[0]
#             technical_indicators[date]=row_dict
#             if sentiment_dic[date]['count']==0:
#                 technical_indicators[date]['sentiment']=0
#             else:
#                 technical_indicators[date]['sentiment']=round(sentiment_dic[date]['sentiment']/sentiment_dic[date]['count'],2)
#             input=[technical_indicators[date]['RSI'],technical_indicators[date]['k_percent'],technical_indicators[date]['r_percent'],technical_indicators[date]['Price_Rate_Of_Change'],technical_indicators[date]['MACD'],technical_indicators[date]['On Balance Volume'],technical_indicators[date]['sentiment']]
#             print(input)
#             technical_indicators[date]['2_class_prediction']=model_2_class.predict([input])[0]
#             technical_indicators[date]['4_class_prediction']=model_4_class.predict([input])[0]

#         storing_dic={"indicators":technical_indicators,"dates":sorted_dates,"articles":dic}
#         redisConnector.setex(cache_key,21600,dumps(storing_dic,default=str))
#         json_str = dumps(storing_dic,default=str)

# # # Write the JSON string to a file
#         with open('final_dictionary.json', 'w') as f:
#             f.write(json_str)
    return render_template('result.html', grouped_articles=dic,dates=sorted_dates,indicators=technical_indicators,company_name=passed_company,symbol=symbol)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=True)
