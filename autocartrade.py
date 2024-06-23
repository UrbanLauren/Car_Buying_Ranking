from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import pandas as pd
import praw
import numpy as np
import re
import requests
import tensorflow
from transformers import pipeline, AutoTokenizer
from transformers import DistilBertTokenizer
import torch
import sys
sys.setrecursionlimit(1500)

##  Credit: Code and functions written by Lauren Urban

# VARIABLES
API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
headers = {"Authorization": "Bearer hf_************"}
pipe = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
# Use a pipeline as a high-level helper
sentiment_pipeline = pipeline("sentiment-analysis")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


def autoscrap_autotrader(base_url, params, page_params, headers, max_pages):
    '''
    Ex:
    base_url = "https://www.autotrader.com/cars-for-sale/all-cars/dealer/cars-under-18000/
    headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) ...."}

    # Parameters for the first page
    params = {
        "marketExtension": "off",
        "mileage": "75000",
        "newSearch": "true",
        "searchRadius": "75",
        "startYear": "2018",
        "vehicleHistoryType": "CLEAN_TITLE",
    }

    # Parameters for subsequent pages
    page_params = {
        "marketExtension": "off",
        "mileage": "75000",
        "newSearch": "false",
        "numRecords": "100",
        "searchRadius": "75",
        "startYear": "2018",
        "vehicleHistoryType": "CLEAN_TITLE",
    }
    '''
    
    first_record = 0  # Starting point for pagination
    max_pages = max_pages  # Example limit
    current_page = 1
    
    # Initialize an empty list to store all listings
    all_listings = []
    
    while current_page <= max_pages:
        if current_page == 1:
            response = requests.get(base_url, headers=headers, params=params)
        else:
            page_params["firstRecord"] = first_record
            response = requests.get(base_url, headers=headers, params=page_params)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
        else:
            print(f"Failed to fetch page {current_page}", response.status_code)
            break
        
        car_listings = soup.find_all('div', class_='inventory-listing')  # Adjust this selector to match the site structure
        
        for listing in car_listings:
            # Extract title
            title_tag = listing.find('h2', class_='text-bold text-size-400 link-unstyled', attrs={'data-cmp': 'subheading'})
            title = title_tag.get_text(strip=True) if title_tag else "N/A"
            
            # Extract price
            price_tag = listing.find('div', class_='text-size-600 text-ultra-bold first-price', attrs={'data-cmp': 'firstPrice'})
            price = price_tag.get_text(strip=True) if price_tag else "N/A"
            
            # Extract dealership
            dealership_tag = listing.find('div', class_='text-subdued')
            dealership = dealership_tag.get_text(strip=True) if dealership_tag else "N/A"
            
            # Extract distance
            distance_tag = listing.find('span', class_='text-normal')
            distance = distance_tag.get_text(strip=True) if distance_tag else "N/A"
            
            # Extract mileage
            mileage_tag = listing.find('div', class_='text-bold text-subdued-lighter')
            mileage = mileage_tag.get_text(strip=True) if mileage_tag else "N/A"
            
            # Extract certification (handle potential absence)
            certification_tag = listing.find('div', class_='text-link text-subdued')
            certification = certification_tag.get_text(strip=True) if certification_tag else "Not Certified"
            
            # Store the listing in a dictionary
            listing_data = {
                "Title": title,
                "Price": price,
                "Dealership": dealership,
                "Distance": distance,
                "Mileage": mileage,
                "Certification": certification
            }
            
            # Append the listing to the list
            all_listings.append(listing_data)
        
        # Increment firstRecord for the next page
        first_record += 100
        current_page += 1
    
    # Create a pandas DataFrame from the list of listings
    df = pd.DataFrame(all_listings)
    
    # Display the DataFrame (you can also save it to a CSV file or further analyze it)
    print(df.head())

    # format df 
    if len(df) !=0:
        # Apply the function to the DataFrame and create new columns
        df[['Year', 'Make', 'Model']] = df['Title'].apply(lambda x: pd.Series(split_title(x)))
        # Drop empty rows
        print("dropping empty rows with empty titles", len(df[df['Title'] == 'N/A']))
        df = df[~(df['Title'] == 'N/A')].copy()
    return df


# Function to split the 'Title' column
def split_title(title):
    # Find the first occurrence of an integer (year)
    match = re.search(r'\b\d{4}\b', title)
    if match:
        year = match.group()
        # Split the string at the year
        parts = title.split(year, 1)
        if len(parts) == 2:
            before_year, after_year = parts
            # Split after the year to get make and model
            make_model = after_year.strip().split(' ', 1)
            if len(make_model) == 2:
                make, model = make_model
                return year, make, model
    return None, None, None


class CarRecommendationAnalyzer:
    def __init__(self, reddit_client_id, reddit_client_secret, reddit_username, reddit_password, reddit_user_agent, subreddit_name,
                hugging_face_token):
        self.API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
        self.headers = {"Authorization": hugging_face_token}
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
         
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            username=reddit_username,
            password=reddit_password,
            user_agent=reddit_user_agent
        )
        self.subreddit_name = subreddit_name
    
    def query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()
    
    def matches_query(self, comment, make, year, model):
        comment = comment.lower()
        make = make.lower()
        year = str(year)
        if make in comment and year in comment:
            if model:
                model = model.lower()
                return fuzz.partial_ratio(model, comment) > 70  # Adjust the threshold as needed
            return True
        return False
    
    def get_query_comments(self, search_queries):
        subreddit = self.reddit.subreddit(self.subreddit_name)
        comments = []
        for search_query in search_queries:
            make = search_query.split(" ")[0]
            year = search_query.split(" ")[1]
            model = search_query.split(" ")[2]
            
            # Handle rate limiting and retry logic
            while True:
                try:
                    for submission in subreddit.search(search_query):
                        submission.comments.replace_more(limit=0)
                        for comment in submission.comments.list():
                            if self.matches_query(comment.body, make, year, model):
                                comments.append(comment.body)
                    break  # Exit the loop if successful
                except praw.exceptions.APIException as e:
                    if e.error_type == "RATELIMIT":
                        delay = int(e.message.split("try again in ")[1].split(" ")[0])
                        time.sleep(delay)
                    else:
                        raise
        
        return comments
    
    def top_comments_by_sentiment(self, comments):
        tokenized = self.tokenizer(comments, truncation=True, padding=True, max_length=588)
        top_comment_dict = {'top_positives': {}, 'top_negatives': {}}
        sentiments = self.sentiment_pipeline(comments)
        
        positive_comments = [(comment['label'], comment['score'], idx) for idx, comment in enumerate(sentiments) if comment['label'] == 'POSITIVE']
        negative_comments = [(comment['label'], comment['score'], idx) for idx, comment in enumerate(sentiments) if comment['label'] == 'NEGATIVE']
        
        top_positive_comments = sorted(positive_comments, key=lambda x: x[1], reverse=True)[:15]
        top_negative_comments = sorted(negative_comments, key=lambda x: x[1], reverse=True)[:15]
        
        top_comment_dict['top_positives']['data'] = {i + 1: {'comment': comments[idx], 'score': score} for i, (_, score, idx) in enumerate(top_positive_comments)}
        top_comment_dict['top_negatives']['data'] = {i + 1: {'comment': comments[idx], 'score': score} for i, (_, score, idx) in enumerate(top_negative_comments)}
        
        return top_comment_dict
    