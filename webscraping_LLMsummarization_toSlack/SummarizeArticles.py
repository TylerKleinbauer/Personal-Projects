import json
import requests
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import openai
from openai import OpenAI
import os
import cloudscraper
from bs4 import BeautifulSoup
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize variables to None
article_title = None
article_body = None
article_url_full = None
article_details = []

# Functions
def summarize_article(article_title, article_text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant who summarizes articles for busy executives. Your summaries are as short as possible, and as detailed as necessary. You summarize in 3 bullet points maximum."},
        {"role": "user", "content": f"Title: {article_title}\n\n{article_text}\n\nCan you summarize this article for me? Please summarize it in bullet point format. You summarize in 3 bullet points maximum."}
    ]
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    summary = chat_completion.choices[0].message.content
    return summary

def send_message_to_slack(summary, article_title, article_url_full, blog_source):
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    message_text = f"*{blog_source}*\n\n*Title:* {article_title}\n\n*Summary:* \n {summary}\n\n*Read the full article:* {article_url_full}\n______________"
    slack_data = {'text': message_text}
    response = requests.post(
        webhook_url, data=json.dumps(slack_data), 
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code != 200:
        raise ValueError(f"Request to Slack returned an error {response.status_code}, the response is:\n{response.text}")

# Get the dates in correct formats
yesterdays_date = (datetime.now() - timedelta(days=1)).strftime('%B %-d, %Y')
logging.info(f"Yesterday's date: {yesterdays_date}")
yesterdays_date_snowflake = (datetime.now() - timedelta(days=1)).strftime('%b %d, %Y')
logging.info(f"Yesterday's date for Snowflake: {yesterdays_date_snowflake}")

# Initialize Chrome options for Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Selenium setup to open and scrape the blog page
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Helper function to accept cookies if present
def accept_cookies():
    try:
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))).click()
    except TimeoutException:
        pass

# Function to collect articles using cloudscraper and Selenium for scrolling
def collect_articles_with_cloudscraper_and_selenium(blog_url, article_selector, date_selector, link_selector, date_format):
    scraper = cloudscraper.create_scraper(
        interpreter="nodejs",
        delay=10,
        browser={
            "browser": "chrome",
            "platform": "ios",
            "desktop": False,
        },
        #captcha={
        #    "provider": "2captcha",
        #    "api_key": "YOUR_2CAPTCHA_API",
        #},
    )
    
    # Use cloudscraper to pass the Cloudflare challenge
    response = scraper.get(blog_url)
    cookies = response.cookies
    headers = response.headers
    
    # Use Selenium to load the page with the cookies from cloudscraper
    driver.get(blog_url)
    for cookie in cookies:
        driver.add_cookie({'name': cookie.name, 'value': cookie.value, 'domain': cookie.domain, 'path': cookie.path})
    driver.refresh()
    accept_cookies()
    
    SCROLL_PAUSE_TIME = 2
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        # Wait to load the page
        time.sleep(SCROLL_PAUSE_TIME)
        
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    # After scrolling, get the page source and parse with BeautifulSoup
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    article_links = []
    articles = soup.select(article_selector)
    logging.info(f"Found {len(articles)} articles on {blog_url}")
    for article in articles:
        try:
            date_element = article.select_one(date_selector)
            if date_element:
                article_date = date_element.get_text(strip=True)
                logging.info(f"Article date: {article_date}")
                if article_date == date_format:
                    link_element = article.select_one(link_selector)
                    logging.info(f"Comparing article date: {article_date} with expected date: {date_format}")
                    if link_element:
                        article_link = link_element['href']
                        if not article_link.startswith('http'):
                            article_link = blog_url.split('/blog')[0] + article_link
                        article_links.append(article_link)
                        logging.info(f"Collected article link: {article_link}")
        except Exception as e:
            logging.error(f"An error occurred while collecting links: {str(e)}")
    return article_links

# Function to process each article
def process_articles(article_links, title_selector, body_selector, source):
    for link in article_links:
        logging.info(f"Processing article {link} from {source}")
        driver.get(link)
        try:
            article_title = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, title_selector))).text
            article_body_elements = driver.find_elements(By.CSS_SELECTOR, body_selector)
            article_body = ' '.join([element.text for element in article_body_elements])
            article_details.append({"title": article_title, "url": link, "body": article_body, "source": source})
            logging.info(f"Processed article titled: {article_title}")
        except Exception as e:
            logging.error(f"An error occurred while processing an article: {str(e)}")

# Collect and process articles for each blog
blogs = [
    {
        "url": 'https://www.tableau.com/blog',
        "article_selector": "ul.views-infinite-scroll-content-wrapper li",
        "date_selector": ".card__attribution-date",
        "link_selector": "h3.card__title a",
        "date_format": yesterdays_date,
        "title_selector": "h1.hero__title",
        "body_selector": 'article.content div p, article.content div h2, article.content div figcaption',
        "source": "Tableau"
    },
    {
        "url": 'https://www.snowflake.com/blog/',
        "article_selector": ".m-blog-latest-posts__post",
        "date_selector": ".blog-post-meta .post-date",
        "link_selector": '.post-link',
        "date_format": yesterdays_date_snowflake,
        "title_selector": "h1",
        "body_selector": '.post-content p, .post-content h3, .post-content h4',
        "source": "Snowflake"
    },
    {
        "url": 'https://blog.dataiku.com/',
        "article_selector": ".post-item",
        "date_selector": "time",
        "link_selector": "a.learn-more",
        "date_format": yesterdays_date,
        "title_selector": "h1",
        "body_selector": '.content p, .content h2',
        "source": "Dataiku"
    },
    {
        "url": 'https://datavault-builder.com/data-vault-blog/',
        "article_selector": ".elementor-post",
        "date_selector": ".elementor-post__meta-data .elementor-post-date",
        "link_selector": '.elementor-post__thumbnail__link',
        "date_format": yesterdays_date,
        "title_selector": "h1",
        "body_selector": '.elementor-widget-container p, .elementor-widget-container h3, .elementor-widget-container h4, .elementor-widget-container h5, .elementor-widget-container h6',
        "source": "Datavault Builder"
    }
]

for blog in blogs:
    logging.info(f"Collecting articles from {blog['url']}")
    article_links = collect_articles_with_cloudscraper_and_selenium(blog['url'], blog['article_selector'], blog['date_selector'], blog['link_selector'], blog['date_format'])
    if article_links:
        process_articles(article_links, blog['title_selector'], blog['body_selector'], blog['source'])
    else:
        logging.info(f"No articles found for {blog['url']}")

driver.quit()

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Process each article: summarize and send to Slack
for article in article_details:
    summary = summarize_article(article['title'], article['body'])
    send_message_to_slack(summary, article['title'], article['url'], article['source'])
