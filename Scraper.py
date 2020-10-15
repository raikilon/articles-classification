import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import os


class Scraper:
    def __init__(self, max_num=50):
        self.max_num = max_num
        self.df = pd.DataFrame(columns=["text", "label"])

    def __parse_links(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        regex = re.compile(r'/wiki/.+')

        # Get article categories
        categories = soup.find(id="mw-normal-catlinks")

        # Check if page contains categories (e.g. an image page does not have this information)
        if categories is None:
            return

        categories = categories.select('li a')
        # Continue only if articles contains the given category
        if any(self.category in cat.text.lower() for cat in categories):

            # Get article content
            text = ''
            for paragraph in soup.find_all('p'):
                text += paragraph.text

            # Drop footnote superscripts in brackets and Replace ‘\n’ (a new line) with ‘’ (an empty string)
            text = re.sub(r'\[.*?\]+', '', text)
            text = text.replace('\n', '')

            self.num += 1

            # Add article to dataframe
            self.df = self.df.append({
                "text": text,
                "label": self.category
            }, ignore_index=True)

            # Scrape other pages from links
            # Get all the links
            links = soup.find(id="bodyContent").find_all("a", {'href': True})

            for link in links:
                # Only interested in other wiki articles (remove external links and images)
                if re.match(regex, link.get('href')):
                    self.to_crawl.put("https://en.wikipedia.org" + link.get('href'))

    def __post_scrape_callback(self, res):
        result = res.result()
        if result and result.status_code == 200 and self.num < self.max_num:
            self.__parse_links(result.text)

    def __scrape_page(self, url):
        try:
            res = requests.get(url, timeout=(3, 30))
            return res
        except requests.RequestException:
            return

    def __run_scraper(self, base_urls, category):

        # Set initial values
        self.pool = ThreadPoolExecutor(max_workers=6)
        self.category = category
        self.scraped_pages = set([])
        self.to_crawl = Queue()
        # add base urls
        for url in base_urls:
            self.to_crawl.put(url)
        self.num = 0

        while True:
            try:
                target_url = self.to_crawl.get(timeout=60)
                # return if reached target number of articles
                if self.num > self.max_num:
                    self.pool.shutdown(wait=False)
                    return
                if target_url not in self.scraped_pages:
                    #print("Scraping URL: {}".format(target_url))
                    self.scraped_pages.add(target_url)
                    job = self.pool.submit(self.__scrape_page, target_url)
                    job.add_done_callback(self.__post_scrape_callback)
            except Empty:
                return
            except Exception as e:
                print(e)
                continue

    def __clean_txt(self, text):
        text = re.sub("'", "", text)
        text = re.sub("(\\W)+", " ", text)
        return text

    def get(self):
        self.__run_scraper(["https://en.wikipedia.org/wiki/Sport"], "sport")
        self.__run_scraper(["https://en.wikipedia.org/wiki/Economy"], "economy")
        self.__run_scraper(["https://en.wikipedia.org/wiki/Engineering"], "engineering")
        self.__run_scraper(["https://en.wikipedia.org/wiki/History"], "history")
        self.__run_scraper(["https://en.wikipedia.org/wiki/Philosophy"], "philosophy")
        self.__run_scraper(["https://en.wikipedia.org/wiki/Politics"], "politics")
        self.__run_scraper(["https://en.wikipedia.org/wiki/Religion"], "religion")
        self.__run_scraper(["https://en.wikipedia.org/wiki/Food"], "food")
        self.__run_scraper(["https://en.wikipedia.org/wiki/Law"], "law")
        self.__run_scraper(["https://en.wikipedia.org/wiki/Culture"], "culture")
        # Clean text
        self.df['text'] = self.df.text.apply(self.__clean_txt)
        self.df.to_pickle("data/dataframe.pkl")
        return self.df

    def load(self):
        if os.path.isfile('data/dataframe.pkl'):
            return pd.read_pickle("data/dataframe.pkl")
        else:
            return pd.DataFrame()
