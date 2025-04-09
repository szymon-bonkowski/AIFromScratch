import requests
from bs4 import BeautifulSoup
import time
import os

urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning"
]

def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs])
        return text.strip()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def save_text_to_file(text, filepath="train.txt"):
    mode = "a" if os.path.exists(filepath) else "w"
    with open(filepath, mode, encoding="utf-8") as f:
        f.write(text + "\n\n")

if __name__ == "__main__":
    output_file = "train.txt"
    if os.path.exists(output_file):
        os.remove(output_file)

    for url in urls:
        print(f"Processing: {url}")
        text = scrape_text_from_url(url)
        if text:
            save_text_to_file(text, output_file)
            print(f"Saved content from {url}")
        time.sleep(1)
    
    print(f"Data collection complete. Data saved in {output_file}")
