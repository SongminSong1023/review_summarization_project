from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import random
import time
import os
import sys
from tqdm import tqdm
import json
import argparse
import logging

"""
Chrome과 Chromedriver 버전 확인
CLASS_NAME에 빈칸이 있는 경우 By.CLASS_NAME 대신에 By.CSS_SELECTOR 사용 + 빈칸은 .으로 대체
미작동시 XPATH, CSS_SELECTER, CLASS_NAME 등 확인
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def init_driver():
    chrome_options = Options()
    chrome_binary = "/usr/bin/google-chrome"    
    chrome_options.binary_location = chrome_binary
    
    # Add arguments
    chrome_options.add_argument("--headless=new")  # Updated headless argument
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    try:
        service = Service(executable_path="/usr/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        print(f"\nFailed to create driver: {str(e)}")
        print("\nEnvironment Information:")
        print("Python executable:", sys.executable)
        print("Working directory:", os.getcwd())
        print("PATH:", os.environ.get('PATH'))
        raise


def open_target_page(driver, url):
    '''
    Open the target URL
    '''
    driver.get(url)
    time.sleep(random.uniform(2, 4))

def click_restaurant(driver):
    '''
    Click the restaurant button
    '''
    try:
        wait = WebDriverWait(driver, 10)
        restaurant_button = wait.until(
            EC.element_to_be_clickable((By.CLASS_NAME, 'hfpxzc'))
        )
        restaurant_button.click()
        time.sleep(random.uniform(2, 4))
    except Exception as e:
        print("Error clicking google restaurant:", e)

def click_review_tab(driver):
    '''
    Click the review tab
    '''

    wait = WebDriverWait(driver, 10)

    try:
        wait = WebDriverWait(driver, 10)
        review_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, '//button[@role="tab" and contains(@aria-label,"리뷰")]'))
        )
        review_button.click()
        time.sleep(random.uniform(2, 4))
    except Exception as e:
        logging.error(f"Error opening review tab for {restaurant_name}: {e}")
        return []

def scrape_reviews_for_restaurant(driver, restaurant_name, url, max_reviews=None):
    """
    Open the page, navigate to the review section, scroll, and gather reviews.
    Optionally limit to `max_reviews` if provided.
    """
    driver.get(url)
    time.sleep(random.uniform(2, 4))

    # Click review tab
    try:
        wait = WebDriverWait(driver, 10)
        review_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, '//button[@role="tab" and contains(@aria-label,"리뷰")]'))
        )
        review_button.click()
        time.sleep(random.uniform(2, 4))
    except Exception as e:
        logging.error(f"Error opening review tab for {restaurant_name}: {e}")
        return []

    # Scroll and collect reviews
    reviews_collected = []
    unique_reviews = set()
    review_container_sel = '.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde'

    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, review_container_sel)))
        review_container = driver.find_element(By.CSS_SELECTOR, review_container_sel)
        
        last_height = -1
        
        while True:
            driver.execute_script("arguments[0].scrollTo(0, arguments[0].scrollHeight);", review_container)
            time.sleep(random.uniform(2, 3))  # random short sleep

            new_height = driver.execute_script("return arguments[0].scrollHeight", review_container)
            if new_height == last_height:
                logging.info("No more reviews could be loaded by scrolling.")
                break
            last_height = new_height

            # Gather visible reviews
            new_items = gather_visible_reviews(driver, unique_reviews)
            reviews_collected.extend(new_items)

            # If we have a limit and we've reached it, stop
            if max_reviews and len(reviews_collected) >= max_reviews:
                break

    except Exception as e:
        logging.error(f"Error scraping reviews for {restaurant_name}: {e}")

    return reviews_collected

def gather_visible_reviews(driver, unique_reviews):
    """
    Find each review block, expands if needed, and extracts relevant info (text, rating, date, etc.).
    Return a list of dictionaries (or strings if just text).
    """
    review_blocks = driver.find_elements(By.CSS_SELECTOR, ".jftiEf.fontBodyMedium")
    extracted = []

    for block in review_blocks:
        # Expand "자세히" if present
        try:
            detail_btn = block.find_element(By.CSS_SELECTOR, '.w8nwRe.kyuRq')
            detail_btn.click()
            time.sleep(0.5)
        except:
            pass

        try:
            text_el = block.find_element(By.CLASS_NAME, "wiI7pd")
            review_text = text_el.text.strip()
            # rate_el = block.find_element(By.CLASS_NAME, "kvMYJc")
            # aria_label = rate_el.get_attribut("aria-label")

            # star_rating = None
            # if aria_label:
                # match = re.search(r'(\d+)', aria_label)
                # if match:
                #     star_rating = float(match.group(1))
            
            
            if review_text and review_text not in unique_reviews:
                unique_reviews.add(review_text)
                extracted.append({
                    "review_text": review_text,
                    # "star_rating": star_rating,
                })
        except:
            continue

    return extracted

def save_reviews(reviews, name, output_dir):
    filename = os.path.join(output_dir, f"reviews_{name}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved {len(reviews)} reviews to {filename}")

def read_restaurants_from_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("restaurants", {})

def main(food, max_reviews=None):
    input_file = f"/mnt/nas4/sms/review_summarization_project/data/url/{food}.json"                         # json 파일 경로
    output_dir = f"/mnt/nas4/sms/review_summarization_project/data/raw/reviews_{food}"                      # 저장 경로

    os.makedirs(output_dir, exist_ok=True)

    # Read the restaurant URLs
    restaurants = read_restaurants_from_json(input_file)
    driver = init_driver()

    for name, url in restaurants.items():
        logging.info(f"Scraping reviews for: {name}")
        reviews = scrape_reviews_for_restaurant(driver, name, url, max_reviews=max_reviews)
        save_reviews(reviews, name, output_dir)

    driver.quit()


if __name__ == "__main__":
    '''
    가상환경 sms_exaone으로
    json 파일에 스크랩할 식당의 구글맵 링크를 딕셔너리 형태로
    python3 review_scrapper_google.py --food "음식이름"
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--food", type=str, required=True, help="음식이름입력")
    parser.add_argument("--max_reviews", type=int, default=None, help="최대 리뷰 수")
    args = parser.parse_args()

    main(args.food)