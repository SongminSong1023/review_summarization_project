from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import os
import sys
from tqdm import tqdm
import json

"""
Chrome과 Chromedriver 버전 확인
CLASS_NAME에 빈칸이 있는 경우 By.CLASS_NAME 대신에 By.CSS_SELECTOR 사용 + 빈칸은 .으로 대체
미작동시 XPATH, CSS_SELECTER, CLASS_NAME 등 확인
"""

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
    # 타켓 URL 열기
    driver.get(url)
    time.sleep(3)

# def search_keyword(driver, keyword):
#     # 구글 검색
#     search_box = driver.find_element(By.XPATH, '//*[@id="searchboxinput"]')
#     search_box.send_keys(keyword)
#     search_box.send_keys(Keys.RETURN)
#     time.sleep(3)

# def click_restaurant(driver):
#     try:
#         wait = WebDriverWait(driver, 10)
#         restaurant_button = wait.until(
#             EC.element_to_be_clickable((By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[1]/div[1]/div[3]/div/a'))
#         )
#         restaurant_button.click()
#         time.sleep(3)
#     except Exception as e:
#         print("Error clicking google restaurant:", e)

def click_review_button(driver):
    # 검색한 음식점 리뷰창으로 이동

    wait = WebDriverWait(driver, 10)
    driver.save_screenshot("/sms/flavor_recommender/before_click_review.png")

    try:
        wait = WebDriverWait(driver, 10)
        review_button = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//a[span[contains(text(),'리뷰')]]"))
        )
        review_button.click()
        time.sleep(3)
    except Exception as e:
        print("Error clicking naver review button:", e)
        driver.quit()

def scrape_reviews(driver):
    wait = WebDriverWait(driver, 10)
    reviews = []
    unique_reviews = set()

    try:
        # Review container 대기
        review_container = wait.until(
            EC.presence_of_element_located(
                (By.CLASS_NAME, 'place_section_content')
            )
        )

        time.sleep(1)

        # 더보기 버튼
        while True:
            try:
                more_button = wait.until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "TeItc"))
                )
                more_button.click()
                time.sleep(1)
            except:
                break


        new_reviews = gather_visible_reviews(driver, unique_reviews)
        reviews.extend(new_reviews)

    except Exception as e:
        print("Error scraping reviews:", e)

    
    return reviews

def gather_visible_reviews(driver, unique_reviews):
    wait = WebDriverWait(driver, 5)
    review_blocks = driver.find_elements(By.CSS_SELECTOR, ".pui__X35jYm.place_apply_pui.EjjAW")
    

    new_texts = []
    for single_review in review_blocks:
        # 자세히 버튼 확인
        try:
            detail_button = single_review.find_element(By.CLASS_NAME, 'pui__wFzIYl')
            detail_button.click()
            time.sleep(0.5)
        except:
            pass  # 자세히 버튼 없으면 생략
        # 리뷰 텍스트 추출
        try:
            review_text_element = single_review.find_element(By.CLASS_NAME, "pui__vn15t2")
            text = review_text_element.text.strip()
            if text and text not in unique_reviews:
                unique_reviews.add(text)
                new_texts.append(text)
        except:
            continue

    return new_texts

def read_restaurants_from_json(filename):
    """
    식당 url을 모아둔 json파일 읽어오기
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("restaurants", {})


def main():
    driver = None
    food = "감자탕2"
    input_file = f"./restaurant_url/{food}.json"        # json 파일 경로
    output_dir = f"reviews_{food}"                      # 저장 경로
    os.makedirs(output_dir, exist_ok=True)
    
    restaurants = read_restaurants_from_json(input_file)

    driver = init_driver()

    for name, url in restaurants.items():
        try:
            print(f"Scraping reviews for: {name}")
            open_target_page(driver, url)
            click_review_button(driver)
            reviews = scrape_reviews(driver)
            
            with open(os.path.join(output_dir, f"reviews_{name}.txt"), "w", encoding="utf-8") as f:
                for review in reviews:
                    f.write(review + "\n")
            
            print(f"Saved {len(reviews)} reviews for {name}")
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    driver.quit()

if __name__ == "__main__":
    main()