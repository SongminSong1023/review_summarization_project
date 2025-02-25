import os
import sys
import json
import logging
import argparse

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import font_manager
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')

NANUM_PATH = "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf"


# matplot 글꼴 설정
nanum_path = "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf" 
if os.path.exists(nanum_path):
    font_prop = font_manager.FontProperties(fname=nanum_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    print(f"Font set to: {font_prop.get_name()}")
else:
    print("NanumGothicCoding font not found. Using default font.")
    plt.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
################################################################################################################################################

def load_data_from_json(path):
    '''
    Load a JSON file and return keyword list
    '''
    if not os.path.isfile(path):
        logging.error(f"JSON file not found: {path}")
        return []

    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            return []

    return data.get("summaries", [])

def list_to_string(keyword_list):
    '''
    Return keyword list as a string (perform additional preprocessing steps such as removing stopwords if necessary)
    '''
    return " ".join(keyword_list)

def generate_wordcloud(string_keyword, args):
    '''
    Generate word cloud from keyword and save it
    '''
    font_path_used = NANUM_PATH if os.path.exists(NANUM_PATH) else None
    mask = np.array(Image.open("/mnt/nas4/sms/review_summarization_project/data/wordcloud_mask/cloud.png"))

    # wordcloud Settings
    wordcloud = WordCloud(
        font_path=font_path_used,
        width=800,
        height=400,
        background_color="white",
        mask=mask,
        max_words=100,
        contour_width=1,
        contour_color='steelblue',
        colormap="gist_ncar"
    )

    wordcloud.generate(string_keyword)
    output_dir = f"/mnt/nas4/sms/review_summarization_project/outputs/visualized/{args.food}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}{args.restaurant}_{args.region}.png"

    # plt Settings
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("Word Cloud Visualization")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    logging.info(f"Saved Word Cloud for {args.restaurant} to {output_path}")

def get_top_n_words(summaries, n=10):
    '''
    Extract the top n most frequent words from the summaries using a Korean tokenizer
    '''
    okt = Okt()  # Initialize Korean tokenizer
    all_words = []

    # Tokenize each summary and collect nouns
    for summary in summaries:
        tokens = okt.nouns(summary)  # Extract nouns (adjustable to other POS if needed)
        all_words.extend(tokens)

    # Define Korean stopwords (expand this list as needed)
    stopword_dir = "/mnt/nas4/sms/review_summarization_project/data/korean_stopwords.txt"
    with open(stopword_dir, 'r', encoding='utf-8') as f:
        korean_stopwords = f.read    
    korean_stopwords = korean_stopwords.split("\n")

    # Filter out stopwords and single-character words
    filtered_words = [word for word in all_words if word not in korean_stopwords and len(word) > 1]

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Get the top n most frequent words
    top_words = word_counts.most_common(n)

    return top_words

def main(args):
    input_path = f"/mnt/nas4/sms/review_summarization_project/outputs/organized/{args.food}/{args.restaurant}_{args.region}.json"
    keyword_list = load_data_from_json(input_path)
    string_keyword = list_to_string(keyword_list)
    generate_wordcloud(string_keyword, args)

    top_words = get_top_n_words(summaries, n=10)
    print(f"\nTop 10 most frequent words for {args.restaurant} in {args.region}:")
    for word, freq in top_words:
        print(f"{word}: {freq}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--food", type=str, required=True, help="음식 이름")
    parser.add_argument("--restaurant", type=str, required=True, help="식당 이름")
    parser.add_argument("--region", type=str, required=True, help="지역 이름")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="출력 다양성을 제어하는 파라미터, 낮을 수록 일관된 응답")
    parser.add_argument("--top_p", type=float, default=0.8,
                        help="후보 단어들을 제한하는 용도의 파라미터, 높을 수록 창의적")
    args = parser.parse_args()

    main(args)