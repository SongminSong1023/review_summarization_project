import os
import json
import re
import argparse

def extract_summary(json_blocks):
    '''
    Extract all summaries from JSON blocks and return them as a list
    '''
    # List for storing extracted summary
    all_summaries = []

    for block in json_blocks:
        try:
            data = json.loads(block)
            summary = data.get("summary")
            all_summaries.append(summary)
        except json.JSONDecodeError as e:
            pass

    return all_summaries

def extract_positives(json_blocks):
    '''
    Extract all positives from JSON blocks and return them as a list
    '''
    all_positives = []

    for block in json_blocks:
        try:
            data = json.loads(block)
            positives = data.get("positives")
            for positive in positives:
                all_positives.append(positive)
        except json.JSONDecodeError as e:
            pass

    return all_positives

def extract_negatives(json_blocks):
    '''
    Extract all negatives from JSON blocks and return them as a list
    '''
    all_negatives = []

    for block in json_blocks:
        try:
            data = json.loads(block)
            negatives = data.get("negatives")
            for negative in negatives:
                all_negatives.append(negative)
        except json.JSONDecodeError as e:
            pass

    return all_negatives

def extract_keywords(json_blocks):
    '''
    Extract all keywords from JSON blocks and return them as a list
    '''
    all_keywords = []

    for block in json_blocks:
        try:
            data = json.loads(block)
            keywords = data.get("keywords")
            for keyword in keywords:
                all_keywords.append(keyword)
        except json.JSONDecodeError as e:
            pass
        except TypeError:
            #print(keywords)
            pass

    return all_keywords

# def extract_sentiment(json_blocks):
#     '''
#     Extract all sentiments from JSON blocks and return them as a list
#     '''
#     # List for storing extracted sentiments
#     all_sentiments = []

#     for block in json_blocks:
#         try:
#             data = json.loads(block)
#             sentiment = data.get("overallSentiment")
#             all_sentiments.append(sentiment)
#         except json.JSONDecodeError as e:
#             pass

#     return all_sentiments

def extract(txt_path):
    '''
    Extract json block from raw data and organize data
    '''
    # Pattern matching for a JSON block
    pattern = r'```json\s*(\{.*?\})\s*```'

    # Filtering txt file in path
    txt_files = [
        f for f in os.listdir(txt_path)
        if os.path.isfile(os.path.join(txt_path, f)) and f.endswith('.txt')
    ]

    # Iterate through all text files
    for txt_file in txt_files:
        print(f"Start to extract data from {txt_file}")
        # Open text file
        with open(os.path.join(txt_path, txt_file), encoding='utf-8') as f:
            txt_data = f.read()

        # Find all the JSON blcok
        json_blocks = re.findall(pattern, txt_data, flags=re.DOTALL)

        # Extract all features as a list
        all_summaries = extract_summary(json_blocks)
        all_positives = extract_positives(json_blocks)
        all_negatives = extract_negatives(json_blocks)
        #all_sentiment = extract_sentiment(json_blocks)
        all_keywords = extract_keywords(json_blocks)

        # Store the extracted data as JSON
        extracted_data = {
            "summaries": all_summaries,
            "positives": all_positives,
            "negatives": all_negatives,
            "keywords": all_keywords
        }

        # Save the organized results in a file with the same name
        output_path = f"/mnt/nas4/sms/review_summarization_project/outputs/organized/{args.food}"
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        base_name, _ = os.path.splitext(txt_file)
        output_file = f"{base_name}.json"
        with open(os.path.join(output_path, output_file), "w", encoding="utf-8") as out_f:
            json.dump(extracted_data, out_f, ensure_ascii=False, indent=2)


def main(args):
    txt_path = f"/mnt/nas4/sms/review_summarization_project/outputs/raw/{args.food}"
    extract(txt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--food", type=str, required=True, help="음식 이름")
    args = parser.parse_args()
    main(args)