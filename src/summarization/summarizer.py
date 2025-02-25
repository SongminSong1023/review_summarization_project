import os
import sys
import json
import torch
import logging
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def load_reviews_from_json(path):
    '''
    Load a JSON file that contains a list of reviews in the following format:
      [
        {"review_text": "...", "star_rating": 4, ...},
        {"review_text": "...", "star_rating": 5, ...},
        ...
      ]
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

    reviews = []
    for item in data:
        text = item.get("review_text", "").strip()
        if text:
            reviews.append(text)
    return reviews

def build_prompt_header(args):
    '''
    Build the initial prompt header that will appear before the combined reviews.
    '''
    header = (
        f"다음은 {args.region} 지역의 음식점 '{args.restaurant}'의 "
        f"'{args.food}'에 대한 리뷰야.\n"
        "음식에 대한 리뷰만을 참고하여 요약해줘. 음식과 관련되지 않은 서비스 등에 관한 리뷰들은 전부 무시해!\n\n"
    )
    return header

def build_request_structure():
    '''
    Build the part of the prompt that asks for a specific JSON structure
    in the answer.
    '''
    structure = (
        "\n\n아래와 같은 JSON 구조로 요약해줘:\n\n"
        "{\n"
        "\"summary\": \"리뷰 내용을 요약한 문장.\",\n"
        "\"positives\": [\"긍정적 요인들\"],\n"
        "\"negatives\": [\"부정적 요인들\"],\n"
        "\"keywords\": [\"지역 정보나 음식점 특색과 같은 정보들\"]\n"
        "}\n\n"
        "위 구조를 꼭 지켜서 작성해줘."
    )
    return structure

def combine_reviews(reviews):
    '''
    Combine a list of review texts into a single string with line breaks.
    '''
    return "\n".join(reviews)

def create_prompt(chunk_reviews, args):
    '''
    Creates the final prompt for a chunk of reviews.
    '''
    header = build_prompt_header(args)
    request_structure = build_request_structure()
    combined = combine_reviews(chunk_reviews)

    # Combine them into one string
    prompt = f"{header}{combined}{request_structure}"
    return prompt

def load_model_and_tokenizer(model_name, token=None):
    '''
    Load the causal language model and tokenizer with BitsAndBytes 8-bit quantization.
    '''
    quantization_config = BitsAndBytesConfig(
        load_in_8bits=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token
    )

    return model, tokenizer

def generate_chunk_summary(
    model, tokenizer, chunk_reviews, args,
    max_new_tokens=512
):
    '''
    Generate the summary for a single chunk of reviews.
    Return the decoded output as a string.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct the prompt
    prompt = create_prompt(chunk_reviews, args)

    messages = [
        {
            "role": "system", 
            "content": (
                "You are EXAONE model from LG AI Research, analyze restaurant "
                "review data and summarize it."
            )
        },
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model.generate(
            input_ids.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

def chunk_reviews(reviews, chunk_size=10):
    '''
    Break a list of reviews into smaller chunks, each up to `chunk_size` items.
    '''
    for i in range(0, len(reviews), chunk_size):
        yield reviews[i : i + chunk_size]

def summarization(model, tokenizer, reviews, args, chunk_size=10):
    '''
    Summarize each chunk of review
    '''
    # Summarize each chunk
    chunk_summaries = []
    for chunk in tqdm(list(chunk_reviews(reviews, chunk_size))):
        summary_text = generate_chunk_summary(model, tokenizer, chunk, args)
        chunk_summaries.append(summary_text)

    return chunk_summaries


def main(args):
    '''
    Main pipeline:
      1) Load reviews from JSON file.
      2) Load the model & tokenizer.
      3) Summarize the reviews (optionally in a hierarchical manner).
      4) Save or print the results.
    '''
    # Build the input path
    input_path = f"/mnt/nas4/sms/review_summarization_project/data/raw/reviews_{args.food}/reviews_{args.restaurant}_{args.region}.json"
    logging.info(f"Loading reviews from {input_path}")
    reviews = load_reviews_from_json(input_path)
    if not reviews:
        logging.warning("No reviews found. Exiting.")
        return

    # Load the model
    logging.info("Loading LLM and tokenizer...")
    model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    sms_token = os.getenv("sms_token")
    model, tokenizer = load_model_and_tokenizer(model_name, sms_token)

    # Decide on chunk size, or compute based on total review count
    if args.chunk_size is None:
        # Simple heuristic: chunk half if large, or everything if small
        chunk_size = max(1, len(reviews) // 2) if len(reviews) > 20 else len(reviews)
    else:
        chunk_size = args.chunk_size

    # Summarize
    logging.info(f"Generating summaries with chunk_size={chunk_size}...")
    summaries = summarization(
        model, tokenizer, reviews, args,
        chunk_size=chunk_size,
    )

    # Summaries might be a list of chunk summaries or a single final summary in index 0
    if len(summaries) == 1:
        final_result = summaries[0]
    else:
        
        final_result = "\n\n".join(summaries)

    # Save the final summary to a file
    output_dir = f"/mnt/nas4/sms/review_summarization_project/outputs/raw/{args.food}"
    output_path = f"/mnt/nas4/sms/review_summarization_project/outputs/raw/{args.food}/{args.restaurant}_{args.region}.txt"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_result)

    
    logging.info("\n" + final_result)
    logging.info(f"Summary saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--food", type=str, required=True, help="음식 이름")
    parser.add_argument("--restaurant", type=str, required=True, help="식당 이름")
    parser.add_argument("--region", type=str, required=True, help="지역 이름")
    parser.add_argument("--chunk_size", type=int, default=10,
                        help="리뷰를 몇 개씩 묶어 요약할지 *OutOfMemoryError가 일어날 가능성이 농후해 10~20 권장")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="출력 다양성을 제어하는 파라미터, 낮을 수록 일관된 응답")
    args = parser.parse_args()

    main(args)