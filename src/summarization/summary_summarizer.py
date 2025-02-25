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

def load_information_from_json(path):
    '''
    Load a JSON file containing a list of summaries, positives, negatives, and keywords and return lists
    '''
    if not os.path.isfile(path):
        logging.error(f"JSON file not found: {path}")
        return []

    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            return None, None, None, None

    return (
        data.get("summaries", []),
        data.get("positives", []),
        data.get("negatives", []),
        data.get("keywords", [])
    )


def load_model_and_tokenizer(model_name, token=None):
    '''
    Load the causal language model and tokenizer with BitsAndBytes 8-bit quantization.
    '''
    quantization_config = BitsAndBytesConfig(
        load_in_4bits=True,
        bnb_4bit_compute_dtype=torch.float16,
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

def prompt_for_summary(summary_list, args):
    '''
    Prompt for generating a single coherent summary from multiple paragraphs.
    Return the constructed prompt as a string.
    '''
    header = (
        f"\n\nSummarize the following summaries into a single, coherent, and concise summary"
        "that retains the key points and main ideas without unnecessary repetition.\n\n"
        "Return the output as a string.\n\n"
    )
    combined = "\n".join(summary_list)
    prompt = f"{header}{combined}"

    return prompt

def prompt_for_list(info_list, args):
    '''
    Build the prompt to abstract key elements
    '''
    header = (
        f"\n\nAbstract the key elements from the following list,"
        "ensuring that similar but distinct elements are paraphrased and differentiated using appropriate synonyms."
        "Maintain the original meaning while improving clarity and conciseness.\n\n"
        "Return the output in Korean \n\n"
    )
    combined = "\n".join(info_list)
    prompt = f"{header}{combined}"

    return prompt

def abstract_final_summary(
    model, tokenizer, args, summaries, max_new_tokens=512
):
    '''
    Summarize multiple summaries.
    Returns the concise and comprehensive summary as a string.
    '''
    if not summaries:
        logging.info("No summaries to process")
        return ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct the prompt
    prompt = prompt_for_summary(summaries, args)

    messages = [
        {
            "role": "system", 
            "content": (
                "You are EXAONE model from LG AI Research"
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
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded_output = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return decoded_output

def abstract_key_elements(
    model, tokenizer, args, elements_list, max_new_tokens=512
):
    '''
    Abstract key elements from a list (for positives, negatives, or keywords).
    Returns a string.
    '''
    if not elements_list:
        logging.info("No list to process for abstraction.")
        return ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct the prompt
    prompt = prompt_for_list(elements_list, args)

    messages = [
        {
            "role": "system", 
            "content": (
                "You are EXAONE model from LG AI Research"
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
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded_output = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return decoded_output

def main(args):
    '''
    Main pipeline:
      1) Load lists(summaries, positives, negatives, keywords) from JSON file.
      2) Load the model & tokenizer.
      3) Abstract key elements from the list (optionally in a hierarchical manner).
      4) Save or print the results.
    '''
    # Build the input path
    input_path = f"/mnt/nas4/sms/review_summarization_project/outputs/organized/{args.food}/{args.restaurant}_{args.region}.json"
    logging.info(f"Loading summary from {input_path}")
    
    # Load lists from a JSON file.
    all_summaries, all_positives, all_negatives, all_keywords = load_information_from_json(input_path)

    # Load the model
    logging.info("Loading LLM and tokenizer...")
    model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    sms_token = os.getenv("sms_token")
    model, tokenizer = load_model_and_tokenizer(model_name, sms_token)

    # Abstract key elements
    final_summary = abstract_final_summary(model, tokenizer, args, all_summaries)
    abstracted_positives = abstract_key_elements(model, tokenizer, args, all_positives)
    abstracted_negatives = abstract_key_elements(model, tokenizer, args, all_negatives)
    abstracted_keywords = abstract_key_elements(model, tokenizer, args, all_keywords)

    logging.info("\n========== Final Summaries ==========\n")
    logging.info(final_summary)
    logging.info("\n========== Positives ==========\n")
    logging.info(abstracted_positives)
    logging.info("\n========== Negatives ==========\n")
    logging.info(abstracted_negatives)
    logging.info("\n========== Keywords ==========\n")
    logging.info(abstracted_keywords)

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