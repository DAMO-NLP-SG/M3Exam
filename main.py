import os
import json
import openai
import pandas as pd
import argparse
import random
import requests
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)
import concurrent.futures

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=str, default='5', help="Number of samples to test")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers to use")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument('--use_api', action='store_true', help='use api or not')
    parser.add_argument('--api_key', type=str, default=None, help='explicitly give an api key')
    parser.add_argument("--selected_langs", type=str, default=None, help="list of string of languages")
    parser.add_argument("--selected_levels", type=str, default=None, help="list of string of levels")
    parser.add_argument("--data_path", type=str, default="./data/text-question/", help="path for writing and reading the data")
    parser.add_argument("--model", type=str, default="chat", help="[chat, gpt4, bloom]")
    parser.add_argument("--setting", type=str, default="few-shot", help="[few-shot, zero-shot]")
    parser.add_argument("--method", type=str, default="default", help="[default, en-instruct, en-trans]")
    return parser.parse_args()


def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")


def parallel_query_chatgpt_model(args):
    return query_chatgpt_model(*args)


def parallel_query_gpt4_model(args):
    return query_gpt4_model(*args)


def parallel_query_bloom_model(args):
    return query_bloom_model(*args)


# @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(6), before=before_retry_fn)
@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_chatgpt_model(api_key: str, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 128, temperature: float = 0):
    openai.api_key = api_key
    try:
        completions = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        output = completions.choices[0].message.content.strip()

    except Exception as e:
        # if the error is due to max context length, save such an error
        if "This model's maximum context length is 4097 tokens." in str(e):
            output = "the question is too long"
        else:
            raise e

    return output


# @retry(wait=wait_random_exponential(min=10, max=60), stop=stop_after_attempt(6), before=before_retry_fn)
@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_gpt4_model(api_key: str, prompt: str, model: str = "gpt-4", max_tokens: int = 128, temperature: float = 0):
    openai.api_key = api_key
    try:
        completions = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        output = completions.choices[0].message.content.strip()

    except Exception as e:
        # if the error is due to max context length, save such an error
        if "This model's maximum context length is 4097 tokens." in str(e):
            output = "the question is too long"
        else:
            raise e

    return output


@retry(wait=wait_fixed(10), stop=stop_after_attempt(6), before=before_retry_fn)
def query_bloom_model(api_key, prompt):
    model_url = "https://api-inference.huggingface.co/models/bigscience/bloom"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": f"{prompt}",
        "temperature": 0.0
    }
    try:
        response = requests.post(model_url, headers=headers, json=payload)
        pred = response.json()[0]['generated_text'].strip()
    except Exception as e:
        response_json = response.json()
        # if the error is due to max context length, save such an error
        if "error" in response_json and response_json['error'].startswith('Input validation error: `inputs`'):
            pred = "the question is too long"
        else:
            raise e

    return pred


def generate_one_example(question, lang, method, fill_answer=False):
    answer_word = {'english': "Answer:", 'chinese': '答案：', 'vietnamese': 'Câu trả lời:', 'thai': 'คำตอบ:', 'italian': 'La risposta:',
                   'javanese': 'Wangsulan:', 'swahili': 'Jibu:', 'afrikaans': 'Antwoord:' ,'portuguese': 'Responder:'}
    background = '\n'+'\n'.join(question['background_description']) if question['background_description'] != [] else ''
    if method == 'default':
        prompt = background + '\n' + question['question_text'] + '\n' + '\n'.join(question['options']) + f'\n{answer_word[lang]}'
    elif method == 'en-instruct':
        prompt = background + '\n' + question['question_text'] + '\n' + '\n'.join(question['options']) + f'\nAnswer:'
    elif method == 'en-trans':
        prompt = question['background_description_english'] + '\n' + question['question_text_english'] + '\n' + question['options_english'] + f'\nAnswer:'
    
    if fill_answer:
        prompt += str(question['answer_text'])
    
    return prompt


def generate_dev_examples(dev_questions, lang, method):

    # save the dev examples into a dict, according to their levels and subject categories
    dev_example_dict = defaultdict(lambda: defaultdict(list))
    for q in dev_questions:
        level = q['level']
        cate = q['subject_category']
        dev_string = generate_one_example(q, lang, method, fill_answer=True)
        dev_example_dict[level][cate].append(dev_string)
    
    return dev_example_dict


def generate_prompt(lang, method, setting, model, test_question, dev_question):
    subject2target = {'english': {'language': 'English', 'math': "Math", 'social-science': "Social Science", 'natural-science': 'Natural Science'},
                      'english4all': {'language': 'Language', 'math': "Math", 'social-science': "Social Science", 'natural-science': 'Natural Science'},
                      'chinese':  {'language': '语文', 'math': "数学", 'social-science': "社会科学", 'natural-science': '自然科学'},
                      'javanese': {'language': 'Bahasa Jawa'},
                      'swahili': {'language': 'KISWAHILI'},
                      'thai': {'language': 'ภาษาไทย', 'math': 'คณิตศาสตร์', 'social-science': 'สังคมศึกษา', 'natural-science': 'วิทยาศาสตร์'},
                      'vietnamese': {'language': 'Tiếng Việt', 'math': "Toán", 'social-science': "Khoa học xã hội", 'natural-science': 'Khoa học tự nhiên'},
                      'italian': {'language': 'Italiano', 'math': "Matematica", 'social-science': "Scienze sociali", 'natural-science': 'Scienze naturali'},
                      'afrikaans': {'language': 'Afrikaans Huistaal', 'math': "Wiskunde", 'social-science': "Sosiale Wetenskappe", 'natural-science': 'Natuurwetenskap'},
                      'portuguese': {'language': 'Linguagens', 'math': 'Matemática', 'social-science': 'Ciências Humanas', 'natural-science': 'Ciências da Natureza'},
                      }
    subject = subject2target[lang][test_question['subject_category']]

    # default to use own target language in the prompt/instruction (monolingual setting)
    if method == 'default':
        if lang == 'english':
            hint = f"The following is a multiple choice question about {subject}."
        elif lang == 'chinese':
            hint = f"以下是关于{subject}的单项选择题。"
        elif lang == 'javanese':
            # have different registered of different levels
            if test_question['level'] == 'low':
                hint = "Ing ngisor iki ana pitakon pilihan ganda babagan Bahasa Jawa."
            else:
                hint = "Menika soal pilihan ganda babagan Bahasa Jawa."
        elif lang == 'thai':
            hint = f"ต่อไปนี้เป็นคำถามแบบปรนัย วิชา{subject}."
        elif lang == 'vietnamese':
            hint = f"Sau đây là các câu hỏi trắc nghiệm về {subject}."
        elif lang == 'italian':
            hint = f"Le seguenti sono domande a risposta multipla su {subject}."
        elif lang == 'afrikaans':
            hint = f"Die volgende is veelvuldige keuse vrae oor {subject}."
        elif lang == 'swahili':
            hint = f"Yafuatayo ni maswali ya chaguo nyingi kuhusu Kiswahili."
        elif lang == 'portuguese':
            hint = f"A seguir estão questões de múltipla escolha sobre {subject}."
        else:
            raise NotImplemented
        
        # need to instruct the model to only output the option text
        if model in ['chat', 'fake'] or setting == 'zero-shot':
            if lang == 'english':
                hint += ' Please only give the correct option, without any other details or explanations.'
            elif lang == 'chinese':
                hint += ' 请仅给出正确选项对应的选项序号而非其他细节。'
            elif lang == 'thai':
                hint += ' โปรดระบุคำตอบเป็นตัวเลือกที่ถูกต้องโดยไม่ต้องให้รายละเอียดอื่นเพิ่มเติม.'
            elif lang == 'vietnamese':
                hint += ' Vui lòng chỉ đưa ra phương án đúng, không có bất kỳ chi tiết hay giải thích nào khác.'
            elif lang == 'italian':
                hint += ' Dai solo l\'opzione corretta, senza altri dettagli o spiegazioni'
            elif lang == 'javanese':
                hint += ' Nyuwun paringaken pilihan wangsulan ingkang leres mawon, tanpa detail utawi penjelasan sanesipun.'
            elif lang == 'afrikaans':
                hint += ' Gee asseblief net die korrekte opsie, sonder enige ander besonderhede of verduidelikings.'
            elif lang == 'swahili':
                hint += ' Tafadhali toa chaguo sahihi pekee, bila maelezo yoyote au maelezo.'
            elif lang == 'portuguese':
                hint += ' Por favor, dê apenas a opção correta, sem quaisquer outros detalhes ou explicações.'
            else:
                raise NotImplementedError
    
    # for any language, just use english instructions
    elif method == 'en-instruct' or method == 'en-trans':
        subject = subject2target['english4all'][test_question['subject_category']]
        hint = f"The following is a multiple choice question about {subject}."
        hint += ' Please only give the correct option, without any other details or explanations.'
    else:
        raise NotImplementedError
    
    if setting == 'zero-shot':
        prompt = hint + '\n\n' + generate_one_example(test_question, lang, method)
    elif setting == 'few-shot':
        dev_questions_list = dev_question[test_question['level']][test_question['subject_category']]
        prompt = hint + '\n\n' + '\n\n'.join(dev_questions_list) + '\n\n' + generate_one_example(test_question, lang, method)
    else:
        raise NotImplementedError

    return prompt


def process_lang(args, lang, api_key, selected_levels):

    model = args.model
    method = args.method
    setting = args.setting

    output_folder = f"outputs/{setting}/{method}/model_{model}/{lang}/"
    os.makedirs(output_folder, exist_ok=True)


    # if conduct few-shot settings
    if setting == 'few-shot':   
        dev_file_path = args.data_path + f"{lang}-questions-dev.json"
        if os.path.exists(dev_file_path):
            with open(dev_file_path, "r") as f:
                dev_questions = json.load(f)
            dev_examples = generate_dev_examples(dev_questions, lang, method)
        else:
            raise FileNotFoundError
    else:
        dev_examples = {}


    test_file_path = args.data_path + f"{lang}-questions-test.json"
    # if exists, process this certain file
    if os.path.exists(test_file_path):
        with open(test_file_path, "r") as f:
            test_questions = json.load(f)

        # only take certain number of examples to test
        if args.num_samples != 'all':
            num_samples = int(args.num_samples)
            test_questions = test_questions[:num_samples]
        
        # if only want to test on certain levels
        if len(selected_levels) < 3:
            test_questions = [q for q in test_questions if q['level'] in selected_levels]

        # generate prompts
        all_prompts = []
        
        for question in test_questions:
            prompt = generate_prompt(lang, method, setting, model, question, dev_examples)
            all_prompts.append(prompt)
        
        # inference in batch
        prompt_args = [(api_key, p) for p in all_prompts]
        
        if api_key is not None:
            if args.model == "chat":
                parallel_call = parallel_query_chatgpt_model
            elif args.model == 'gpt4':
                parallel_call = parallel_query_gpt4_model
            elif args.model == "bloom":
                parallel_call = parallel_query_bloom_model
            else:
                raise NotImplementedError
        
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                predictions = list(tqdm(executor.map(parallel_call, prompt_args), total=len(prompt_args), desc=f"Conducting inference"))

        else:
            # generate fake answers for checking the prompt only
            predictions = ['fake'] * len(prompt_args)

        # save the predictions
        for idx, question in enumerate(test_questions):
            question[model+'_pred'] = predictions[idx]    # save the pred
            question['prompt'] = all_prompts[idx]         # also save the prompt
        
        with open(f"{output_folder}/{lang}-pred.json", "w") as f:
            json.dump(test_questions, f)
        
        print(f"Done: {len(test_questions)} {lang} questions!")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    all_langs = ['english', 'chinese', 'afrikaans', 'italian', 'javanese', 'thai', 'vietnamese', 'portuguese', 'swahili']
    selected_langs = eval(args.selected_langs) if args.selected_langs else all_langs
    selected_levels = eval(args.selected_levels) if args.selected_levels else ['low', 'mid', 'high']

    # read in the api key
    api_key = args.api_key

    for lang in selected_langs:
        process_lang(args, lang, api_key, selected_levels)


if __name__ == "__main__":
    main()