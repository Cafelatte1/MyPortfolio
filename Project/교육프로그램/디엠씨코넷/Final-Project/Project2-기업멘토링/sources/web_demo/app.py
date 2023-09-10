GLOBAL_SEED = 42

import os
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)

import sys
from datetime import datetime
import threading
import random as rnd
from numpy import random as np_rnd

from flask import Flask, render_template, request, jsonify
from pyngrok import ngrok
from time import time
import gc
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from utils.post_processing import PostGenerationStageProcesser

# === Flask configuration ===
app = Flask(__name__)
port = 5000
# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(port).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))
# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

# === model configuration
# config on model for quantization
bnb_config = BitsAndBytesConfig(
    # 모델을 4bit로 로딩하도록 설정합니다
    load_in_4bit=True,
    # double quantization 모드를 활성화합니다 (weight 저장과 계산을 다른 타입으로 할 수 있게 합니다)
    bnb_4bit_use_double_quant=True,
    # double quantization 모드에서 저장될 4bit 데이터 타입을 지정합니다
    bnb_4bit_quant_type="nf4",
    # double quantization 모드에서 계산에 사용할 데이터 타입을 지정합니다
    bnb_4bit_compute_dtype=torch.bfloat16,
    # set device
    device_map="auto",
)
model_root_path = "/content/drive/MyDrive/Colab Notebooks/architecture/kullm_12.8b_5주차_v3_2023-09-02_08-42_case0/"
# # debug
# model_root_path = "/content/drive/MyDrive/Colab Notebooks/architecture/polyglot_1.3b_5주차_v1-2_2023-08-30_11-34_case0/"
tokenizer = AutoTokenizer.from_pretrained(model_root_path + "tokenizer/", padding_side="left")
config_tokenizer = {
    "max_length": 256,
    "padding": "max_length",
    "truncation": True,
    "return_token_type_ids": False,
    "return_tensors": "pt",
}
model = AutoModelForCausalLM.from_pretrained(model_root_path + "model/", quantization_config=bnb_config)
model.eval()
model.config.use_cache = True
inference_fixed_params = {
    "max_new_tokens": 128,
    "num_beams": 1,
    "early_stopping": False,
    "do_sample": True,
    "temperature": 0.01,
    "top_k": 50,
    "use_cache": True,
    "pad_token_id": tokenizer.eos_token_id,
}

postprocessor = PostGenerationStageProcesser(
    # control similarity threshold with previous sentences
    # thr lower, the target sentence has higher probability to be dropped
    edit_distance_threshold=0.5,
    # control duplicated urls
    max_url_size=2,
    eval_rouge_n=[1, 2, 3],
    stopwords_path="./utils/post_generation_stopwords.txt",
    valid_urls_path="./utils/post_generation_urls.txt",
)

def do_inference(params, model, prompt):
    start_time = time()
    seed_everything(int(datetime.now().timestamp()))

    # generating
    response = []
    with torch.no_grad():
        gened = model.generate(
            **{"input_ids": prompt["input_ids"].to(device), "attention_mask": prompt["attention_mask"].to(device)},
            **params,
        )
    response.extend(tokenizer.batch_decode(gened, skip_special_tokens=True))
    del gened
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time()
    # decoding & return
    output = {
        "response": response,
        "inference_runtime": round(end_time - start_time, 3),
        "inference_runtime_per_sample": round((end_time - start_time) / len(prompt), 3),
    }

    return output

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # python random
    rnd.seed(seed)
    # numpy random
    np_rnd.seed(seed)
    # RAPIDS random
    try:
        cupy.random.seed(seed)
    except:
        pass
    # tf random
    try:
        tf_rnd.set_seed(seed)
    except:
        pass
    # pytorch random
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass

@app.route('/', methods=['GET', 'POST'])
def index():
    output = {'generated_text': ""}
    if request.method == 'POST':
        input_text = request.form['input-text']
        if (input_text == "") or (input_text is None):
            output = {'generated_text': ""}
        else:
            start_time = time()
            input_text = f"### 질문: {input_text.strip()}\n\n### 답변:"
            input_text = tokenizer(input_text, **config_tokenizer)
            gened_text = do_inference(inference_fixed_params, model, input_text)
            before_pruning_text = gened_text["response"][0]
            before_pruning_text = "=== Before pruning ===\n" + before_pruning_text
            gened_text = postprocessor.processing(gened_text["response"][0], reference_text=None)
            after_pruning_text = gened_text[2]
            after_pruning_text = "\n\n=== After pruning ===\n" + after_pruning_text
            inference_time_text = "\n\nInference time (sec) -> " + str(round(time() - start_time, 3))
            output = {'before_pruning_text': before_pruning_text, 'after_pruning_text': after_pruning_text, "inference_time": inference_time_text}
        return render_template('index.html', **output)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()





