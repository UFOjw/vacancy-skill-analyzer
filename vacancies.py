import requests
from bs4 import BeautifulSoup
from llama_cpp import Llama
import re
import pandas as pd
from tqdm import tqdm
import ast

llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=6048,
    n_threads=6,
    n_gpu_layers=35,
    verbose=False,
)

# Функция для получения текста по ссылке
def extract_text_from_url(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=1)

        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Удалим скрипты и стили
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        print(f"[ERROR] Failed to process {url}: {e}")
        return ""

# Выделение из страниц информации
def extract_topics(text: str) -> str:
    prompt = f"""[INST] Ты — эксперт по анализу вакансий. На вход ты получаешь текст вакансии от работодателя. Твоя задача:

Выдели только ключевые требования и технологии, которые работодатель ожидает от кандидата. Сформируй их **в одну строку**, перечисляя через запятую.

Включи:
- стек технологий: языки программирования, библиотеки, фреймворки, базы данных, инструменты;
- дополнительные требования: опыт в годах, уровень английского, знание методологий, диплом, область знаний и т.п.

⚠️ Не пиши объяснений, заголовков, списков, абзацев — только строку со сжатыми требованиями.

Ответ строго оберни в:
<answer>
[строка через запятую]
</answer>

Текст вакансии:
{text[:4000]} 
[/INST]"""
    result = llm(prompt, max_tokens=728, temperature=0.0, stop=["</s>"]) ### Макс аутпут модели + текст описания вакансии
    return result["choices"][0]["text"].strip()

def extract_filtered_skills(text: str) -> list:
    prompt = f"""
У тебя есть категории навыков:
{categories}

Теперь выбери из следующего списка **только те навыки**, которые попадают хотя бы в одну из этих категорий:

Skills: {text}

Верни ответ строго в виде **списка навыков** через запятую, **без пояснений и без категорий**. Например:
Python, Git, Deep Learning, English

Правила:
- Не меняй названия навыков.
- Не группируй по категориям.
- Не добавляй никакого дополнительного текста, только перечисление подходящих навыков через запятую.

Ответ:
"""
    result = llm(prompt, max_tokens=512, temperature=0.0, stop=["</s>"])
    skills_raw = result["choices"][0]["text"].strip()
    return [s.strip() for s in skills_raw.split(',') if s.strip()]

categories = """
Классическое ML (регрессия, классификация, кластеризация)
Продвинутые методы ML: бустинг, SVM, Random Forest, L1/L2 регуляризация
Временные ряды, анализ сигналов, детекция выбросов
Scikit-learn, CatBoost, XGBoost, LightGBM, GLM
Feature Engineering, Optuna, GridSearch, кросс-валидация
Оценка моделей: accuracy, precision, recall, F1, ROC-AUC
Продакшн и внедрение моделей, автоматизация пайплайнов
ML-платформы: MLflow, DVC, ClearML, Triton Inference Server, ONNX
Обработка табличных данных, лог-анализ, фильтрация

PyTorch, TensorFlow, Keras
LSTM, RNN, Transformers, BERT, GPT
Fine-tuning, LoRA, Prompt Tuning, Quantization
Нейронные сети, понимание архитектур, pruning, DDP/DeepSpeed/FSDP

LLM: GPT, Claude, LLaMA, Mistral
LangChain, AutoGEN, CrewAI, Hugging Face Transformers
Text Classification, NER, Summarization, RAG, Semantic Search
Sentence Transformers, spaCy, NLTK
Структуры: позиционные энкодинги, attention, embedding models

OpenCV, torchvision, timm, Albumentations
YOLO, Detectron2, MMDetection
Обработка изображений: img2img, IPAdapter, ControlNet, CFG
Метрики: IoU, mAP, BLEU, ROUGE
CV инструменты: CVAT, Supervisely, Label Studio
Stable Diffusion, ComfyUI

A/B-тесты, проверка гипотез, генерация гипотез
Метрики и аналитика: построение, интерпретация
Uplift modeling
BI-инструменты

SQL
ClickHouse, PostgreSQL, Oracle, GreenPlum, Hive, Spark, HDFS, Hadoop
Pandas, NumPy, PySpark
Работа с сырыми данными, лог-анализ, геоданные, графы

Docker, Kubernetes, Airflow, ML pipeline, CI/CD
MLOps: мониторинг, жизненный цикл моделей, продакшн-поддержка
DevOps: Kafka, TeamCity, Dagster, Nexus, GitHub, Bitbucket

FAISS, Chroma, Qdrant, Weaviate, Milvus
Vector Search, Recall@k, BM25
Knowledge Graph, embedding models (OpenAI, Hugging Face)

Теория вероятностей, мат. статистика, мат. анализ, линейная алгебра
Оптимизационные методы
Tervеr/matstat, технический анализ

Опыт 1го, опыт 3года
Участие в Kaggle, публикации, GitHub-портфолио

Прочее
"""

urls = [
    ### добавьте свои сайты
]

### Обработчик
answers = []

for url in urls:
    print(f"\n🔗 URL: {url}")
    text = extract_text_from_url(url)
    if text:
        output = extract_topics(text)
        print("📌 Ключевые темы:\n", output)

        match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            answers.append(answer)
            print("✅ Извлечено из <answer>:\n", answer)
        else:
            print("❌ <answer> не найден")

df = pd.read_excel("vac.xlsx") # Если есть старые описания
answers_df = pd.DataFrame({"parsed": answers})

df = pd.concat([df, answers_df], axis=0).reset_index(drop=True)
df.to_excel("vac.xlsx", index=False)

"""
### Для поиска категорий (обновления categories)
combined_list = []

for line in df["parsed"]:
    tokens = [item.strip() for item in line.split(",")]
    combined_list.extend(tokens)

# Удалим дубликаты, если нужно:
combined_list = list(dict.fromkeys(combined_list))
print(combined_list)
"""

# Пройтись по всем строкам и сохранить отфильтрованные навыки
df['filtered_skills'] = None

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering relevant skills"):
    skills_text = row['parsed']
    try:
        filtered = extract_filtered_skills(skills_text)
        df.at[idx, 'filtered_skills'] = filtered
    except Exception as e:
        print(f"Ошибка в строке {idx}: {e}")
        df.at[idx, 'filtered_skills'] = []

skills = {}
for row in df.iterrows():
    for skill in ast.literal_eval(row[1]['filtered_skills']):
        if skill not in skills:
            skills[skill] = 0
        skills[skill] += 1

print({k: v for k, v in sorted(skills.items(), key=lambda item: item[1])})