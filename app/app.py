import os

import streamlit as st
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM
from scripts.generate import generate
from transformers import AutoTokenizer
from validation.Inputs import Inputs


def app_init():
    # Загрузка переменных из .env файла
    load_dotenv()
    # Чтение токена
    token = os.environ["HUGGING_FACE_ACCESS_TOKEN"]
    # Авторизация в Hugging Face
    login(token)
    # Устройство для тензорных вычислений и хранения модели в памяти.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return token, device


token, device = app_init()


# Загрузка и кэширование ML модели
@st.cache_resource()
def load_model():
    model_name = "nymless/gemma-2-2b-lora-finetuned-30k"

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        token=token,
    )
    model = model.eval().to(device)
    return model, tokenizer


model, tokenizer = load_model()

# Заголовки
st.title("Review Generator")

# Ввод параметров поездки
address = st.text_input("Адрес")
name = st.text_input("Название")
rating = st.text_input("Рейтинг")
rubrics = st.text_input("Рубрики")

# Получены все данные
all_inputs_received = address and name and rating and rubrics

# Кнопка расчёта нажата
button_pressed = st.button("Генерация отзыва")

if button_pressed and all_inputs_received:
    inputs = None

    # Валидация
    inputs = Inputs(
        address=address,
        name=name,
        rating=rating,
        rubrics=rubrics,
    )

    # Генерация
    review = generate(model, tokenizer, inputs, max_length=200, device=device)
    st.text(review)
