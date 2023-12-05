FROM python:3.11-slim

ENV PORT 8000

COPY requirements.txt /
RUN pip install -r requirements.txt
RUN python -m spacy download es_core_news_md
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('karina-aquino/spanish-sentiment-model')"
RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ./text-sentiment-recognizer /text-sentiment-recognizer
CMD uvicorn eldeberanalyzing.app:app --host 0.0.0.0 --port ${PORT}