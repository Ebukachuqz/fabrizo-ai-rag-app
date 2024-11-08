FROM python:3.10-slim

ENV STREAMLIT_PORT=8501
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE ${STREAMLIT_PORT}

CMD ["streamlit", "run", "app.py", "--server.port=${STREAMLIT_PORT}", "--server.address=0.0.0.0"]
