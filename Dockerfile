FROM python:3.10.12

WORKDIR /app

COPY requirements.txt .

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY . .
CMD ["python","gradio_app.py"]

