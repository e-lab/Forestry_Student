FROM python:3.9-slim

WORKDIR /code 

RUN apt-get update

COPY . . 

RUN pip install -r /code/requirements.txt

EXPOSE 8501

CMD bash streamlit.sh