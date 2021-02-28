FROM python:3.7-slim
ENV PYTHONUNBUFFERED 1
RUN mkdir /src
WORKDIR /src
#COPY requirements.txt /code/
RUN pip install jupyter

CMD ["jupyter", "notebook", "--no-browser", "--allow-root", "--port=8080", "--ip=*"]

