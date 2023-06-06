#FROM python:3.9
#
#WORKDIR /app
#
#RUN pip3 install pandas scikit-learn flask gunicorn
#
#ADD model.pkl model.pkl
#ADD server.py server.py
#
#EXPOSE 8080
#
#CMD [ "gunicorn","--bind","0.0.0.0:8080","server:app" ]

#FROM python:3.9
#
#RUN pip install joblib pandas scikit-learn flask
#
#COPY train.csv ./train.csv
#COPY test.csv ./test.csv
#
#COPY train.py ./train.py
#COPY inference.py ./inference.py
#
#RUN python3 train.py

FROM python:3.9

COPY server.py ./server.py
COPY rf.pkl ./rf.pkl

EXPOSE 8888

# WORKDIR /Notebook

RUN pip --no-cache-dir install \
    flask \
    pickle5 \
    scikit-learn \
    numpy \
    pandas \
    flasgger==0.9.5

CMD python3 server.py