FROM python:3.7

RUN apt-get update
RUN apt-get install -y supervisor

EXPOSE 5000
EXPOSE 8888

RUN mkdir /usr/src/app/

COPY requirements.txt /usr/src/app/
WORKDIR /usr/src/app/

RUN pip install -r requirements.txt

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

COPY . /usr/src/app/

CMD ["/usr/bin/supervisord"]