FROM selenium/standalone-chrome

WORKDIR /app

COPY . .

# RUN ls -la /app

USER root
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

USER seluser

CMD ["python3", "SummarizeArticles.py"]