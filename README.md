# Wikipedia Articles Classification

## Contents

- **notebook/report_wikipedia.ipynb** jupyter notebook with my analysis and tests
- **Scraper.py** class to scrape Wikipedia articles
- **Classifier.py** class to perform training and predictions using a pertained BERT model
- **API.py** Flask API
- **requirements.txt** Python requirements
- **Dockerfile** Docker instructions to build the image
- **supervisord.conf** Supervisor configuration file to expose multiple services in a Docker image
- **data** Directory to store model weights and dataset
- **project.pdf** Technical presentation

## Docker

To *build* the Docker image please execute the following command:

```bash
docker build -t classifier -f Dockerfile .
```

To *run* the Docker image please execute the following command:

```
docker run -p 8888:8888 -p 5000:5000 classifier
```

## Notebook

Once the Docker image is running you can find the Jupyter notebook at the following adresss:

* [Report](http://localhost:8888/notebooks/report_wikipedia.ipynb)
* [Jupyter Home Page](http://localhost:8888/)


## API

The Flask API can be used by calling http://0.0.0.0:5000/COMMAND. As asked in the assignment the API contains two endpoints:

- One for scraping the data and training the model (http://0.0.0.0:5000/train). The call will start the scraping and then the training of the model. This endpoint returns the best validation accuracy achieved during the training.
  
	**This process is quite slow (around 20 minutes for scraping and 5 minutes for training)**. 
	
- One for querying the last trained model with an article of our choice in the dataset (http://0.0.0.0:5000/classify/id where the id is the `x` of the dataset). This endpoint return the prediction and true label if the given id is valid otherwise it returns an error message.
  
  To speed up the process (scraping and training) and test the evaluation, you can donwload a pretrained model and dataset from [Google Drive](https://drive.google.com/drive/folders/1UAPXy8IIZOWX9XbDJm9nw8ferBf5ftAp?usp=sharing) and place the two file into the data directory. 
  
  To test this last entripoint, you can execute the `test.py` file. This file executes a simple call to classify an article and then it prints the result.
  
  