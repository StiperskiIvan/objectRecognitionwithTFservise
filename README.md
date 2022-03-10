# objectRecognitionwithTFservise

To use the app first you need to run the tf service 
This can be done using docker, first you need to install a docker if you don't have it on your OS
after that you can type in your terminal 
```docker pull tensorflow/serving```

then you need to run the docker container and mount your model that you want to serve:

```docker run -p 8501:8501 --name tfserving_classifier --mount type=bind,source=/Users/objectRecognitionwithTFservise/img_classifier/,target=/models/img_classifier \```
```-e MODEL_NAME=img_classifier -t tensorflow/serving```


source=/Users/objectRecognitionwithTFservise/img_classifier/ - this is the path to your model on your Computer, be sure to point to the folder, not the model file

after yor tf server is running you need to install all the libraries with command ```pip install -r requirements.txt``` in another terminal

after that you can run the app by typing ```python3 app.py```

You can open your favorite browser then and go to localhost:5000 and use the app

