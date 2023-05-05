# Regression Model

This repository contains the example and sample files to deploy a Regression Model on Katonic Platform.

## Prerequisites for Deployment:


- `launch.py`: This file consists of `loadmodel`, `preprocessing` and `predict` functions.
 The first function helps to fetch the model. The second function is optional,if you are performing any kind of preprocessing on the data before prediction please add all the necessary steps into it and return the formatted input, else you can just return `False` if no processing is required. In the third function write down the code for prediction and return the results in the data structure supported by API response.   

- `schema.py`: Define your schema on how you should pass your input data in predict function.

- `requirements.txt`: Define the required packages along with specific versions this file.

## Sample Input Data for Prediction API

```python
{
    "data":[[4.000e+00, 9.000e+01, 7.500e+01, 2.125e+03, 1.450e+01, 7.400e+01,
        1.000e+00],
       [4.000e+00, 1.400e+02, 8.800e+01, 2.890e+03, 1.730e+01, 7.900e+01,
        1.000e+00]]
}
```

## Sample Input Data for Feedback API

```python
{
  "predicted_label":[27.39816903, 25.99190419, 10.61215193, 32.8755205 , 32.13724679,23.17668144, 33.34045739, 24.97687414, 21.11993012, 26.51062283],
  "true_label": [31.39816903, 25.99190419, 8.61215193, 32.8755205 , 35.13724679, 21.17668144, 30.34045739, 21.97687414, 21.1, 25.51062283]
}
```
