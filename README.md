# Machine-Learning

Learning new things is always wonderful, and teaching what we understood to others is another challanges in itself to improve our understanding related to the topics that we love. Here, i will explain what i learned and improve my understanding further related to the topcis.


Topics: 


### Pipelines

Utilizing pipeline in building model was considered as an advanced method to make the code:
1. Clear code 
2. Easier to understand
3. Easier to debug


================================================================================

While making the model without pipeline is doable, that in itself might be harder for other people than the creator to implement, while a model with pipeline can be implemented with other easily and they can improve the model necessarily.

Materials related to pipeline that i read in kaggle, briefly explain how the pipeline works in creating the model.

There are plenty ways to make pipeline, while for this example, i will use what has been provided in kaggle course for **Intro to Programming**

The model that will be used is *RandomForestRegressor* utilizing MelbourneHousingDataset. 

```
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
```
Some important library to note, is it utilize scikit-learn library that utilizing:
1. ColumnTransformer - To grouped the pipeline
2. Pipeline - To make the modeling steps concise
3. SimpleImputer - This library will impute any missing number in dataset
4. OneHotEncoder - To Encode any categorical / object data contain in the dataset.

================================================================================

```
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')
```
This code declare a model named *numerical_transformer* that contain *SimpleImputer* model with strategy variable being 'constant'. This model named *numerical_transformer* will replace any numerical missing value in the data, and utilizing constant value in the data to replace the missing value.

================================================================================

```
# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```
This code declare a model named *categorical_transformer* that utilize module named *Pipeline*, with the first steps is *SimpleImputer* variable 'strategy' being'most_frequent' and *OneHotEncoder* variable 'handle_unknown = ignore'. 

The model utilize *Pipeline*, where the first steps is to handle any missing value and replace with 'most_frequent' and to encode the non-numeric data into numerical value.

This means, the *SimpleImputer* here utilize different strategy, where it will replace the missing value for the non-numeric data with 'most_frequent' data
And for the *OneHotEncoder*, it will encode non-numeric data into numerical value for easier machine interpretation.

================================================================================

```
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```

This code declare a model named *preprocessor* that will preprocess each column in the dataset separately with its own respective transformers. The first being is using *numerical_transformer* that will preprocess every column that was assigned as *numerical_cols*, and the *categorical_transformer* will process any column that was assigned as *categorical_cols*.

================================================================================

Next, creating the model that will be used. The model in the example will utilize *RandomForestRegressor* based on the used dataset.

```
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
```

The code declare on how the *RandomForestRegressor* is made with utilizing 100 estimator. *RandomForestRegressor* utilize *non-supervised* learning, and it was a learning method in Machine Learning, where the model will be trained with non-structured data, and to predict any continuous value. Whereas, we will use it to train with this dataset.

================================================================================

```
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
```
Next, we define a model named *my_pipeline* utilizing Pipeline for preprocess the dataset and create the model. The steps in this *Pipeline* utilize model that has been created previously to preprocess necessary dataset and create necessary model.

================================================================================

```
# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```

Then training the created *Pipeline* into the training and testing to measure the model performance.

Overall, this is how the concept of *Pipeline* that can be utilized to make the structure of the code easier to understand and will make the next contributor to make the code more concise easier to understand.

