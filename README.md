# Predicting Anomalous Spatio-Temporal Traffic Patterns with Meta-Learning

## Overview

This repository houses the implementation of a meta-learning framework designed to enhance the performance of spatio-temporal predictive models on anomalous traffic data, such as those observed during the COVID-19 pandemic. The project uses a multi-input convolutional neural network (CNN) that incorporates meta-learning to adapt and predict these unusual traffic patterns effectively.

Our work extends the capabilities of traditional traffic prediction models, focusing particularly on the integration of historical traffic data to understand underlying structures that aid in predicting anomalous conditions.

## Dataset

The raw traffic data for this project is derived from New York City's open-source Citi Bike dataset. The dataset includes detailed biking trip records, which have been preprocessed into a 2x16x8 tensor format representing inflow and outflow in a grid overlay of Manhattan.

### Getting the Data

1. Navigate to the `raw_traffic_data` folder in this repository.
2. Download the pre and post-pandemic traffic datasets from the provided OneDrive link.
3. Place the downloaded files in the `raw_traffic_data` folder.

For those interested in using a different dataset from the same source, you can obtain additional data sets from [NYC Trip Data](https://s3.amazonaws.com/tripdata/index.html). Use the `raw_data_processing.ipynb` notebook to preprocess this raw data into the required tensor format.

## Running the Code

To run the main predictive model:
1. Ensure that the dataset is correctly placed in the `raw_traffic_data` folder.
2. Open and run the `main.ipynb` notebook in a Jupyter environment.

This notebook will guide you through the process of using the multi-input CNN model to predict traffic patterns.

## Acknowledgments

Special thanks to Professors Bing Wang and Suining He, and to all contributors and authors whose works have inspired and made this research possible. Their insights into meta-learning and spatio-temporal prediction have been invaluable.

## License

This project is open-sourced under the MIT license. See the LICENSE file for more details.
