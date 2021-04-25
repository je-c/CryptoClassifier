<p align="center">
  <a href="https://example.com/">
    <img src="https://via.placeholder.com/72" alt="Logo" width=72 height=72>
  </a>

  <h3 align="center">Trade Decision Classifier</h3>

  <p align="center">
    This repository houses a collection of code repsonsible for data collection, storage and processing with the intent of feeding a convolutional neural network (ResNet9 architecture) in order to make buy/sell/hold decisions for a given cryptocurrency using images comprised of technical indicators over vaired timescales
    <br>
  </p>
</p>


## Table of contents

- [Quick start](#quick-start)
- [Status](#status)
- [What's included](#whats-included)
- [Bugs and feature requests](#bugs-and-feature-requests)
- [Contributing](#contributing)
- [Creators](#creators)
- [Thanks](#thanks)
- [Copyright and license](#copyright-and-license)


## Quick start

This is a pseudo-self-contained 'application' designed to be deployed on a cloud hosted server, though can be run locally in a virtual environment. As such, all required processes are executed behind the scenes by 'main.py'.

## Structure

The project is structured in a series of sub-packages contained in `CryptoClassifier/lib` pertaining to function. Functionality contains data ingestion, processing and the required neural network architecture. Additionally, host device availability recognition is implemented such that CUDA can be leveraged given the specifications of the host environment. 

Storage contains modules concerned with data storage, both locally and in either a local or hosted relational database. Natively, the application supports PostGreSQL and will automatically detect when new data has been retrieved and append it to the database.

Additional directories containing .json files that store application parameters and database credentials, previous datasets (in both csv and unpacked .png form), and previously trained model states are also contained in `CryptoClassifier/lib` and, if detected, are called when run in a deployment setting to make predictions on new data.

```text
CryptoClassifier/
  ├── deploy.py
  ├── train.py
  ├── main.py
  └── lib/
      ├── functionality/
      │   ├── data.py
      │   ├── device.py
      │   ├── model.py
      │   ├── processing.py
      │   └── visualisation.py
      ├── storage/
      │   ├── pipeline.py
      │   └── sql.py
      ├── saves/
      │   └── modelstate.pth
      ├── json/
      │   ├── parameters.json
      │   └── _credentials/
      │       └── classifierDB.json
      └── datasets/
          └── 04-23-21.csv
```

## Copyright and license

Code released under the [Apache Licence](https://github.com/je-c/CryptoClassifier/blob/main/LICENSE).
