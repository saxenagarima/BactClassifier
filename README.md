# BactClassifier

The program identifies following bacterial species:

* Acinetobacter baumanii
* Acinetomyces Israeli
* Bacteroides fragilis
* Bifidobacterium
* Candida albicans

Features were extracted using Inception V3 (trained on ImageNet).
These extracted features were used to train an SVM model.

## Usage

Build Docker:
```shell
docker build -t bact .
```

Start Docker:
```shell
docker run -it -p 8000:8000 bact
```

Run Jupyter notebook:
```shell
jupyter notebook --ip 0.0.0.0 --port 8000 --no-browser --allow-root
```

Run Script"
```shell
python3 main.py 
```
Uses the default data folder for training and default test data folder for testing.

It will generate features.csv file, containing features.
This file is then used to train SVM classifier which gets saved as svmClassifier.pkl

To skip generation of features.csv and avoid training of SVM classifier, i.e only test images, do:
```shell
python3 main.py --useDumpedFeature --useSavedModel
```

To pass your own custom test data use:
```shell
python main.py --useDumpedFeeature --useSavedModel -testData [Path to test data] 
```

To train on a custom dataset, pass:
```shell
python main.py -data [Path to training data]
```

## License
 
The MIT License (MIT)

Copyright (c) 2020 Garima Saxena

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
