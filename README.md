# Cat-Facial-Detection
![facial landmarks sample](https://raw.githubusercontent.com/BruceMacD/Cat-Facial-Detection/master/data/sample_output.png)

:cat: Detecting and mapping cat faces.


## Usage
detect_cat_faces.py -i <path/to/inputCatImg.jpg>

```
detect_cat_faces.py -i data/mimi.jpg
```

## Requirements
* OpenCV v3.0+
* numpy
* dlib (for the landmark detection)
* Python 3

## Sources
Cat facial landmark detection model

https://www.kaggle.com/aakashnain/cats-everywhere

Cat data set

https://www.kaggle.com/crawford/cat-dataset

Detecting cats in images with OpenCV

https://www.pyimagesearch.com/2016/06/20/detecting-cats-in-images-with-opencv/

Rapid Object Detection using a Boosted Cascade of Simple Features

https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf

## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
