# Antarstick
## Brief description
A collection of methods/functions for analyzing snow in pictures from **Antarctica**. This will be used in a visualization tool
that is being developed for scientists from the **Mendel Polar Station**.

In each photo there is a number of bambus sticks sticking from the ground. The goal is to measure the height of snow cover
from each photo by measuring how much each stick is covered by snow. 

## Rough outline
### 1. Initialization
- [x] 1. Find a photo without snow
- [x] 2. Detect sticks
- [ ] 3. Ability to identify the same stick seen from different cameras

### 2. Individual photo analysis of snow (perform repeatedly until all photos processed)
- [ ] 1. Detect misalignment of sticks from previous photo and take it into account
- [x] 2. Measure snow height for each stick

### 3. Analysis of measurements
- [ ] 1. Detect uncertain/inaccurate/outlier measurements and let the user correct them

## Language used
[Python 3.8.x](https://www.python.org/)

## Packages used
- [Opencv wrapper](https://pypi.org/project/opencv-python/)
- [Jsonpickle](https://pypi.org/project/jsonpickle/)
- [Numpy](https://pypi.org/project/numpy/)
