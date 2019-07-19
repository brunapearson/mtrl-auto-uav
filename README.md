# Multi-Task Regression-based Learning for Autonomous Unmanned Aerial Vehicle Flight Control within Unstructured Outdoor Environments



**Abstract** *— Increased growth in the global Unmanned Aerial Vehicles (UAV) (drone) industry has expanded possibilities for fully autonomous UAV applications. A particular application which has in part motivated this research is the use of UAV in wide area search and surveillance operations in unstructured outdoor environments. The critical issue with such environments is the lack of structured features that could aid in autonomous ﬂight, such as road lines or paths. In this paper, we propose an End-to-End Multi-Task Regression-based Learning approach capable of deﬁning ﬂight commands for navigation and exploration under the forest canopy, regardless of the presence of trails or additional sensors (i.e. GPS). Training and testing are performed using a software in the loop pipeline which allows for a detailed evaluation against state-of-the-art pose estimation techniques. Our extensive experiments demonstrate that our approach excels in performing dense exploration within the required search perimeter, is capable of covering wider search regions, generalises to previously unseen and unexplored environments and outperforms contemporary state-of-the-art techniques.*

Tested using: [Anaconda 2018.12](https://www.anaconda.com/distribution/) | [Python 3.7.3](https://www.python.org/downloads/release/python-373/) | [Keras 2.2.4](https://pypi.org/project/Keras/) | [Tensorflow 1.13.1](https://www.tensorflow.org/install/pip) | [OpenCV 3.4.2](https://pypi.org/project/opencv-python/)

## Network Architecture:

![MTRL Network](https://github.com/brunapearson/mtrl-auto-uav/blob/master/images/fig2.jpg)

---

## Prerequisites and setup


### Environment Setup

1. Install [AirSim](https://github.com/microsoft/AirSim).
2. Install [Anaconda with Python 3.5 or higher](https://www.anaconda.com/distribution/).
3. Create an Anaconda environment (preferable using Python 3.7).
4. Install Keras with Tensorflow backend.

```
$ conda install keras-gpu
```

5. Install additional dependencies:
   * matplotlib v.
   * image
   * opencv
   * msgpack-rpc-python
   * pandas
   * numpy
   * scipy
   * h5py

```
$ pip install opencv-python h5py matplotlib image pandas msgpack-rpc-python
```

---

### Simulator 

**n_predictions[n]**

Here, you can define how many *predictions* should be computed by the network. At each iteration, one set of predicted values for waypoints (x,y,z) and orientations are outputted.

**behaviour [search,flight]**

Here, the behaviour of the UAV can be defined either as *search* or *flight*. In the *search* mode, the UAV navigational change in *x* and *y* directions are increased, which results in a wider angular rotation of the head. In contrast, when the behaviour is set to *flight* the predicted values are smoothed, which reduces the angular rotation of the head. During the production of this paper, all tests were carried using the *search* mode.

**start position [-100,100]**

Here, the starting position of the drone can be defined by assigning a value for the *x*,*y*,*z* coordinates. We recommend these value to be between -100 and 100.

**smoothness [-2.0,2.0]**

When using the flight mode, you can also define the smoothness of the flight in the *x*,*y* and *z* directions. We recommend using values between -2.0 and 2.0.

```
$ python test_mtrl.py n_predictions behaviour x y z smoothness_x smoothness_y smoothness_z
```
Example:
```
$ python test_mtrl.py 150 search -3 -10 -10 0.75 -0.75 0.15
```

---
### Pre-trained Model:
* To provide better testing opportunities, we provide a set of pre-trained weights. The pre-trained are separately stored on [Zenodo](https://zenodo.org/record/3338078#.XTGBO6Yo8UG) due to their large size.  

* The script entitled "download_model.sh" will download the pre-trained weights and check the file integrity.

* To download the pre-trained model, run the following commands:

```
$ chmod +x ./download_model.sh
$ ./download_model.sh
```

### Reference:
If you make use of this work in any way (including our [pre-trained models](https://zenodo.org/record/3338078#.XS32h-hKguU) or [dataset](https://zenodo.org/record/3270774#.XS32sehKguU), please reference the following:

Multi-Task Regression-based Learning for Autonomous Unmanned Aerial Vehicle Flight Control within Unstructured Outdoor Environments 
[Maciel-Pearson et al., In Robotics and Automation Letters IEEE, 2019]( ).

```
@article{pearson19regression,
  title={Multi-Task Regression-based Learning for Autonomous Unmanned Aerial Vehicle Flight Control within Unstructured Outdoor Environments},
  author={Maciel-Pearson, B.G., Akçay, S., Atapour-Abarghouei, A., Holder, C. and Breckon, T.P.},
  journal={IEEE Robotics and Automation Letters},
  volume={x},
  pages={1-8},
  year={2019},
  publisher={IEEE}
}
```

### TO-DO
- [ ] write installation/setup instructions, including conda container
