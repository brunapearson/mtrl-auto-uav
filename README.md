# Multi-Task Regression-based Learning for Autonomous Unmanned Aerial Vehicle Flight Control within Unstructured Outdoor Environments



**Abstract** *— Increased growth in the global Unmanned Aerial Vehicles (UAV) (drone) industry has expanded possibilities for fully autonomous UAV applications. A particular application which has in part motivated this research is the use of UAV in wide area search and surveillance operations in unstructured outdoor environments. The critical issue with such environments is the lack of structured features that could aid in autonomous ﬂight, such as road lines or paths. In this paper, we propose an End-to-End Multi-Task Regression-based Learning approach capable of deﬁning ﬂight commands for navigation and exploration under the forest canopy, regardless of the presence of trails or additional sensors (i.e. GPS). Training and testing are performed using a software in the loop pipeline which allows for a detailed evaluation against state-of-the-art pose estimation techniques. Our extensive experiments demonstrate that our approach excels in performing dense exploration within the required search perimeter, is capable of covering wider search regions, generalises to previously unseen and unexplored environments and outperforms contemporary state-of-the-art techniques.*

Tested using: Anaconda 2018.12 | Python 3.7.3 | Keras 2.2.4 | Tensorflow 1.13.1 | cv2 3.4.2

## Network Architecture:

![MTRL Network](https://github.com/brunapearson/mtrl-auto-uav/blob/master/images/fig2.jpg)

## Prerequisites and setup


### Environment Setup

1. Install [AirSim](https://github.com/microsoft/AirSim).
2. Install [Anaconda with Python 3.5 or higher](https://www.anaconda.com/distribution/).
3. Install Keras with Tensorflow backend.
4. Install h5py.
5. Install additional dependencies:
   * matplotlib v.
   * image
   * opencv
   * msgpack-rpc-python
   * pandas
   * numpy
   * scipy

### Simulator 

**n_predictions[n]**

Here you can define how many *predictions* should be computed by the network. At each iteration, one set of predicted values for waypoints (x,y,z) and orientations are outputted.

**behaviour [search,flight]**

Here the behaviour of the UAV can be defined either as *search* or *flight*. In the *search* mode, the UAV navigational change in *x* and *y* directions are increased, which results in a wider angular rotation of the head. In contrast, when the behaviour is set to *flight* the predicted values are smoothed, which reduces the angular rotation of the head. During the production of this paper, all test were carried using the *search* mode.

**start position [-100,100]**

Here the starting position of the drone can be defined by assigning a value for the *x*,*y*,*z* coordinates. We recommend these value to be between -100 and 100.

**smoothness [-2.0,2.0]**

When using the flight mode, you can also define the smoothness of the flight in the x,y and z directions. We recommend using values between -2.0 and 2.0.

```
$ python test_mtrl.py n_predictions behaviour x y z smoothness_x smoothness_y smoothness_z
```
Example:
```
$ python test_mtrl.py 150 search -3 -10 -10 0.75 -0.75 0.15
```
### Reference:
If making use of this work in any way (including our [pre-trained models](https://zenodo.org/record/3338078#.XS32h-hKguU) or [dataset](https://zenodo.org/record/3270774#.XS32sehKguU), please reference the following:

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
