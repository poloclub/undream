# UnDREAM: Bridging Differentiable Rendering and Photorealistic Simulation for End-to-end Adversarial Attacks

<p align="center">
    <img src="images/Slice 2-2.png" alt="drawing" width="40%"/>
</p>

Deep learning models deployed in safety critical applications like autonomous driving use simulations to test their robustness against adversarial attacks in realistic conditions. However, these simulations are non-differentiable, forcing researchers to create attacks that do not integrate simulation environmental factors, reducing attack success. To address this limitation, we introduce UnDREAM, the first software framework that bridges the gap between photorealistic simulators and differentiable renderers to enable end-to-end optimization of adversarial perturbations on any 3D objects. UnDREAM enables manipulation of the environment by offering complete control over weather, lighting, backgrounds, camera angles, trajectories, and realistic human and object movements, thereby allowing the creation of diverse scenes. We showcase a wide array of distinct physically plausible adversarial objects that UnDREAM enables researchers to swiftly explore in different configurable environments. This combination of photorealistic simulation and differentiable optimization opens new avenues for advancing research of physical adversarial attacks.


## Getting Started

### Environment Setup
Install Unreal Engine 5 [here](https://dev.epicgames.com/documentation/en-us/unreal-engine/install-unreal-engine) and Mitsuba 3 [here](https://www.mitsuba-renderer.org)

#### Enabling Python in Unreal Engine 5
To use Python in Unreal Engine 5, you will need to enable the Python plugin. Go to Edit > Plugins to find a list of plugins, and install Python plugin into Unreal Engine.

You will need to find the Python path refered to by Unreal Engine to create the environment and install all libraries. To find the Python path you can open up the Output Log window in Unreal Engine (Window > Output Log), select Python from the drop down menu near the command line and run the following commands in the command line.

```
import sys
print(sys.path)
```

#### Creating virtual environment
To create the virtual environment, run 

```pip install -r requirements.txt```

### Initializing XML files

Insert the StaticSequence from sequences\sequence_bin into the Editor, and run 

```python3 initialize.py```

### Adversarial Attack

To start the adversarial attack add sequences to the MovieRenderQueue (Windows > MovieRenderQueue), or run 

```python3 add_sequence_to_queue.py```

then run

```python3 attack_pipeline.py```

