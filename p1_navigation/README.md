[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Create (and activate) a new environment with Python 3.6.

- __Linux__ or __Mac__:
```bash
conda create --name drlnd python=3.6
source activate drlnd
```
- __Windows__:
```bash
conda create --name drlnd python=3.6
activate drlnd
```

2. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/aime01efh/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

2. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

4. Place the file in the deep-reinforcement-learning GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 


6. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-do
wn `Kernel` menu.


### Instructions

Open the `Navigation.ipynb` notebook and execute all cells after **"4. It's Your Turn!"**. This will train the agent, save the resulting model parameters, and run a sample episode using the trained agent.

If all you want to do is to watch a sample episode, then you can execute just the first two code cells after **"4. It's Your Turn!"** with the imports, the next cell creating the UnityEnvironment, and then the final two cells that instantiate the agent and run the sample episode.
