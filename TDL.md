# Todo List For The Maze:
* color ant
* implement close() to disconnect from the pb client (for the case when running notebook)
* Add a check in *start_state_validation* that the ant and the target are not on maze tiles
* fix observation bounds (now its not used due to bug)
* Create wrapper to change observation to [-1,1] 
  * update : that can be done later, for DDPG it is actually less important since it does not use gaussian distribution in the algorithm

# Todo List For Training :
* add option to evaluate outside the training process.(Now recording and video creation is done 
solely at Callback class)
* Get a trained model for dummy maze (with good video aside) v

  
# Todo List - Infrastructures :
* Make some order in the repository files, delete unnecessary, move some usefull code from notebook to
  infrastructure files
* remove videos directory creation, videos are in the log only.

# SB3 Model Training Notes (i wrote when reading the documentation):
* Use check_env method first - v
* Evaluation should be done for 5-20 episodes, always, as a general rule for RL
* Use verbose=1 or verbose=2 to debug

* After trying some basic training use HER to handle the task as a goal task
    * Read the paper or some tutorial
    * change the enviorment to gym goalEnvironment
    * important note: when using HER, batch size is a very important hyperparameter
