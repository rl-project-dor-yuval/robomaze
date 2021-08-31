# Todo List For The Maze:
* Add a check in *start_state_validation* that the ant and the target are not on maze tiles
* color ant
* Create wrapper to change observation to [-1,1] 
  * update : that can be done later, for ddpg it is actually less important since it does not use gaussian distribution in the algorithm

# SB3 Model Training Notes (i wrote when reading the documentation):
* Use check_env method first

* Use EvalCallback to evaluate the model during training
  
  maybe it can be done automatically with passing arguments to model.learn() 
* Evaluation should be done for 5-20 episodes, always, as a general rule for RL
  
* Look at the example for using pybullet with google colab, it might be usefull but the remote
interpreter should be connected somehow (or just run the notebook on the server?)
  
* Use verbose=1 or verbose=2 to debug

* After trying some basic training use HER to handle the task as a goal task
    * Read the paper or some tutorial
    * change the enviorment to gym goalEnvironment
    * important note: when using HER, batch size is a very important hyperparameter
    