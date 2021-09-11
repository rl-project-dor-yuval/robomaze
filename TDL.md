# Todo List For The Maze:
* color ant
* Add a check in *start_state_validation* that the ant and the target are not on maze tiles
* fix observation bounds (now its not used due to bug)
* Create wrapper to change observation to [-1,1] 
  * update : that can be done later, for ddpg it is actually less important since it does not use gaussian distribution in the algorithm

# Todo List For Training and infrastructure:
* Solve cuda error
* Solve multiply environments bug
* Handle video size and embed video in notebook
* Get a trained model for dummy maze (with good video aside)
* Read about HER
* Make some order in the repository files, delete unnecessary, move some usefull code from notebook to
  infrastructure files

# SB3 Model Training Notes (i wrote when reading the documentation):
* Use check_env method first

* Evaluation should be done for 5-20 episodes, always, as a general rule for RL
  
* Use verbose=1 or verbose=2 to debug

* After trying some basic training use HER to handle the task as a goal task
    * Read the paper or some tutorial
    * change the enviorment to gym goalEnvironment
    * important note: when using HER, batch size is a very important hyperparameter
    