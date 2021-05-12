# Todo List For Version 0 of the simulator
* move maze to be around (0,0) and update start_state validation
* return obvesrvation as concatenated vector of:
    * ant position (COM) - R^2 (-maze_size/2, maze_size/2)
    * ant velocity (COM) - R^2  
    * joint position - R^8 (-1, 1)
    * joint velocity - R^8 (-1, 1)
    * facing direction - R (-pi, pi)
* add the option to choose gui or direct mode
* if there is time - handle maze creating
    
* Read Stable Baselines library (each of us)
* Run on lab computer using SSH interpreter (each of us)
