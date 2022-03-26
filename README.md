<div id="top"></div>

## Long Term planning with Deep Reinforcement Learning agent ##

___
<p align="center">
    <a href="https://github.com/dorbittonn">Dor Bitton</a> •
    <a href="https://github.com/yuvalgos">Yuval Goshen</a>
  </p>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="assets/p5.gif" alt="Logo" height=250>
  </a>

</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

The following is the graduation project of Dor Bitton & Yuval Goshen, 2 Computer engineering Bsc. students from Technion
Insitute - Haifa. The main Goal of the project is to use solve problem of long term planning, like solving a maze given
an ant robot and a certain goal. The solution we've chosen for the problem is Deep Reinforcement learning based. Using
DDPG algorithm we managed to reach any goal within reasonable radius, controlling the ant's joints.

Our solution for solving a maze is built of 2 agents:

***The Stepper***

- This agent is able to control the ant robot's joints given a certain relatively close goal (subgoal), and arrive it in
  short time.

***The Navigator***

- This agent is able to generate sub goals to the stepper, given a main goal from the user.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

* [Pytorch](https://pytorch.org/)
* [Stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/)
* [Pybullet](https://pybullet.org/)
* [Tensorboard](https://www.tensorflow.org/tensorboard/get_started)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

Just install the requirements file to ensure you've got the needed libraries.

  ```sh
  pip install requirements.txt -r
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't
rely on any external dependencies or services._

1.Clone the repo

   ```sh
   git clone https://github.com/rl-project-dor-yuval/robomaze.git
   ```

2.Install required packages

   ```sh
    pip install requirements.txt -r
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->

## Usage

to be written.


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->

## Roadmap

- [x] Train Stepper Agent
- [ ] Train Navigator Agent
- [ ] Add observation with computer vision
- [ ] Try to generalize maze solving to generic maze.
- [ ] Multi-language Support

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->

## Contact

Dor Bitton - [Linkedin](https://www.linkedin.com/in/dor-bitton-54a1b919a/) - dorbittonn@gmail.com

Yuval Goshen - [Linkedin](https://www.linkedin.com/in/yuval-goshen-a8390b1ba/) - yuvalgos@gmail.com

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

Out work is mainly based on the following papers

* [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)
* [Playing Atari With Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)

Second part will be extending the following paper:
* [Harnessing Reinforcement Learning for Neural Motion Planning](https://arxiv.org/abs/1906.00214)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge

[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge

[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members

[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge

[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers

[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge

[issues-url]: https://github.com/othneildrew/Best-README-Template/issues

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge

[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/othneildrew

[product-screenshot]: images/screenshot.png
