<p align="center">
  A Failure Detector Model Using Reinforcement Learning:<br> A Cloud Computing Experiment <br><br>
  This project is a partial fulfillment of the requirements for the degree of MSc in Computer Science 
  (Artificial Intelligence - Online) <br>
  May 2020
</p>

<div align="center">
  <a href="https://opensource.org/licenses/mit-license.php">
    <img alt="MIT License" src="https://badges.frapsoft.com/os/mit/mit.svg?v=103" />
  </a>
  <a href="https://github.com/ellerbrock/open-source-badge/">
    <img alt="Open Source Love" src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103" />
  </a>
</div>

<br />

## Parameters
```
--log default: 'logs', description: Root log dir for tensorboard purposes
--model  default: 'model' , description: Root model dir for tensorboard purposes
--input default: '../data/input.csv', description: CSV file with input data
--steps default: 20000, description: Total of steps to train

--train  description: Training mode
--eval   description: Evaluation mode
--lin  description: Learning Rate linear decay
--newmodel  description: Create a new model rather than evolve the current one
```

## Obs
```
Logs and model new folders may be created following the rule of incrementing the number at the end:
<folder_name>
<folder_name>_1
<folder_name>_2 
..
```

## Usage

```bash
$ git clone https://github.com/maikeldf/msc_ai.git
$ cd msc_ai/project/fdm
# Set Up a Virtual Environment (recommended)
$ python -m pip install --upgrade pip
$ python -m pip install -e .

# Example 1 - Training and evaluation
# - Input: input csv file
# - Mode: training and evaluation
# - Creating a new model from scratch
# - Learning Rate: linear decay
# - Steps: 20000
$ python fdm.py --input ../../data/input.csv --train --eval --newmodel --lin --steps 20000

# Example 2 - Only evaluating
# - Input: input csv file
# - Mode: evaluation
# - Steps: 20000
$ python fdm.py --input ../../data/input.csv --eval --steps 20000

# Run tensorboard graphics tool
$ tensorboard --port 6005 --logdir ./<logs folder>/progress_tensorboard/
```

## License

Provided under the terms of the [MIT License](https://github.com/maikeldf/msc_ai/blob/main/project/fdm/LICENSE).
