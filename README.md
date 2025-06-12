# wts_70B
## Installation
Get start with the environment setup:

```bash
conda env create -f wts.yaml
```
## SFT Model
Use the folowing command to sft model:

```bash
./lora.sh
```
## Evaluation
launch the controller of FastChat:

```bash
python -m fastchat.serve.controller
```

Use the following commands to conduct the evaluation, take WebShop Task as an example:

```bash
python -m fastchat.serve.model_worker --model-path <your model path> --port 21021 --worker-address http://localhost:21021 --gpus 0,1,2,3 --num-gpus 4 --max-gpu-memory 40GB --controller-address http://localhost:21020
```
```bash
python -m eval_agent.main --agent_config fastchat --model_name <your model name> --exp_config webshop --split test --verbose
```
