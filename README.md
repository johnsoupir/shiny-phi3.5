
# Shiny-Phi3.5

**Shiny-Phi3.5** is a reflection fine-tune of Phi3.5 using mahiatlinux's dataset. 

Reflection fine-tuning guides the model to generate a plan, and then reflect on the plan before proceeding to the final output. A similar approach has been used by instucting the model to plan and reflect via system promps. Reflection tuning "bakes in" the behaviour and may or may not have additional bennefits. The early results are promising.


This repository contains datasets in both their original and processed forms, as well as the code necessary for fine-tuning.
## Model
The model weights are available on my [huggingface page](https://huggingface.co/johnsoupir/Shiny-Phi3.5).


## Contents

- `train-shiny-phi3.5.py`: The main script for fine-tuning the Phi3.5 model.
- `Datasets/`: Contains both the original datasets and the pre-processed datasets ready for training. It also includes the pre-processing script (`format_data.py`).
- `format_data.py`: A Python script to format raw data into a specific structure suitable for fine-tuning.

## Usage

### Fine-Tuning

To fine-tune the model, run the `train-shiny-phi3.5.py` script. This script utilizes a custom training setup optimized for performance.

```sh
python train-shiny-phi3.5.py
```
### Data Preparation

The original dataset is available in the `Datasets/` directory. The pre-processed version, `formatted_reflection_v2.jsonl`, is also included for convenience. If you wish to process a new dataset, use the `format_data.py` script:

```sh
python format_data.py <input_file.jsonl> <output_file.jsonl>
```


## Dependencies

Install the required dependencies with:

```sh
pip install -r requirements.txt
```

## Credits

- **Datasets**: Provided by Maheswar KK (mahiatlinux on Hugging Face)[https://huggingface.co/datasets/mahiatlinux/Reflection-Dataset-v2]


## License

This project is licensed under the MIT License.