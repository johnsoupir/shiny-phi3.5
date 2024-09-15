
# Shiny-Phi3.5

**Shiny-Phi3.5** is a reflection fine-tune of Phi3.5 using mahiatlinux's dataset. 

Recently "Reflection 70B" drew a lot of attention after making claims of massive performance gains via reflection tuning. However, independent testing has been unable to reproduce these results.

Reflection fine-tuning guides the model to generate a plan, and then reflect on the plan before proceeding to the final output. A similar approach has been used by Claude: instructing the model to plan and reflect via system prompts. Reflection tuning "bakes in" the behavior. 

I was curious to try it myself, so I made this model. If you'd like to try a smaller reflection model for yourself, or just one that's not associated with the original - then here you go!

This repository contains the code necessary for fine-tuning.
## Model
The model weights are available on my [huggingface page](https://huggingface.co/johnsoupir/Shiny-Phi3.5).


## Contents

- `train-shiny-phi3.5.py`: The main script for fine-tuning the Phi3.5 model.

## Usage

### Fine-Tuning

To fine-tune the model, run the `train-shiny-phi3.5.py` script.

```sh
python train-shiny-phi3.5.py
```

## Dependencies

Install the required dependencies with:

```sh
pip install -r requirements.txt
```

## Credits

- **Datasets**: Provided by Maheswar KK [mahiatlinux on Hugging Face](https://huggingface.co/datasets/mahiatlinux/Reflection-Dataset-v2)


## License

This project is licensed under the MIT License.
