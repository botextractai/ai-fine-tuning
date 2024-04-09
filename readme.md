# Model fine-tuning with Hugging Face and Weights & Biases (wandb.ai)

Fine-tuning a model requires extensive computational resources. Therefore, it's a good idea to do fine-tuning in an environment with access to powerful Graphics Processing Units (GPU's) and large memory resources.

This project uses professional training parameter settings. It might therefore take multiple days for a full run on a regular desktop computer. If you just want to give a try, you might want to lower the training parameters.

If you don't have enough computing power, then both "Google Colab" and "Amazon Web Services SageMaker Studio Lab" might offer limited free access to Jupyter Notebooks with powerful environments.

This project requires (NVIDIA) GPU's for quantization.

If you run this project in your own environment using NVIDIA GPU's, then you need to ensure that you have CUDA installed. You can check this with the `nvidia-smi` command, which will show you detailed information about your GPU's, if CUDA is installed. If this command does not work, then please follow the [NVIDIA CUDA Installation Guides](https://docs.nvidia.com/cuda/) for your operating system.

This project works with the [OpenLLaMA 3B V2](https://huggingface.co/openlm-research/open_llama_3b_v2) open source Large Language Model (LLM).

The model gets fine-tuned for question-answering performance with the [SQuAD V2](https://huggingface.co/spaces/evaluate-metric/squad_v2) dataset. The SQuAD V2 dataset has a part that's supposed to be used in training and another part in validation.

The fine-tuning uses the Low-Rank Adaptation (LoRA) training technique. LoRA is a type of Parameter Efficient Fine-Tuning (PEFT), which updates only a small portion of the weights relative to the full model size, keeping the bulk of the model unchanged. This helps preventing catastrophic forgetting. PEFT requires much less computing power and training data than full fine-tuning of a whole model.

You need a Hugging Face API key with write permissions for this project. [Get your free Hugging Face API key with write permission here](https://huggingface.co/settings/tokens). Insert your Hugging Face API key in the "ai-fine-tuning.py" file. The write permissions are required, if you want to upload the new (trained) model adapter to Hugging Face.

Weights & Biases (W&B) is a MLOps platform that can help developers monitor and document Machine Learning training workflows from end to end. W&B is used to get an idea of how well the training is working and if the model is improving over time. You need a W&B API key for this project. [Get your free Weights & Biases API key here](https://wandb.ai/authorize). Insert your W&B API key in the "ai-fine-tuning.py" file.

You can check your W&B projects at https://wandb.ai/YOUR_WANDB_USER_NAME/projects .

After the training has finished, the new (trained) model can be used with the following Python program code. Usually, the new (trained) PEFT model is stored as an adapter, not as a full model, therefore the loading in LangChain is a bit different:

```
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

model_id = 'openlm-research/open_llama_3b_v2'
config = PeftConfig.from_pretrained("YOUR_HUGGING_FACE_USER_NAME/openllama-3b-peft-squad_v2")
model = AutoModelForCausalLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(model, "YOUR_HUGGING_FACE_USER_NAME/openllama-3b-peft-squad_v2")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256
)
llm = HuggingFacePipeline(pipeline=pipe)
```
