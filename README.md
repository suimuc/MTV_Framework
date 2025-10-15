# [NeurIPS2025] MTV: Audio-Sync Video Generation with Multi-Stream Temporal Control
## [Project page](https://hjzheng.net/projects/MTV/) | [Paper](https://arxiv.org/abs/2506.08003)
Official implementation of "Audio-Sync Video Generation with Multi-Stream Temporal Control", which is accepted by NeurIPS 2025.
<table align='center' border="0" style="width: 100%; text-align: center; margin-top: 80px;">
  <tr>
    <td>
      <video align='center' src="https://github.com/user-attachments/assets/e9927831-f0d0-4195-9ead-ecd2fe1ca7c6" autoplay loop></video>
    </td>
  </tr>
    <tr align="center">
    <td>
      <em>For the best experience, please enable audio.</em>
    </td>
  </tr>
</table>

### An open Veo3-style audio-video generation demo
We further developed a Veo3-style audio-video generation demo in [MTVCraft](https://github.com/baaivision/MTVCraft), a framework for generating videos with synchronized audio from a single text prompt , exploring a potential pipeline for creating general audio-visual content.

## ⚙️ Installation

For CUDA 12.1, you can install the dependencies with the following commands. Otherwise, you need to manually install `torch`, `torchvision` , `torchaudio` and `xformers`.

Download the codes:

```bash
git clone https://github.com/suimuc/MTV_Framework
cd MTV_Framework
```

Create conda environment:

```bash
conda create -n mtv python=3.10
conda activate mtv
```

Install packages with `pip`

```bash
pip install -r requirements.txt
```

Besides, ffmpeg is also needed:

```bash
apt-get install ffmpeg
```

## 📥 Download Pretrained Models

You can easily get all pretrained models required by inference from our [HuggingFace repo]().

Using `huggingface-cli` to download the models:

```shell
pip install "huggingface_hub[cli]"
huggingface-cli download BAAI/MTVCraft --local-dir ./pretrained_models
```

Or you can download them separately from their source repo:

- [mtv](https://huggingface.co/BAAI/MTVCraft/tree/main/mtv): Our checkpoints
- [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl): text encoder, you can download from [text_encoder](https://huggingface.co/THUDM/CogVideoX-2b/tree/main/text_encoder) and [tokenizer](https://huggingface.co/THUDM/CogVideoX-2b/tree/main/tokenizer)
- [vae](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main/vae): Cogvideox-5b pretrained 3d vae
- [wav2vec](https://huggingface.co/facebook/wav2vec2-base-960h): wav audio to vector model from [Facebook](https://huggingface.co/facebook/wav2vec2-base-960h)

Finally, these pretrained models should be organized as follows:

```text
./pretrained_models/
|-- mtv
|   |--single/
|		|		|-- 1/
|		|			|-- mp_rank_00_model_states.pt
|   |   `--latest
|		|
|   |--multi/
|		|		|-- 1/
|		|			|-- mp_rank_00_model_states.pt
|		|		`-- latest
|		|
|		`--accm/
|				|-- 1/
|					|-- mp_rank_00_model_states.pt
|				`--latest
|
|-- t5-v1_1-xxl/
|   |-- config.json
|   |-- model-00001-of-00002.safetensors
|   |-- model-00002-of-00002.safetensors
|   |-- model.safetensors.index.json
|   |-- special_tokens_map.json
|   |-- spiece.model
|   `-- tokenizer_config.json
|
|-- vae/
|		|--3d-vae.pt
|
`-- wav2vec2-base-960h/
    |-- config.json
    |-- feature_extractor_config.json
    |-- model.safetensors
    |-- preprocessor_config.json
    |-- special_tokens_map.json
    |-- tokenizer_config.json
    `-- vocab.json
```

## 🎮 Run Inference

#### Batch

Once the model is downloaded and placed in the right directory, you can run inference using the provided script:

```bash
# for one_person
bash scripts/inference_long.sh ./examples/One_person.txt output

#for multi_person
bash scripts/inference_long_multi.sh ./examples/Multi_person.txt output

#for no conversation
bash scripts/inference_long_effect.sh ./examples/Accm.txt output
```
This will read the input prompts from `./examples/samples.txt` and the results will be saved at `./output`.

## Training

### Prepare data for training

Download the [DEMIX]() dataset (coming soon). After get the `csv_file` contain absolute path of video clips and corresponding text descriptions, provide it to the `data_process/demix.py` script to separate audios into distinct controlling tracks (*i.e.*, speech, effects, and music): 

```shell
python data_process/demix.py --csv_file csv_file --high_quality
```

Next, process the controlling tracks with the following commands:
```bash
python mtv/track_preprocess.py --input_csv csv_file
```

### Training

Update the path of csv_file in train_data settings in the configuration YAML files, `configs/training.yaml` :

```yaml
#training.yaml
train_data: [
    "path_of_csv_file"
] # Train data path
```

Start training with the following command:
```bash
bash scripts/finetune_mtv.sh
```

## 📝 Citation

If you find our work useful for your research, please consider citing the paper:

```
@article{MTV,
      title={Audio-Sync Video Generation with Multi-Stream Temporal Control},
      author={Weng, Shuchen and Zheng, Haojie and Chang, Zheng and Li, Si and Shi, Boxin and Wang, Xinlong},
      journal={arXiv preprint arXiv:2506.08003},
      year={2025}
}
```
