# Intro

This is just an adaptation of [a Google Colab notebook](https://colab.research.google.com/drive/1V-Bt5Hm2kjaDb4P1RyMSswsDKyrzc2-3?usp=sharing) originally by [Dwarkesh Patel (@dwarkesh_sp)](https://twitter.com/dwarkesh_sp/status/1579672641887408129?s=46&t=8yLFQ2vByL6vA61wFVJCvA). All I did was clean it up a little to make it more robust, package it up, and make it callable as a function! All credit goes to Dwarkesh, though!

Here were Dwarkesh's comments in his Google Colab notebook:

> High level overview of what's happening here:
>
> 1.  I'm using Open AI's Whisper model to seperate audio into segments and generate transcripts.
> 2.  I'm then generating speaker embeddings for each segments.
> 3.  Then I'm using agglomerative clustering on the embeddings to identify the speaker for each segment.
>
> Let me know if I can make it better!

# Installation

```bash
pip install git+https://github.com/jrc03c/transcribe_with_speaker_names
```

# Usage

```py
from transcribe_with_speaker_names import transcribe_with_speaker_names

transcript_path = transcribe_with_speaker_names(
    "path/to/some-audio-file.mp3",
    pretrained_speaker_embedding="speechbrain/spkrec-ecapa-voxceleb",
    torch_device="cuda",
    num_speakers=2,
    whisper_model="tiny.en",
)

print(transcript_path)
# "path/to/some-audio-file.txt"
```

## Parameters

### `[path]`

The path to the audio file. The path can be relative or absolute. The function returns the path to the newly-created transcript, which will be placed right beside the audio file and given the same name (except with a `.txt` extension).

Note that non-WAV files will be automatically converted to WAV using [`ffmpeg`](https://ffmpeg.org/). So, you may need to install `ffmpeg` if you don't have another way of converting non-WAV files to WAV (e.g., by converting them from some other format to WAV in Audacity). If it's necessary for the function to convert your file to WAV first, then the WAV file will be placed right beside the original audio file and given the same name (except with a `.wav` extension). The WAV file will be deleted, though, before the function finishes running. So, you should be left only with the original audio file and its transcript. (I don't _think_ converting from non-WAV to WAV should take very long with `ffmpeg`, even for very large files; but if it turns out to take longer than I'm imagining, then I'll keep the WAV file around in case the function needs to be called again.)

### `pretrained_speaker_embedding`

Possible values for the `pretrained_speaker_embedding` parameter are:

- `"pyannote/embedding"`
- `"speechbrain/spkrec-ecapa-voxceleb"`
- `"nvidia/speakerverification_en_titanet_large"`

See the source [here](https://github.com/pyannote/pyannote-audio/blob/9a5b2afb3b74276f0d1cc17f37f729e7b311808c/pyannote/audio/pipelines/speaker_verification.py#L415) for further clarification.

### `torch_device`

A couple of the possible values for the `torch_device` parameter are:

- `"cuda"`
- `"cpu"`

See the Torch docs [here](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) for more info about possible values.

### `num_speakers`

An integer representing the number of speakers to listen for.

### `whisper_model`

A few of the possible values for the `whisper_model` parameter are:

- `"tiny.en"`
- `"base.en"`
- `"small.en"`
- `"medium.en"`

See the Whisper docs [here](https://github.com/openai/whisper#available-models-and-languages) for more model names.

# Notes

This library probably won't work on Windows because of the ways I've been handling file paths!
