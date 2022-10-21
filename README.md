# Intro

This library is just an adaptation of [a Google Colab notebook](https://colab.research.google.com/drive/1V-Bt5Hm2kjaDb4P1RyMSswsDKyrzc2-3?usp=sharing) originally by [Dwarkesh Patel (@dwarkesh_sp)](https://twitter.com/dwarkesh_sp/status/1579672641887408129?s=46&t=8yLFQ2vByL6vA61wFVJCvA). All I did was clean it up a little to make it more robust, package it up, and make it callable as a function! All credit goes to Dwarkesh, though!

To summarize Dwarkesh's notes, the main steps in his algorithm are:

1. Use OpenAI's [Whisper](https://github.com/openai/whisper) to transcribe an audio file into segments.
2. Use [pyannote.audio](https://github.com/pyannote/pyannote-audio) to get the embeddings of each chunk of audio delimited by the timestamps in the segments.
3. Use sklearn's [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) to cluster and label the embeddings.

# Installation

```bash
pip install -U \
    git+https://github.com/jrc03c/transcribe_with_speaker_names \
    git+https://github.com/openai/whisper \
    git+https://github.com/pyannote/pyannote-audio
```

# Usage

```py
from transcribe_with_speaker_names import transcribe_with_speaker_names

segments = transcribe_with_speaker_names(
    "path/to/some-audio-file.mp3",
    pretrained_speaker_embedding="speechbrain/spkrec-ecapa-voxceleb",
    torch_device="cuda",
    num_speakers=2,
    whisper_model="tiny.en",
    verbose=True,
)

# do something with `segments` ...
```

The function returns an array of segments, which are just simple objects with these properties:

- `speaker` = an integer representing which speaker is speaking in the segment
- `time` = a string in the format `"hh:mm:ss"` representing the time at which the segment started
- `text` = a string representing the words spoken by the speaker during the segment

## Parameters

### `[path]`

The path to the audio file. The path can be relative or absolute.

Note that non-WAV files will be automatically converted to WAV using [`ffmpeg`](https://ffmpeg.org/). So, you may need to install `ffmpeg` if you don't have another way of converting non-WAV files to WAV (e.g., by converting them from some other format to WAV in Audacity). If it's necessary for the function to convert your file to WAV first, then the WAV file will be placed right beside the original audio file and given the same name (except with a `.wav` extension). The WAV file will be deleted, though, before the function finishes running. So, you should be left only with the original audio file and its transcript. (I don't _think_ converting from non-WAV to WAV should take very long with `ffmpeg`, even for very large files; but if it turns out to take longer than I'm imagining, then I'll change things to keep the WAV file around in case the function needs to be called again.)

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

### `verbose`

A boolean value representing whether or not the program should print messages to the console as it works. Unfortunately, it's not currently possible to turn _all_ messages off because Whisper uses a progress bar and (perhaps) prints its own messages. But setting `verbose = False` will turn off all of the extra messages I've added.

# Notes & caveats

**Speaker changes inside segments:** When Whisper transcribes an audio file, it returns an array of segments. The original Google Colab notebook written by Dwarkesh Patel (linked at the top) takes each of those segments, gets the embeddings of the audio, clusters the embeddings based on the specified number of speakers, and then labels each segment with a speaker's name (like "Speaker 1"). I've kept his original solution in this library, but do note that it has at least one flaw: if the speaker changes _in the middle of_ a clip (i.e., if a single clip contains two speakers), then that clip will still only receive one speaker label. I haven't yet thought of a way around this, though I admit that I haven't done any research on the problem or even read through the full Twitter thread linked at the top; so maybe someone has already solved it. If so, please let me know!

> The only solution I've considered is trying to break up the audio by speaker changes _first_ and then passing the segments into Whisper for transcription. But I'm not really sure of the best way to break the audio into chunks for identification. In the current version, the timestamps generated by Whisper are used to break the original audio up into chunks, and then those chunks are passed into `pyannote.audio` for embedding. But if I try to get the embeddings _before_ transcription, then I'll have to make decisions about how to break the audio into chunks without Whisper's timestamps. Here are some possible ways of making such decisions:
>
> 1. I could use a sliding window of (e.g.) 3 seconds such that each clip passed into `pyannote.audio` is 3 seconds long. The window could march forward by some timestep, like 0.25 seconds. I wonder, though, if there's an ideal amount of time to use to get embeddings. In other words, does 3 seconds of audio contain enough information? Or is it way too much because (e.g.) 0.25 seconds would contain sufficient information?
> 2. I could break the audio along silences of some particular length. For example, everytime there's a silence that lasts at least 0.5 seconds long, I break the audio half-way through the silence, closing off a clip and starting a new one. This method assumes, though, that one speaker never cuts another off.
> 3. I could use a sliding window to get median frequency for some chunk of time, and then use significant changes in median frequency as breakpoints. This assumes, though, that no two speakers' voices are in exactly the same range, so it's probably not a very robust solution.
>
> I'll tinker with these and see if any produce better results than the current setup.

**Windows compatibility:** This library probably won't work on Windows because of the ways I've been handling file paths!
