from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import contextlib
import numpy as np
import os
import subprocess
import torch
import wave
import whisper


def transcribe_with_speaker_names(
    path,
    pretrained_speaker_embedding="speechbrain/spkrec-ecapa-voxceleb",
    torch_device="cuda",
    num_speakers=2,
    whisper_model="tiny.en",
    verbose=True,
):
    # define helper functions
    def left_pad(s, n):
        out = str(s)

        while len(out) < n:
            out = "0" + out

        return out

    def get_time_string(s):
        hours = int(s / (60 * 60))
        minutes = int((s - hours * 60 * 60) / 60)
        seconds = int(s - minutes * 60 - hours * 60 * 60)

        return (
            left_pad(hours, 2) + ":" + left_pad(minutes, 2) + ":" + left_pad(seconds, 2)
        )

    def pretty_print(x):
        if verbose:
            print("==========")
            print(x)
            print("==========")

    # get path parts
    path = os.path.abspath(path)
    dir = os.path.dirname(path)
    filename = os.path.basename(path)
    filename_parts = filename.split(".")
    filename_without_extension = (".").join(filename_parts[:-1])
    extension = filename_parts[-1].lower()

    # get pretrained speaker model
    embedding_model = PretrainedSpeakerEmbedding(
        pretrained_speaker_embedding, device=torch.device(torch_device)
    )

    # convert file to wav if not one
    created_temp_wav_file = False

    if extension != "wav":
        wav_path = dir + "/" + filename_without_extension + ".wav"

        pretty_print("Converting %s to WAV format..." % (filename))

        subprocess.call(
            ["ffmpeg", "-i", path, wav_path, "-y"],
            stdout=None if verbose else subprocess.DEVNULL,
            stderr=None if verbose else subprocess.DEVNULL,
        )

        path = wav_path
        created_temp_wav_file = True

    # transcribe the file
    pretty_print("Transcribing...")

    model = whisper.load_model(whisper_model)
    transcript = model.transcribe(path, verbose=verbose)
    segments = transcript["segments"]

    with contextlib.closing(wave.open(path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    # get clip embeddings
    pretty_print("Getting audio clip embeddings...")

    audio = Audio()
    embeddings = np.zeros(shape=(len(segments), embedding_model.dimension))

    for i, segment in enumerate(segments):
        start = segment["start"]

        # whisper overshoots the end timestamp in the last segment
        end = min(duration, segment["end"])

        clip = Segment(start, end)
        waveform, _ = audio.crop(path, clip)
        embeddings[i] = embedding_model(waveform[None])

    embeddings = np.nan_to_num(embeddings)

    # cluster embeddings
    pretty_print("Clustering embeddings...")

    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_

    # assign each segment's speaker
    for i in range(len(segments)):
        segments[i]["speaker"] = labels[i] + 1

    # join consecutive speaker segments
    out = []
    temp_text = ""
    temp_speaker = -1
    temp_time = ""

    for (i, segment) in enumerate(segments):
        if len(temp_text) == 0:
            temp_text = segment["text"]
            temp_speaker = segment["speaker"]
            temp_time = get_time_string(segment["start"])

        elif segment["speaker"] != temp_speaker:
            out.append(
                {"speaker": temp_speaker, "time": temp_time, "text": temp_text,}
            )

            temp_text = segment["text"]
            temp_speaker = segment["speaker"]
            temp_time = get_time_string(segment["start"])

        else:
            temp_text += segment["text"]

    if created_temp_wav_file:
        pretty_print("Deleting WAV file (%s) ..." % (path))
        os.remove(path)

    return out
