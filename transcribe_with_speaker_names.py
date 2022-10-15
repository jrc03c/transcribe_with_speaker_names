from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import contextlib
import datetime
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
):
    # define helper functions
    def time(secs):
        return datetime.timedelta(seconds=round(secs))

    # get path parts
    path_parts = path.split("/")
    dir = ("/").join(path_parts[:-1])
    filename = path_parts[-1]
    filename_parts = filename.split(".")
    filename_without_extension = (".").join(filename_parts[:-1])
    extension = filename_parts[-1]

    # get pretrained speaker model
    embedding_model = PretrainedSpeakerEmbedding(
        pretrained_speaker_embedding, device=torch.device(torch_device)
    )

    # convert file to wav if not one
    created_temp_wav_file = False

    if extension != "wav":
        wav_path = dir + "/" + filename_without_extension + ".wav"
        subprocess.call(["ffmpeg", "-i", path, wav_path, "-y"])
        path = wav_path
        created_temp_wav_file = True

    # transcribe the file
    model = whisper.load_model(whisper_model)
    transcript = model.transcribe(path)
    segments = transcript["segments"]

    with contextlib.closing(wave.open(path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

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
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_

    for i in range(len(segments)):
        segments[i]["speaker"] = "SPEAKER " + str(labels[i] + 1)

    # write the transcript to disk
    transcript_path = dir + "/" + filename_without_extension + ".txt"
    outfile = open(transcript_path, "w")

    for (i, segment) in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            outfile.write(
                "\n" + segment["speaker"] + " " + str(time(segment["start"])) + "\n"
            )

        outfile.write(segment["text"][1:] + " ")

    outfile.close()

    # if a wav file was created, then delete it
    if created_temp_wav_file:
        os.remove(path)

    return transcript_path
