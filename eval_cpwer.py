import sys
from itertools import permutations
import os
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio import Pipeline, Audio
from pyannote.audio.pipelines.speech_separation import SpeechSeparation as SeRiouSLy
from pyannote.database import (
    registry,
    FileFinder,
)

import torch
import numpy as np
from jiwer import wer, process_words
import whisperx
from whisper.normalizers import EnglishTextNormalizer
import argparse

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = EnglishTextNormalizer()
    modelx = whisperx.load_model(args.whisper_model, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
    registry.load_database(
        args.database_dir
    )
    protocol = registry.get_protocol(
        "AMI-SDM.SpeakerDiarization.only_words", {"audio": FileFinder()}
    )
    references_dir = args.references_dir
    pipeline = pipeline = Pipeline.from_pretrained(
        "pyannote/speech-separation-ami-1.0",
        use_auth_token=args.access_token
    )
    

    def apply_whisperx(audio, sample_rate=16000):
        audio = np.float32(audio / np.max(np.abs(audio)))
        result = modelx.transcribe(audio, batch_size=16, language="en")
        output = result["segments"]  # after alignment
        text = ""
        for utterance in output:
            text = text + " " + utterance["text"]
        return normalizer(text)

    files = list(protocol.test())
    all_references = []
    all_predictions = []
    
    for i in range(len(files)):
        test = files[i]
        diarization, sources = pipeline(test)
        file_start = test["uri"]

        # fetch references files
        references_file_paths = [
            os.path.join(references_dir, file)
            for file in os.listdir(references_dir)
            if file.startswith(file_start)
        ]

        references = []
        for file_path in references_file_paths:
            with open(file_path, "r") as f:
                references.append(f.read())

        references_formatted = []
        for j in range(len(references)):
            if references[j] != "":
                references_formatted.append(normalizer(references[j]))
        references = references_formatted
        all_references = all_references + references

        predictions = []
        for j in range(sources.data.shape[1]):
            text = apply_whisperx(sources.data[:, j])
            predictions.append(text)

        # only consider 10 longest predictions to save computation time
        predictions = sorted(predictions, key=len, reverse=True)[:10]
        min_wer = 100
        best_perm = None

        if len(predictions) < len(references):
            predictions = predictions + [""] * (len(references) - len(predictions))

        for j in range(len(predictions)):
            predictions[j] = normalizer(predictions[j])

        all_permutations = list(permutations(predictions, len(references)))
        for j, perm in enumerate(all_permutations):
            error = wer(references, list(perm))
            if error < min_wer:
                min_wer = error
                best_perm = perm
        all_predictions = all_predictions + list(best_perm)

    total_wer = wer(all_references, all_predictions) * 100
    processed = process_words(all_references, all_predictions)
    H = processed.hits
    S = processed.substitutions
    D = processed.deletions
    I = processed.insertions
    deletion_rate = D / (H + S + D) * 100
    insertion_rate = I / (H + S + D) * 100
    substitution_rate = S / (H + S + D) * 100

    print("cpWER breakdown\n")
    print(f"Substitution rate: {substitution_rate:.1f}\n")
    print(f"Deletion rate: {deletion_rate:.1f}\n")
    print(f"Insertion rate: {insertion_rate:.1f}\n")
    print(f"Total WER: {total_wer:.1f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--references_dir", type=str, required=True)
    parser.add_argument("--database_dir", type=str, required=True)
    parser.add_argument("--access_token", type=str, required=True)
    parser.add_argument("--whisper_model", type=str, default="small.en")
    args = parser.parse_args()
    main(args)