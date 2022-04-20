import datetime
import logging
import os
from argparse import ArgumentParser
import soundfile as sf
import ast
from os.path import exists
import os
from augmentation import augment, augment_time_stretch, augment_gaussian, augment_pitch_shift

# This is a max of approximately one minute of audio per batch. Longest sample is slightly shorter.
# Fairseq wav2vec batches by audio duration (samples/sec) rather than number of utterances.
# So a batch could be that one utterance, or it could be 12 5-second utterances.
MAX_TOKENS = 1120000


def get_duration_frames(audio_path):
    samples, sample_rate = sf.read(audio_path)
    length = len(samples)
    return length

def listToString(s): 
    
    # # initialize an empty string
    str1 = "" 
    
    # # traverse in the string  
    # for ele in s: 
    #     str1 += ele + " "
    
    # return string  
    return " ".join(s)
        


def main():

    manifest_root_dir = "/home/bmoell/lithuanian-asr/data/cv-corpus-8.0-2022-01-19/lt/clips/"

    import pandas as pd

    data = pd.read_csv("/home/bmoell/lithuanian-asr/data/cv-corpus-8.0-2022-01-19/lt/train.tsv", sep="\t")

    manifest_filename = "/home/bmoell/lithuanian-asr/out/data/common-voice/train.ltr"
    transcript_filename = "/home/bmoell/lithuanian-asr/out/data/common-voice/train.tsv"

    manifest_validation_filename = "/home/bmoell/lithuanian-asr/out/data/common-voice/valid.ltr"
    transcript_validation_filename = "/home/bmoell/lithuanian-asr/out/data/common-voice/valid.tsv"


    with open(manifest_filename, "w") as mf, open(transcript_filename, "w") as tf:
        with open(manifest_validation_filename, "w") as mvf, open(transcript_validation_filename, "w") as tvf:
            # d = os.path.join(manifest_root_dir)
            # mf.write(f"{d}\n")
            for index, row in data.iterrows():

                filename =  manifest_root_dir + row['path'] + ".wav"

                #filename =  manifest_root_dir + row['path']
                # import pdb
                # pdb.set_trace()
                file_exists = exists(filename)
                if file_exists and os.stat(filename).st_size > 1024:

                    duration_frames = get_duration_frames(filename)
                    transcript = row['sentence']
                    # import pdb
                    # pdb.set_trace()
                    #transcript = ast.literal_eval(transcript)
                    #transcript = listToString(transcript)
                    #
                    # transcript = transcript.replace(",","")
                    # transcript = transcript.replace(".","")
                    # transcript = transcript.replace("!","")
                    # transcript = transcript.replace("?","")
                    # transcript = transcript.replace("'","")
                    # transcript = transcript.replace("  "," ")
                    # transcript = transcript.strip()
                    transcript = transcript.lower()

                    # print(duration_frames)
                    # print(transcript)

                    if duration_frames > MAX_TOKENS:
                        logging.warning(f"SKIPPING: {transcript} because duration {duration_frames} > {MAX_TOKENS}")
                        continue

                    # get the duration of the file

                    ## 
                    if index % 100 == True:                    
                            # print("we are validating")
                            tvf.write(f"{filename}\t{duration_frames}\n")
                            mvf.write(f"{transcript}\n")
                        ## make validation
                    
                    tf.write(f"{filename}\t{duration_frames}\n")
                    mf.write(f"{transcript}\n")

                    augment_path, duration_frames = augment_gaussian(filename)

                    tf.write(f"{augment_path}\t{duration_frames}\n")
                    mf.write(f"{transcript}\n")

                    augment_path, duration_frames = augment_pitch_shift(filename)

                    tf.write(f"{augment_path}\t{duration_frames}\n")
                    mf.write(f"{transcript}\n")

                else:
                    print("this one doesn't exist", filename)


if __name__ == '__main__':
    main()
