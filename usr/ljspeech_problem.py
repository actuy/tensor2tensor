# coding=utf-8
"""Ljspeech dataset."""

import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import speech_recognition
from tensor2tensor.utils import registry

import tensorflow as tf

_LJSPEECH_TTS_DATASET = "http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

def _collect_data(directory, input_ext, transcription_ext):
    """Traverses directory collecting input and target files."""
    # Directory from string to tuple pair of strings
    # key: the filepath to a datafile including the datafile's basename. Example,
    #   if the datafile was "/path/to/datafile.wav" then the key would be
    #   "/path/to/datafile"
    # value: a pair of strings (media_filepath, label)
    data_files = {}
    for root, _, filenames in os.walk(directory):
        transcripts = [filename for filename in filenames
                       if transcription_ext in filename]
        for transcript in transcripts:
            transcript_path = os.path.join(root, transcript)
            with open(transcript_path, "r") as transcript_file:
                for transcript_line in transcript_file:
                    line_contents = transcript_line.strip().split(" ", 1)
                    media_base, label = line_contents
                    key = os.path.join(root, media_base)
                    assert key not in data_files
                    media_name = "%s.%s" % (media_base, input_ext)
                    media_path = os.path.join(root, media_name)
                    data_files[key] = (media_base, media_path, label)
    return data_files


def set_ljspeech_hparams(model_hparams):
    model_hparams.audio_sample_rate = 22050
    model_hparams.num_freq = 1025
    model_hparams.rescale = True  # Whether to rescale audio prior to preprocessing
    model_hparams.rescaling_max = 0.999  # Rescaling value
    model_hparams.hop_size = 275  # For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
    model_hparams.win_size = 1100  # For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    model_hparams.preemphasize = True  # whether to apply filter
    model_hparams.preemphasis = 0.97  # filter coefficient.
    model_hparams.min_level_db = -100
    model_hparams.ref_level_db = 20
    model_hparams.fmin = 55  # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
    model_hparams.fmax = 7600  # To be increased/reduced depending on data.
    model_hparams.n_fft = 2048  # Extra window size is filled with 0 paddings to match this parameter
    model_hparams.signal_normalization = True  # Extra window size is filled with 0 paddings to match this parameter
    model_hparams.allow_clipping_in_normalization = True  # Only relevant if mel_normalization = True
    model_hparams.symmetric_mels = True  # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
    model_hparams.max_abs_value = 4.  # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion,
    # Griffin Lim
    model_hparams.power = 1.5  # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
    model_hparams.griffin_lim_iters = 60  # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.

    # #M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
    model_hparams.trim_fft_size = 512
    model_hparams.trim_hop_size = 128
    model_hparams.trim_top_db = 23
    model_hparams.frame_shift_ms = None  # Can replace hop_size parameter. (Recommended: 12.5)
    model_hparams.use_lws = False
    model_hparams.silence_threshold = 2  # silence threshold used for sound trimming for wavenet preprocessing
    model_hparams.trim_silence = True  # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)


@registry.register_problem()
class LjspeechProblem(speech_recognition.SpeechRecognitionProblem):
    @property
    def num_shards(self):
        return 100

    @property
    def use_subword_tokenizer(self):
        return False

    @property
    def num_dev_shards(self):
        return 1

    @property
    def num_test_shards(self):
        return 1

    @property
    def use_train_shards_for_dev(self):
        """If true, we only generate training data and hold out shards for dev."""
        return False

    def hparams(self, defaults=None, model_hparams=None):
        super().hparams(defaults, model_hparams)
        set_ljspeech_hparams(self.model_hparams)
        self.model_hparams.add_hparam('symbol_size', 64)

    def generator(self, data_dir, tmp_dir, datasets,
                  eos_list=None, start_from=0, how_many=0):
        del eos_list
        i = 0
        for url, subdir in datasets:
            filename = os.path.basename(url)
            compressed_file = generator_utils.maybe_download(tmp_dir, filename, url)

            read_type = "r:gz" if filename.endswith("tgz") else "r"
            with tarfile.open(compressed_file, read_type) as corpus_tar:
                # Create a subset of files that don't already exist.
                #   tarfile.extractall errors when encountering an existing file
                #   and tarfile.extract is extremely slow
                members = []
                for f in corpus_tar:
                    if not os.path.isfile(os.path.join(tmp_dir, f.name)):
                        members.append(f)
                corpus_tar.extractall(tmp_dir, members=members)

            raw_data_dir = os.path.join(tmp_dir, "LjSpeech", subdir)
            data_files = _collect_data(raw_data_dir, "flac", "txt")
            data_pairs = data_files.values()

            encoders = self.feature_encoders(data_dir)
            audio_encoder = encoders["waveforms"]
            text_encoder = encoders["targets"]

            for utt_id, media_file, text_data in sorted(data_pairs)[start_from:]:
                if 0 < how_many == i:
                    return
                i += 1
                wav_data = audio_encoder.encode(media_file)
                spk_id, unused_book_id, _ = utt_id.split("-")
                yield {
                    "waveforms": wav_data,
                    "waveform_lens": [len(wav_data)],
                    "targets": text_encoder.encode(text_data),
                    "raw_transcript": [text_data],
                    "utt_id": [utt_id],
                    "spk_id": [spk_id],
                }

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        train_paths = self.training_filepaths(
            data_dir, self.num_shards, shuffled=False)
        # dev_paths = self.dev_filepaths(
        #     data_dir, self.num_dev_shards, shuffled=False)
        test_paths = self.test_filepaths(
            data_dir, self.num_test_shards, shuffled=True)
        data = self.generator(data_dir, tmp_dir, _LJSPEECH_TTS_DATASET, start_from=0, how_many=100)
        generator_utils.generate_files(data, test_paths)
        data = self.generator(data_dir, tmp_dir, _LJSPEECH_TTS_DATASET, start_from=100, how_many=-1)
        generator_utils.generate_files(data, train_paths)
        generator_utils.shuffle_dataset(train_paths)

        # generator_utils.generate_files(
        #     self.generator(data_dir, tmp_dir, _LJSPEECH_TTS_DATASET), test_paths)
        #
        # if self.use_train_shards_for_dev:
        #     all_paths = train_paths + dev_paths
        #     generator_utils.generate_files(
        #         self.generator(data_dir, tmp_dir, _LJSPEECH_TTS_DATASET), all_paths)
        #     generator_utils.shuffle_dataset(all_paths)
        # else:
        #     generator_utils.generate_dataset_and_shuffle(
        #         self.generator(data_dir, tmp_dir, _LJSPEECH_TTS_DATASET), train_paths,
        #         self.generator(data_dir, tmp_dir, _LJSPEECH_TTS_DATASET), dev_paths)

    def dataset_filename(self):
        return 'ljspeech_speech_problem'


