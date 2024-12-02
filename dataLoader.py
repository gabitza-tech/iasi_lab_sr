import glob, numpy, os, random, torch
from scipy import signal
import ffmpeg
import numpy as np

class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        # Load and configure augmentation files
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment using ffmpeg
        audio, sr = read_audio_with_ffmpeg(self.data_list[index])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        # Data Augmentation
        augtype = random.randint(0, 5)
        if augtype == 0:   # Original
            audio = audio
        elif augtype == 1: # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2: # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3: # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4: # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5: # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)

    def read_audio_with_ffmpeg(self, file_path):
        # Use ffmpeg to read the audio file and output raw PCM data
        out, err = ffmpeg.input(file_path).output('pipe:1').run(capture_stdout=True, capture_stderr=True)

        # Extract sample rate (ffmpeg outputs sample rate as part of stderr)
        for line in err.decode().split('\n'):
            if 'Audio' in line and 'Hz' in line:
                # Sample rate is in the form: 44100 Hz, 22050 Hz, etc.
                sr = int(line.split(',')[2].split()[0])
                break
        else:
            sr = None  # If sample rate couldn't be determined

        # Convert the byte output from ffmpeg to numpy array (16-bit signed integers)
        audio = numpy.frombuffer(out, numpy.int16)

        # Normalize the audio to range [-1, 1]
        audio = audio.astype(numpy.float32) / numpy.max(numpy.abs(audio))

        return audio, sr

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = read_audio_with_ffmpeg(rir_file)
        rir = numpy.expand_dims(rir.astype(float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = read_audio_with_ffmpeg(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio


class test_loader(object):
    def __init__(self, test_list, num_frames=300, **kwargs):
        self.num_frames = num_frames
        # Load data & labels
        self.data_list = []
        self.data_label = []
        with open(test_list, 'r') as f:
            lines = f.readlines()

        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        for index, line in enumerate(lines):
            line = line.strip()
            speaker_label = dictkeys[line.split()[0]]
            file_name = line.split()[1]
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment using ffmpeg
        audio, sr = read_audio_with_ffmpeg(self.data_list[index])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)

        return torch.FloatTensor(audio[0]), self.data_label[index], self.data_list[index]

    def __len__(self):
        return len(self.data_list)

def read_audio_with_ffmpeg(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        # Use ffmpeg to decode the audio to raw PCM data
        out, err = ffmpeg.input(file_path).output('pipe:1', format='s16le', acodec='pcm_s16le', ar='16000').run(capture_stdout=True, capture_stderr=True)

        # Extract sample rate from stderr output
        sr = 16000
        if sr is None:
            raise ValueError(f"Unable to determine the sample rate for file: {file_path}")

        # Convert byte output from ffmpeg into a numpy array (16-bit signed integers)
        audio = np.frombuffer(out, dtype=np.int16)

        # Normalize audio to float in the range [-1, 1]
        audio = audio.astype(np.float32) / np.max(np.abs(audio))

        return audio, sr

    except ffmpeg.Error as e:
        # Log the error from ffmpeg
        print(f"FFmpeg error while processing {file_path}:")
        print(e.stderr.decode())  # Print stderr to get more details
        raise  # Reraise the exception to handle it in the calling code

    except Exception as e:
        print(f"Error while reading {file_path}: {e}")
        raise  # Reraise the exception to handle it in the calling code

