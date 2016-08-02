## Database Info
The RMBL-Robin database is a Robin song database collected by using a close-field song meter (www.wildlifeacoustics.com) at the Rocky Mountain Biological Laboratory near Crested Butte, Colorado in the summer of 2009. The recorded Robin songs are naturally corrupted by different kinds of background noises, such as wind, water and other vocal bird species. Non-target songs may overlap with target songs. Each song usually consists of 2-10 syllables. The timing boundaries and noise conditions of the syllables and songs, and human inferred syllable patterns are annotated. Each song usually consists of 2-10 syllables. The timing boundaries and noise conditions of the syllables and songs, and human inferred syllable patterns are annotated.


## Usage with Praat

The database is 78.3 minutes long. In data/ directory, the .WAV files are mono-channel recordings with a sampling rate of 44.1 kHz in 16 bit PCM WAV format; the .TextGrid files are the transcription of the recordings in text format. To see the waveform or spectrogram of the recordings and the transcription, you can open the .WAV and .TextGrid files in 'Praat'. Praat is a free software with audio visualizing and annotating functions. To download Praat, check this: http://www.fon.hum.uva.nl/praat/.

Here is an example for opening a recording (A-01june09-0702-robin.WAV) and its Robin song and
syllable annotation ('A-01june09-0702-robin.TextGrid') in Praat: open Praat program, choose
'Open' from the menu in the 'Praat Object' window, then choose 'Read from file ...' from the list in the menu; in the pop-out file selection box, go to the data/ directory,
press Ctrl to select 'A-01june09-0702-robin.WAV' and 'A-01june09-0702-robin.TextGrid' at the same time, click 'Open' button in the bottom right; now the left panel of 'Praat Object' window has two items in it, i.e., 'TextGrid A-01june09-0702-robin' and 'Sound A-01june09-0702-robin',
press Ctrl to select both of them at the same time, then click the
'View & Edit' button on the right panel, Praat will pop out a new window
showing the waveform and spectrogram of the recording, and the
transcription in different layers. There are two transcription layers:
'song' and 'syllable-quality' layers. In the song layer, each Robin song is
annotated as a segment with a number denoting the background
noise level: 1 -- not noisy; 2 -- noisy, 3 -- very noisy.
The noise level is inferred by a human annotator.
In the syllable-quality layer, each Robin syllable is annotated
as a 'pattern index-noise level' pair. The patterns are also inferred
by the human annotator. The noise level here is defined
the same as the one defined in the song layer.

## Naming Protocol
Take 'A-01june09-0702-robin' for example, 'A' means the Robin individual index (we assume the recordings from the same district are from the same individual); '01june09-0702' means that it is recorded at 7:02 (24hr), June 1th, 2009. The 'A-01june09-0702-robin.TextGrid' is the transcription file of 'A-01june09-0702-robin.WAV'.

## Reference
To reference the RMBL-Robin database, please use the following:
W. Chu, D.T. Blumstein, “Noise robust bird song detection using syllable pattern-based hidden Markov models,” ICASSP 2011.

## Contact
If you have questions or suggestions, feel free to contact:

Wei Chu
University of California, Los Angeles
weichu@ucla.edu
http://www.ee.ucla.edu/~weichu

or

Prof. Dan Blumstein
University of California, Los Angeles
marmots@ucla.edu
http://www.eeb.ucla.edu/Faculty/Blumstein
