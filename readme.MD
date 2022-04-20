## Lithuanian ASR
The goal of our project is to train a state of the art model for lithuanian ASR.

Modern ASR models usually start out from a base trained audio attention model (Wav2vec2 or similar) and are then fine-tuned on pairs of audio and text for ASR.
Apart from text and audio pairs, a vocabulary file is needed for the language (Lithuanian)

Our vocabulary uses the letters in the lithuaninan alhapbet.
a	ą	b	c	č	d	e	ę	ė	f	g	h	i	į	y	j	k	l	m	n	o	p	r	s	š	t	u	ų	ū	v	z	ž

## Steps
0. Download common voice (done)
1. Reproduce training in Huggingface (done) WER 0.6546
1.5 Prepare for fairseq training (done)
2. Reproduce training in Fairseq (doing)
3. Find additional data sources (doing)
4. Augment common voice data using known augmentation techniques (found in the augmentation-folder) (doing)
5. Train with a larger model size.
6. Achieve State of the Art.
7. Release model on Huggingface.
8. Celebrate with cake


## Fun things to do

## Find more data sources
https://github.com/liepa-project

### Train a base model in Lithuanian.
We need a lot of data in Lithuanian.

## Start training from a similar language
Start from a base trained model in Latvian?

## Train a joint model on similar languages
