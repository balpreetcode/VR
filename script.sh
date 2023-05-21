#!/bin/bash

# Clone whisper-jax repository
git clone https://github.com/sanchit-gandhi/whisper-jax.git
cd whisper-jax
pip install -r requirements.txt
pip install numpy scipy jax jaxlib flax optax transformers
cd ..

# Clone whisper_real_time repository
git clone https://github.com/davabase/whisper_real_time.git
cd whisper_real_time/
pip install -r requirements.txt
sudo apt-get install -y portaudio19-dev
pip install pyaudio
pip install -r requirements.txt
python3.7 transcribe_demo.py
cd ..

# Clone whisper repository
git clone https://github.com/openai/whisper.git
cd whisper
pip install -U openai-whisper
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
sudo apt update && sudo apt install ffmpeg

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

pip install setuptools-rust

# Convert audio files using whisper
whisper audio.mp3 --model medium

# Set up whisper_api
cd ..
cd whisper_api/
touch app.py
mkdir uploads
pip install flask flask-restful
python app.py

# Install ffmpeg and verify installation
sudo apt-get update
sudo apt-get install ffmpeg
ffmpeg -v

# Transcribe audio using whisperx
whisperx audio.mp3 --compute_type float32

# Install dependencies for whisper-jax
cd ..
cd whisper-jax/
pip install git+https://github.com/deepmind/dm-haiku
pip install git+https://github.com/sanchit-gandhi/whisper-jax
pip install git+https://github.com/deepmind/optax
pip install -e .["endpoint"]

# Run the Gradio app
python app/app.py

# Save command history to a file
history > command_history.txt
