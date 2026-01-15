
# - install uv if not
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -p 3.12

# - install ffmpeg
# sudo apt-get install -y ffmpeg
# brew install ffmpeg

uv init
uv add 'audio-separator[cpu]' gradio

# show info
uv run --project ./ audio-separator --env_info

# copy model dir to ./models
# or you can just run the script, and it will download it once the network is accessible.