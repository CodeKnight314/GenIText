apt update
apt install lshw tmux unzip
(curl -fsSL https://ollama.com/install.sh | sh && ollama serve > ollama.log 2>&1) &
export OLLAMA_HOST=0.0.0.0