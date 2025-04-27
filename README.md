An OpenAI-compatible API for Dia text-to-speech, providing a drop-in replacement for applications using OpenAI's TTS service.

## Features

- 🎯 **OpenAI API Compatibility**: Implements the OpenAI `/audio/speech` endpoint for seamless integration with existing applications.
- 🔊 **Dia 1.6B Model**: Uses Nari Lab's [Dia 1.6B](https://github.com/nari-labs/dia) text-to-speech model for high-quality voice synthesis.
- 🎭 **Voice Customization**: Create and manage custom voices by uploading audio samples for voice cloning.
- 🗣 **Multi-speaker Support**: Use the [S1] and [S2] tags familiar to Dia users for multi-speaker dialogues.
- 📊 **Optimized Performance**: GPU acceleration with CUDA support for faster inference.
- 🐳 **Docker Ready**: Easy deployment using Docker with NVIDIA GPU support.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) for GPU acceleration

### Installation & Startup

1. **Clone the repository**:

```bash
git clone https://github.com/phildougherty/dia_openai.git
cd dia_openai
```

2. **Start the service with Docker Compose**:

```bash
docker-compose up -d
```

3. **Access the API**:

The API will be available at: `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## API Usage

### Text-to-Speech (OpenAI Compatible)

```python
import requests
import json

url = "http://localhost:8000/v1/audio/speech"

payload = {
    "model": "dia-1.6b",
    "input": "[S1] Hello, I'm speaking with the Dia TTS model. [S2] And I'm responding with a different voice.",
    "voice": "alloy",  # Standard voices: alloy, echo, fable, onyx, nova, shimmer
    "response_format": "mp3"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

# Save the audio file
if response.status_code == 200:
    with open("output.mp3", "wb") as file:
        file.write(response.content)
    print("Audio saved as output.mp3")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### Working with Custom Voices

#### Creating a Custom Voice

```python
import requests

url = "http://localhost:8000/v1/audio/voices"

# Prepare multipart form data
files = {
    'file': ('sample.wav', open('path/to/voice_sample.wav', 'rb'), 'audio/wav')
}
data = {
    'name': 'My Custom Voice',
    'description': 'A custom voice created from my audio sample'
}

response = requests.post(url, files=files, data=data)

if response.status_code == 201:
    voice_data = response.json()
    voice_id = voice_data['voice_id']
    print(f"Created custom voice with ID: {voice_id}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

#### Using a Custom Voice

```python
import requests

url = "http://localhost:8000/v1/audio/speech"

payload = {
    "model": "dia-1.6b",
    "input": "This is my custom voice speaking through the Dia model.",
    "voice": "custom_abc123def", # Replace with your actual custom voice ID
    "response_format": "mp3"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

# Save the audio file
if response.status_code == 200:
    with open("custom_voice_output.mp3", "wb") as file:
        file.write(response.content)
    print("Custom voice audio saved")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## Configuration

The application can be configured using environment variables in the `docker-compose.yml` file:

```yaml
environment:
  - HOST=0.0.0.0
  - PORT=8000
  - DEBUG=false  # Set to true for more logs
  - ENABLE_CORS=true
  - USE_TORCH_COMPILE=true  # Set to false for better compatibility on some systems
  - COMPUTE_DTYPE=float16  # Options: float16, bfloat16, float32
  - OUTPUT_FORMAT=mp3
  - MAX_AUDIO_LENGTH_SEC=60
```

## Advanced Configuration

### Customizing Model Paths

If you have a custom model version, you can mount it by modifying the `docker-compose.yml`:

```yaml
volumes:
  - ./my_custom_model:/app/models/dia-1.6b
```

### Performance Tuning

For better performance on GPUs with limited VRAM, try the following settings:

```yaml
environment:
  - COMPUTE_DTYPE=float16
  - USE_TORCH_COMPILE=false
```

For maximum quality but higher VRAM usage:

```yaml
environment:
  - COMPUTE_DTYPE=float32
  - USE_TORCH_COMPILE=true
```

## Project Structure

```
dia_openai/
├── app/                # Main application code
│   ├── api/            # API routes and schemas
│   ├── core/           # Core configuration
│   ├── models/         # Data models
│   ├── services/       # Business logic services
│   └── utils/          # Helper utilities
├── cache/              # Runtime cache directory
├── voices/             # Custom voice metadata storage
├── static/             # Static files and audio samples
├── tests/              # Test cases
├── docker-compose.yml  # Docker Compose configuration
└── Dockerfile          # Docker build instructions
```

## Troubleshooting

### Common Issues

- **CUDA out of memory**: Try lowering `MAX_AUDIO_LENGTH_SEC` or using `COMPUTE_DTYPE=float16`
- **Slow first-time speech generation**: The DAC model is being downloaded; subsequent calls will be faster
- **Audio quality issues**: Try increasing the `cfg_scale` parameter in API calls for better quality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Nari Labs](https://github.com/nari-labs) for creating the Dia model
- [OpenAI](https://openai.com) for the API design this project emulates

---

*Note: This is an unofficial project and is not affiliated with Nari Labs or OpenAI.*# dia_openai
