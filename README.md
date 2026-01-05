---
title: Animal Extinct Species Classifier
emoji: 🦁
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000 
pinned: false
---

# Animal Extinction Classifier

A full-stack Deep Learning application that identifies 11 different animal species. Built with **GoogLeNet**, **ONNX Runtime**, and **FastAPI**.

<p align="center">
  <img src="screenshot.png" alt="Application Screenshot" width="800">
  <br>
  <a href="https://huggingface.co/spaces/zayesosa/animal-extinct-species">Hugging Face Live Demo</a>
</p>


## 🌟 Features

- **Instant Classification**: High-accuracy identification of 11 species using a fine-tuned GoogLeNet model.
- **User Interface**: A friendly and intuitive UI designed to educate users on extinction risks and help document species at risk.

- **Multi-Input Support**:
  - 📁 **Upload**: Process local images directly from your device.
  - 🔗 **URL**: Paste a direct image link to classify images from the web.
- **Modern UI**: Clean, professional interface built with **Tailwind CSS**.
- **Performance Optimized**: Model exported to **ONNX** format for faster inference and lower memory usage.
- **Production Ready**: Fully containerized using **Docker** and managed with the **uv** package manager.

## 🐾 Supported Species

The model is trained to recognize:
_African Elephant, Amur Leopard, Arctic Fox, Chimpanzee, Jaguars, Lion, Orangutan, Panda, Panthers, Rhino, and Cheetahs._

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch (GoogLeNet)
- **Inference Engine**: ONNX Runtime
- **Backend**: FastAPI (Python 3.11)
- **Frontend**: Tailwind CSS & Vanilla JavaScript
- **Dependency Management**: `uv`
- **Containerization**: Docker

## 🚀 Installation & Setup

### Using Docker (Recommended)

1. Clone the repository:

```bash
   git clone [https://github.com/cyrexez/animal-classif.git](https://github.com/cyrexez/animal-extinct-species.git)
   cd animal-classifier
```

2. Build the Docker image:

```bash
   docker build -t animal-classifier .
```

3. Run the Container:

```bash
   docker run -p 8000:8000 animal-classifier
```

4. Open your browser to http://localhost:8000

Local Development (Using uv)

1. Sync dependencies:

```bash
uv sync
Run the server:
```

2. Run the server

```bash
uv run uvicorn app:app --reload
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Dataset: [Danger Of Extinction animal image set](https://www.kaggle.com/datasets/brsdincer/danger-of-extinction-animal-image-set)

Project was insipired by the general IUCN Red List classifications.

```

```
