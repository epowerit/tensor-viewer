# TensorViewer

[![CI](https://github.com/epowerit/tensor-viewer/actions/workflows/ci.yml/badge.svg)](https://github.com/epowerit/tensor-viewer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/epowerit/tensor-viewer)

An IDE for tensors. Run real PyTorch and watch every tensor it makes: each step's shapes, axes, and values on a canvas you can play back, with the code beside it. Ask what changes if one input value were different or the model trained a few steps, trace a NaN to where it began, and see which inputs and weights a prediction leans on.

TensorViewer runs on your own machine. A FastAPI backend executes your PyTorch code and records every operation, and a React frontend lets you explore the recording.

<img width="2056" height="1178" alt="Screenshot 2026-10-04 at 5 37 59 AM" src="https://github.com/user-attachments/assets/e1d8c248-a037-4af4-8508-54e6da1de3e1" />

## What you can do

- **See every step.** Each operation becomes a card on a canvas, with its result's shape, axes, and values. Play the run through, step into it like a debugger, set breakpoints on lines, and fold loops and layers so a whole transformer reads at a glance. The editor shows each line's result as you write.
- **Read the axes, not just the numbers.** Every axis of the input keeps its color wherever it goes, through reshapes, splits into heads, and merges. Shapes can read in the input's own sizes (`[B, T, 48]`), the sentence's words label token axes, and any layout change can be read as the `einops` call it amounts to.
- **Look through lenses.** Color the whole model by spread, zeros, change since the last run, compute (FLOPs), memory, live memory with its peak, broadcast reuse, or gradient size, and project compute and memory to larger batch and sequence sizes.
- **Ask what if.** Change one input value, run in bfloat16, train the weights a few steps on a chosen value or on the whole sentence, or knock out one step's result (a head, a word) or patch it in from another run, all without saving, and see what changed at every step. A sweep knocks out each head or word in turn and ranks which ones the output leans on.
- **Read every layer.** The logit lens shows what a language model would predict after each block, so you can see the layer where a prediction forms. Causal tracing patches each layer and position in from another run and shows where a changed word's effect lives. The map draws each word's vector on its two main directions, and its path across layers.
- **Find out why.** Trace a NaN or an infinity back to the step that made it, see which input cells move a value, read run notes that flag dead units, saturated softmaxes, and numbers near float limits, and apply checked one-line fixes when a run fails.
- **Watch it like a debugger.** Watch expressions evaluate Python over the recorded tensors at the current step, draw themselves across the whole run, and can pause playback where a condition holds.
- **Keep what you found.** Save any run as a pytest file that rebuilds its inputs and checks every module's output shapes and the model's result, with only PyTorch needed.
- **Inspect the weights.** Every weight's norm and singular-value spectrum, its condition and effective rank, and how much each one changed after training, and in how many directions.

It comes with a library of 24 models, from tensor shapes and broadcasting to complete GPT, LLaMA-style, BERT-style, Vision Transformer, Swin, CLIP, DETR, Perceiver, and image-captioning models. You can also paste or upload an `nn.Module`, write a few lines in the console, build a model from components on a canvas, or import a Git repository.

## Quick start

Requirements: Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), and Node.js 22 or newer with npm. Each project has its own lockfile.

In one terminal:

```sh
cd backend
uv sync --locked --python 3.11
uv run uvicorn tensorviewer.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
cd frontend
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. The first `uv sync` downloads PyTorch, which is large (on Linux, with its CUDA libraries); later runs reuse it.

**No install:** the **Open in GitHub Codespaces** button above starts both servers in a cloud machine of your own and opens the app on port 5173. The first start takes a few minutes while PyTorch installs.

> **Security.** TensorViewer runs the Python code you give it, with your user's permissions, so it is meant for your own machine. The backend listens on `127.0.0.1` only. Do not bind it to `0.0.0.0` or expose its port to a network, and run projects only from sources you trust: a public server would let anyone run code on it.

## Learn more

- **[How it works](docs/how-it-works.md)** is the full manual: every view, lens, and tool, each with an example from the library.
- **[Architecture](docs/architecture.md)** covers how sources are read, how execution is traced, and how the frontend presents it.

## Contributing

Issues and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to set up both projects, the checks a change should pass, and the conventions the code follows.

## License

TensorViewer is released under the [MIT License](LICENSE).
