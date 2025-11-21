# DQuantize

A Python research project for quantization research.

## Project Structure

```
DQuantize/
├── dquantize/          # Main source code
├── scripts/            # Utility and experiment scripts
├── third_party/       # Third-party code and dependencies
└── experiment_notes.md # Research notes and experiment documentation
```

## Setup

### Prerequisites

Install `uv` (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or using pip:

```bash
pip install uv
```

### Initial Setup

Initialize the project and create a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies

Install project dependencies:

```bash
uv pip install -r requirements.txt
```

Or if using `pyproject.toml`:

```bash
uv sync
```

### Development

For development, install the package in editable mode:

```bash
uv pip install -e .
```

## Usage

[Add usage instructions here]

## Experiments

See `experiment_notes.md` for detailed experiment documentation and results.

## License

[Add license information here]

## Contributing

[Add contributing guidelines here]


## Resources 
Post training quantization techniques :  
* SmoothQuant : https://www.youtube.com/watch?v=U0yvqjdMfr0 
* AWQ : https://www.youtube.com/watch?v=3dYLj9vjfA0 
* GPTQ : https://www.youtube.com/watch?v=OKpSgL9oMWU

