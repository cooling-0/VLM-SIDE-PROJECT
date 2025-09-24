# CLIP Mini-Project — Docker (CPU)

## Build
```bash
docker build -t clip-lab:cpu -f Dockerfile .
```

## Run (mount current project)
From your project root (which has `notebooks/`, `data/`, `out/`), run:
```bash
docker run --rm -it   -p 8888:8888   -v "${PWD}:/workspace"   clip-lab:cpu
```
Then open: http://localhost:8888/

## Notes
- Place images in `data/images/` before running the notebook.
- Open `notebooks/01_clip_intro.ipynb` and Run All.
- Outputs will be saved to `out/` on your host.
- If you need GPU later, we can add a CUDA-based image separately.
