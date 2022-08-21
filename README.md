### Transformer-based Affine Registration Framework
This repository is based on [DeepHistReg framework](https://github.com/MWod/DeepHistReg) and uses [LoFTR](https://github.com/zju3dv/LoFTR) with [QuadTree Attention](https://github.com/tangshitao/quadtreeattention) for affine registration.

### How to run registration:

Clone repo and cd:

```
git clone https://github.com/VladPyatov/ImgRegWithTransformers.git
```

```
cd ImgRegWithTransformers
```

Create and activate virtual environment:

```
python3 -m venv env
```

```
source env/bin/activate
```

Install dependencies from requirements.txt:

```
python3 -m pip install --upgrade pip
```

```
pip install -r requirements.txt
```

Install QuadTree Attention module:

```
cd ./networks/QuadTreeAttention && python setup.py install
cd ../../
```

Configure main.py and run registration:

```
python3 main.py
```
