```bash
Image-Classifier/
├── assets/ # Not required for general usage
│   ├── Flowers.png 
│   └── inference_example.png
├── data/
│   ├── train/ # Install with wget from local-setup.sh
│   ├── valid/ # Install with wget from local-setup.sh
│   ├── test/ # Install with wget from local-setup.sh
│   └── cat_to_name.json
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   ├── utils.py
│   ├── data_loader.py
│   └── model.py
├── checkpoints/
│   └── [Saved checkpoint files]
├── scripts/
│   └── setup.sh
├── env.yaml
├── README.md
└── .gitignore
```


# TO-DO LIST

1. Docstrings
2. Module strings
3. Other comments
4. Jupyter notebook reqs
5. Jupyter notebook doc
6. Other doco
7. Fix setup scripts (auto activate and output alignment)