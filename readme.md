# MORF

## Requirements

```
Python==3.6
pandas==1.4.3
numpy==1.21.0
scikit-learn==1.1.1
scipy==1.9.1
```

## Usage

### Choose method

In main.py, change the method_type among these four items, mvhh(n), mvhh(n)_bag, mvhh(r) and mvhh(r)_bag.

### Choose combinations of views

Firstly, in function load_data, change the corresponding block of code for one, two, three, or four views.  Secondly, in function changeData, repeat the same operation.

### Training the MORF

Run the main.py to perform the MORF.

```
python main.py
```

## Get results

The accuracy and F1-score of MORF can be seen in the log_May-10.txt

