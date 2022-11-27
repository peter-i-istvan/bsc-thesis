# bsc-thesis
My BSc thesis that compares Convolutional and Transformer-based object detectors, and their usage in multi-object tracking.

## Submodule management

When cloning, call
```
git clone --recursive git@github.com:peter-i-istvan/bsc-thesis.git
```
You can move the commit pointer of the submodule to commit X by calling `git checkout X` inside the submodule folder, then `git commit` outside the submodule folder.

If the latest submodule commit pointer is advanced from outside, call 
```
git submodule update
```
When pulling:
```
git pull --recurse-submodules
```