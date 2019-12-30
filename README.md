# GMM_MATLAB
this demo use two lever of GMM model to do classification task, the first level uses 4 kernel GMM model and second level uses 16 kernel GMM.

##### #step 1: Do DCT transformer to covert image info to 18 dimension vector

```matlab
path = "\your\path\to\GMM\dataset\"
GMM4(path, 'train', 'class1')
GMM4(path, 'train', 'class2')
GMM4(path, 'test', 'class1')
GMM4(path, 'test', 'class2')
```

after this operation, in each data directory will contain a CSV file named as **GMM4.csv**

##### #step 2: Do GMM 16 kernel training

```matlab
GMM16(path, 'class1')
GMM16(path, 'class2')
```

​	After this operation, in each train category will contain a CSV file named as **GMM16.csv**

##### #step 3: Make final decision

```matlab
Decide4_16(path, 'class1', 'class2'); 
```

​	After this operation, in the dataset directory will contain a CSV file named as **Result of class1 vs class2.csv**

##### #Tricks for train

- if your dataset is very large(image size, image quantity), try to resize your image to suitable size (recommend about 200 * 200 pixel, 70~100 images per category)

- if the speed in step 1 is very slow, try to modify the parameter larger, then do it again with smaller parameter

    ```matlab
       if abs(sum(log(tmpGM)) - L) > 1e-15    % 似然函数增量判决(终止条件判决)  
    ```

- if your want to make your classification task more precise, try to use 8/64 kernel GMM model by simple adjusting the 4/16 kernel GMM model.
