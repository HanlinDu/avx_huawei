```
有关原工程文件和实验说明，参见 https://github.com/Makiras/leisurely-matrix，作者为Makiras Yun-long Xie
```

# x86向量扩展指令AVX监测实验

## 上述实验环境

```
略
```

## 如何使用

### 环境配置

#### Cilk

Link: https://github.com/OpenCilk/opencilk-project/releases/tag/opencilk/v1.0  
Extract To: /opt/OpenCilk-10.0.1-Linux

> If not installation is not in /opt, please modify the Makefile


#### Intel MKL (Part of Intel oneAPI，暂不需要)

Page: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html

- Plz Follow the installation guide to install Intel oneAPI.

> If not installation is not in /opt/intel/oneapi/mkl/2022.1.0/, please modify the Makefile.

### 使用

0. 依据指示修改Makefile中的依赖库路径（Ubuntu和CentOS大概率不用更改）。
1. 修改代码中的空缺部分。
2. `make cilk_avx` 运行某部分代码（请注意依照**代码中已有的宏**填写**循环及阈值参数**）
3. `make` 运行所有测试

> Tips:  
> 1. `make MSIZE=n` 会将矩阵大小设置为n*n。可以使用128进行调试，1024或4096等进行结果生成。
> 2. `make > log` 将会把输出重定向到log文件中，方便查看结果。
> 3. `make clean` 会清除所有生成的文件。
> 4. `make cilk_avx` 将会只运行你指定的修改部分，建议了解简单的Makefile语法。  

