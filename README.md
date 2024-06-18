# 三体模拟器

该项目是一个三体模拟器，使用Python编写，并使用Pygame库进行可视化。它包含多个文件，每个文件都有其特定的功能和用途。

## 文件结构

- `Connection.py`：定义了连接类型，包括弹簧、阻尼器、绳索、排斥力和引力。
- `Field.py`：连接了`Connection`模块的内容，简化了场的导入。
- `Force.py`：定义了力的作用方式。
- `gpthelp.py`：记录日志文件。
- `Matrix.py`：定义了一个自定义的矩阵类，用于存储对象之间的连接信息。
- `misc.py`：包含了一些辅助函数。
- `Object.py`：定义了物体类，表示模拟中的物体。
- `phy.py`：主程序文件，定义了不同的模拟场景。
- `Show.py`：定义了显示类，负责绘制和更新屏幕内容。
- `System.py`：定义了系统类，管理模拟中的所有对象及其交互。

## 安装

1. 克隆仓库
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. 安装依赖
    ```sh
    pip install -r requirements.txt
    ```

## 使用方法

在终端中运行以下命令来启动模拟：
```sh
python phy.py
```

可以选择运行不同的模拟场景，如弹簧阻尼系统、三体问题等。修改`phy.py`中的`__main__`函数来选择不同的模拟场景。

## 类和函数介绍

### Connection.py

定义了以下类：

#### Spring
- `__init__(self, k, L)`: 初始化弹簧，参数为弹性系数`k`和原长`L`。
- `update(self, obj1, obj2)`: 更新弹簧力，计算并返回两个物体之间的弹簧力。

#### Damper
- `__init__(self, f)`: 初始化阻尼器，参数为阻尼系数`f`。
- `update(self, obj1, obj2)`: 更新阻尼力，计算并返回两个物体之间的阻尼力。

#### Rope
- `__init__(self, L)`: 初始化绳索，参数为绳索的最大长度`L`。
- `update(self, obj1, obj2)`: 更新绳索力，计算并返回两个物体之间的绳索力。

#### Repulsion
- `__init__(self, k, L)`: 初始化排斥力，参数为排斥系数`k`和临界距离`L`。
- `update(self, obj1, obj2)`: 更新排斥力，计算并返回两个物体之间的排斥力。

#### Gravity
- `__init__(self, G)`: 初始化引力，参数为引力常数`G`。
- `update(self, obj1, obj2)`: 更新引力，计算并返回两个物体之间的引力。

### Field.py

连接了`Connection`模块的内容，简化了场的导入：

```python
import Connection

Spring = Connection.Spring
Damper = Connection.Damper
Rope = Connection.Rope
Repulsion = Connection.Repulsion
Gravity = Connection.Gravity
```

### Force.py

定义了以下类：

#### Force
- `__init__(self, F)`: 初始化力，如果`F`是一个可调用对象，则直接使用，否则转换为一个常量力函数。
- `update(self, obj)`: 更新力，计算并返回作用在物体上的力。

### gpthelp.py

生成日志文件。

### Matrix.py

定义了自定义的矩阵类：

#### Matrix
- `__init__(self, n)`: 初始化矩阵，创建一个大小为`n`的矩阵。
- `__getitem__(self, ind)`: 获取矩阵中指定位置的元素。
- `__setitem__(self, ind, val)`: 设置矩阵中指定位置的元素。

### misc.py

包含辅助函数：

- `np_ndarray(a)`: 将输入`a`转换为numpy数组。

### Object.py

定义了物体类：

#### Object
- `__init__(self, m, pos, v)`: 初始化物体，参数为质量`m`、位置`pos`和速度`v`。
- `update(self, F, dt)`: 更新物体状态，计算新的加速度、速度和位置。

### phy.py

包含主程序和不同的模拟场景：

- `spr_dam()`: 模拟弹簧阻尼系统。
- `threeStar()`: 模拟三体问题。
- `f(obj)`: 返回一个函数，用于计算物体的抛物线轨迹。
- `Projectile()`: 模拟抛物线运动。

### Show.py

定义了显示类：

#### Show
- `__init__(self, win_size, caption, center=None)`: 初始化显示窗口，参数为窗口大小`win_size`、标题`caption`和中心位置`center`。
- `drawgrid(self)`: 绘制网格。
- `updategrid(self, k1, k2)`: 更新网格。
- `updatek(self, spos)`: 更新缩放系数`k`。
- `update(self, sys, msg=None)`: 更新显示，绘制物体位置和力。
- `drawFunction(self, f, t)`: 绘制函数轨迹。
- `postranslate(self, poss, k=None)`: 转换坐标。

### System.py

定义了系统类：

#### System
- `__init__(self, objs, forces=[], fields=[], connections=[])`: 初始化系统，参数为对象`objs`、力`forces`、场`fields`和连接`connections`。
- `update(self)`: 更新系统状态，计算每个对象所受的力并更新其位置和速度。
- `objpos(self)`: 获取所有对象的位置列表。
- `objFs(self)`: 获取所有对象的力列表。

## 贡献

欢迎提交issue和pull request来帮助改进本项目。如果有任何问题或建议，请随时联系。

## 许可证

此项目基于MIT许可证开源。详细信息请参阅LICENSE文件。