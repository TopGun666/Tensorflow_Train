# -*- coding: UTF-8 -*-

from __future__ import print_function
'''
从python2.1开始以后, 当一个新的语言特性首次出现在发行版中时候, 如果该新特性与以前旧版本python不兼容, 则该特性将会被默认禁用. 如果想启用这个新特性, 
则必须使用 "from __future__import *"

在开头加上from __future__ import print_function这句之后，即使在python2.X，使用print就得像python3.X那样加括号使用。python2.X中print不
需要括号，而在python3.X中则需要。

如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容，也就是它在该版本中不是语言标准，那么我如果想要使用的话就需要从future模块导入。
'''

import tensorflow as tf

# Simple helloworld using Tensorflow

# Create a Constant op
# The op is added as a node to the default graph
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, Tensorflow!')

# Start tf session
sess = tf.Session()

# Run the op
print(sess.run(hello))
