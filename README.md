 Traffic Flow Prediction Model for Mining Implicit Associations in the Dynamic Diffusion Process of Traffic Flow
 
 软件框架基于 PyTorch 2.5.1，使用 Python 3.9.23 作为编程语言。
 
 数据集可从 [BasicTS](sslocal://flow/file_openurl=https%3A%2F%2Fgithub.com%2FGestaltCogTeam%2FBasicTS%2Ftree%2Fmaster%2Fdatasets&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=) 下载数据集，将PEMS03, PEMS04, PEMS08三个数据集放到data/data/文件路径下。
 
 首先运行utils/creat_laplace.py生成laplace矩阵。
 
 之后运行main.py文件。
