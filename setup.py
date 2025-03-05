import setuptools

setuptools.setup(
    name="bbbrisk", 
    version = "1.0.2", 
    author="bbbdata", 
    author_email="yhbbbdata@163.com", 
    description="model for loan risk controll,i.e. score card", 
    long_description=open('README.md', encoding = 'utf-8').read(),
    long_description_content_type="text/markdown", 
    packages=setuptools.find_packages(), # 自动找到项目中导入的模块
    
    package_data={
        'bbbrisk.datasets': ['*.csv'] 
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    keywords =['scorecard'],
    install_requires=['numpy','pandas','scikit-learn>=0.18'],
    python_requires='>=3.7',
    url="https://github.com/bbbdata/bbbrisk", # github地址
)
