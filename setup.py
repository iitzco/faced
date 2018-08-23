from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='faced',
      version='0.1',
      description='Face detection using deep learning',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='face detection deep learning cnn neural network',
      url='http://github.com/iitzco/faced',
      author='Ivan Itzcovich',
      author_email='i.itzcovich@gmail.com',
      license='MIT',
      packages=['faced'],
      scripts=["bin/faced"],
      install_requires=required,
      python_requires='>3.4',
      include_package_data=True,
      zip_safe=False)
